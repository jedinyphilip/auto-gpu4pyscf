#Requires -Version 5.1
<#
.SYNOPSIS
    Build gpu4pyscf from source for the GPUs in this machine. Windows twin of
    build.sh -- same Dockerfile, same build args, the container is Linux either way.
.EXAMPLE
    .\build.ps1
.EXAMPLE
    .\build.ps1 -CudaArch 90-real -SkipTest
#>
[CmdletBinding()]
param(
    [string]$CudaArch,                                  # force a target, e.g. 120-real
    [string]$Ref = 'master',                            # branch, tag or SHA
    [string]$Image = 'gpu4pyscf:local',
    [string]$CudaImage,                                 # override the build-stage base image
    [string]$RuntimeImage,                              # override the final-stage base image
    [switch]$KeepToolkit,                               # keep nvcc + sources (~3x bigger)
    [ValidateSet('ON', 'OFF')][string]$BuildLibxc = 'OFF',
    [switch]$SkipTest,
    [switch]$DryRun,
    [switch]$KeepCache,                                 # keep the build cache warm
    [switch]$Force,                                     # build despite the disk check
    [int]$NeedGb = 25,
    [string]$Builder = 'gpu4pyscf-build'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)  # repo root: the build context
$probeImage = 'nvidia/cuda:12.8.2-base-ubuntu24.04'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'docker not found. Install Docker Desktop and enable the WSL2 backend.'
}

# Anything this script pulls, this script removes. Note what was already here.
docker image inspect $probeImage 2>$null | Out-Null
$probePreAbsent = ($LASTEXITCODE -ne 0)

# How does this host hand GPUs to a container? --gpus needs the nvidia hook on
# the default runtime; where the toolkit only registered the 'nvidia' runtime it
# is rejected outright, and CDI setups want a device name instead.
function Get-GpuArgs {
    param([string]$TestImage)
    $candidates = @(
        @('--gpus', 'all'),
        @('--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=all'),
        @('--device', 'nvidia.com/gpu=all')
    )
    foreach ($c in $candidates) {
        docker run --rm @c $TestImage true 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return , $c }
    }
    return $null
}

# --- 1. what hardware is this? ------------------------------------------------
# The host driver first (nvidia-smi.exe ships with the Windows driver), then a
# throwaway container as a fallback.
function Get-ComputeCaps {
    $query = @('--query-gpu=compute_cap', '--format=csv,noheader')
    $out = $null
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $out = & nvidia-smi @query 2>$null
    }
    if (-not $out) {
        $probeGpuArgs = Get-GpuArgs $probeImage
        if ($probeGpuArgs) {
            $out = & docker run --rm @probeGpuArgs $probeImage nvidia-smi @query 2>$null
        }
    }
    return @($out | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+\.\d+$' })
}

if ($CudaArch) {
    $caps = @([regex]::Matches($CudaArch, '\d+') | ForEach-Object { [int]$_.Value })
} else {
    $raw = Get-ComputeCaps
    if (-not $raw) {
        Write-Host 'error: could not read a compute capability from any GPU.' -ForegroundColor Red
        Write-Host '       Docker Desktop must use the WSL2 backend -- GPUs are not'
        Write-Host '       exposed under the Hyper-V backend. Check that "nvidia-smi"'
        Write-Host '       runs, or pass -CudaArch 120-real to build blind.'
        # An AMD/Intel-only box should hear why, not just "detection failed".
        $others = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name | Where-Object { $_ -notmatch 'NVIDIA' })
        if ($others) {
            Write-Host "       Non-NVIDIA GPUs present:"
            $others | ForEach-Object { Write-Host "         $_" }
            Write-Host "       gpu4pyscf is CUDA-only, so it cannot use them -- see the"
            Write-Host "       'Non-NVIDIA GPUs' section of the README."
        }
        exit 1
    }
    $caps = @($raw | ForEach-Object { [int]($_ -replace '\.', '') } | Sort-Object -Unique)
    $CudaArch = ($caps | ForEach-Object { "$_-real" }) -join ';'
}
$maxCap = ($caps | Measure-Object -Maximum).Maximum

# --- 2. a toolkit that can target it ------------------------------------------
# sm_103 (B300) and sm_121 (RTX PRO / GB10) landed in CUDA 12.9; everything else
# gpu4pyscf supports is covered by 12.8.
$tag = if ($maxCap -eq 103 -or $maxCap -ge 121) { '12.9.1' } else { '12.8.2' }
if (-not $CudaImage)    { $CudaImage = "nvidia/cuda:$tag-devel-ubuntu24.04" }
if (-not $RuntimeImage) { $RuntimeImage = "nvidia/cuda:$tag-runtime-ubuntu24.04" }
# Default image carries no toolkit: the compiler is only needed to produce the
# .so files, and dropping it saves roughly 8 GB.
$target = if ($KeepToolkit) { 'devel' } else { 'runtime' }
if ($maxCap -lt 70) {
    Write-Warning "compute capability $maxCap is below the 7.0 gpu4pyscf supports; the build may fail or the kernels may not run."
}

# --- 3. is there room? --------------------------------------------------------
# Docker Desktop's data lives in the WSL2 VM, which df cannot see from here, so
# use the drive that holds it as a proxy -- best effort, not exact.
$availGb = $null
try {
    $drive = Get-PSDrive -Name ($env:SystemDrive).Substring(0, 1) -ErrorAction Stop
    $availGb = [int]($drive.Free / 1GB)
} catch { }

# --- 4. pin the source so rebuilds are cache-correct --------------------------
$sha = $Ref
if (Get-Command git -ErrorAction SilentlyContinue) {
    $line = & git ls-remote https://github.com/pyscf/gpu4pyscf.git $Ref 2>$null | Select-Object -First 1
    if ($line) { $sha = ($line -split "`t")[0] }
}

Write-Host "GPUs            : compute capability $($caps -join ', ')"
Write-Host "CUDA_ARCH       : $CudaArch"
Write-Host "base image      : $CudaImage"
Write-Host "runtime image   : $RuntimeImage"
Write-Host "gpu4pyscf       : $Ref -> $sha"
Write-Host "image target    : $target$(if ($target -eq 'devel') { ' (keeps nvcc)' })"
$diskText = if ($availGb) { "$availGb GB" } else { 'unknown' }
$cacheText = if ($KeepCache) { 'kept (-KeepCache)' } else { "disposable builder '$Builder', removed on success" }
Write-Host "disk free       : $diskText, peak need ~$NeedGb GB"
Write-Host "build cache     : $cacheText"
Write-Host ''

if ($DryRun) { Write-Host '(DryRun) would build with the args above'; exit 0 }

if ($availGb -and $availGb -lt $NeedGb -and -not $Force) {
    Write-Host "error: only $availGb GB free on $env:SystemDrive; the build peaks near $NeedGb GB." -ForegroundColor Red
    Write-Host '       What docker is holding right now:'
    docker system df
    Write-Host '       Free some of that, or re-run with -Force (or -NeedGb <n>).'
    exit 1
}

# --- 5. build in a disposable builder, then throw it away ---------------------
# The docker-container driver keeps every intermediate layer, the 11 GB toolkit
# image and all the object files inside its own container and volume. Removing
# the builder therefore reclaims all of it without touching the build cache of
# any other project -- which a plain `docker builder prune` would.
$isolated = -not $KeepCache
$built = $false

function Test-BuilderExists {
    $names = docker buildx ls 2>$null | ForEach-Object { ($_ -split '\s+')[0] }
    return $names -contains $Builder
}

try {
    $buildArgs = @(
        '--build-arg', "CUDA_IMAGE=$CudaImage",
        '--build-arg', "RUNTIME_IMAGE=$RuntimeImage",
        '--build-arg', "CUDA_ARCH=$CudaArch",
        '--build-arg', "GPU4PYSCF_REF=$sha",
        '--build-arg', "BUILD_LIBXC=$BuildLibxc"
    )
    if ($isolated) {
        docker buildx create --name $Builder --driver docker-container 2>$null | Out-Null
        docker buildx build --builder $Builder --load -t $Image --target $target `
            @buildArgs -f docker/Dockerfile .
    } else {
        docker build -t $Image --target $target @buildArgs -f docker/Dockerfile .
    }
    if ($LASTEXITCODE -ne 0) { throw 'docker build failed' }
    $built = $true

    $size = '{0:N1} GB' -f ([double](docker image inspect $Image --format '{{.Size}}') / 1e9)
    Write-Host ''
    Write-Host "built $Image ($size)"
    if (-not $SkipTest) {
        Write-Host 'running smoke test...'
        $gpuArgs = Get-GpuArgs $Image
        if (-not $gpuArgs) {
            throw 'no working way to attach a GPU to a container on this host: tried --gpus, --runtime=nvidia and CDI. Check the NVIDIA container toolkit.'
        }
        Write-Host "gpu flags       : $($gpuArgs -join ' ')"
        docker run --rm @gpuArgs $Image smoke-test
        if ($LASTEXITCODE -ne 0) { throw 'smoke test failed' }
    }
    Write-Host ''
    Write-Host 'use it with:  .\run.ps1 python3 your_script.py'
    if ($isolated) {
        Write-Host 'everything else this build created is removed on exit;'
        Write-Host 're-run with -KeepCache if you would rather keep the cache warm.'
    }
}
finally {
    if ($isolated -and (Test-BuilderExists)) {
        if ($built) {
            $freed = docker buildx du --builder $Builder 2>$null |
                Where-Object { $_ -match '^Total:' } |
                ForEach-Object { ($_ -split '\s+')[1] }
            $freedText = if ($freed) { ", reclaiming $freed" } else { '' }
            Write-Host "cleaning up     : removing builder '$Builder'$freedText"
            docker buildx rm $Builder 2>$null | Out-Null
        } else {
            # A 40-minute build that died at minute 39 should not also lose its
            # object files; say what is left and how to drop it.
            Write-Warning "build cache kept in builder '$Builder' so a re-run resumes."
            Write-Warning "Drop it with: docker buildx rm $Builder"
        }
    }
    if ($probePreAbsent) { docker image rm $probeImage 2>$null | Out-Null }
}
