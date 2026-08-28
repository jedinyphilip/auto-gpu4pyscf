#Requires -Version 5.1
<#
.SYNOPSIS
    Run something in the image with the GPUs attached and the current directory
    mounted at /work.
.EXAMPLE
    .\run.ps1                      # interactive shell
.EXAMPLE
    .\run.ps1 python3 my_calc.py   # run a script from the current directory
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)

$ErrorActionPreference = 'Stop'
$Image = if ($env:IMAGE) { $env:IMAGE } else { 'gpu4pyscf:local' }
if (-not $Command) { $Command = @('/bin/bash') }

# --gpus works only where the nvidia hook is on the default runtime; fall back
# to the explicit runtime, then to CDI.
$gpuArgs = $null
foreach ($c in @(@('--gpus', 'all'),
                 @('--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=all'),
                 @('--device', 'nvidia.com/gpu=all'))) {
    docker run --rm @c $Image true 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $gpuArgs = $c; break }
}
if (-not $gpuArgs) {
    throw 'no working way to attach a GPU to a container on this host: tried --gpus, --runtime=nvidia and CDI.'
}

docker run --rm -it @gpuArgs `
    -v "$($PWD.Path):/work" -w /work `
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/.cupy -e PYSCF_TMPDIR=/tmp `
    $Image @Command
