#Requires -Version 5.1
<#
.SYNOPSIS
    Menu front end. Runs the package from the checkout, installed or not.
#>
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$PSScriptRoot\src;$env:PYTHONPATH"
} else {
    "$PSScriptRoot\src"
}

foreach ($candidate in @(@('python', @()), @('python3', @()), @('py', @('-3')))) {
    $exe = Get-Command $candidate[0] -ErrorAction SilentlyContinue
    if ($exe) {
        & $exe.Source @($candidate[1]) -m auto_gpu4pyscf @args
        exit $LASTEXITCODE
    }
}
Write-Host 'error: python 3 is required to run the menu (neither backend is).' -ForegroundColor Red
exit 1
