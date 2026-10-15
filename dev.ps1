# Forward arguments to the Python entry point without changing execution policy.
$ErrorActionPreference = 'Stop'
$devArguments = @($args)
if ($devArguments.Count -gt 0 -and $devArguments[0] -eq 'ci') {
    $ciArguments = @($devArguments | Select-Object -Skip 1)
    & (Join-Path $PSScriptRoot 'ci.ps1') @ciArguments
    exit $LASTEXITCODE
}
$ciToolchain = $devArguments.Count -gt 0 -and $devArguments[0] -eq 'ci-toolchain'
if ($ciToolchain) { $devArguments = @($devArguments | Select-Object -Skip 1) }
$controlArguments = @()
foreach ($argument in $devArguments) {
    if ($argument -eq '--') { break }
    $controlArguments += $argument
}

try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if ($ciToolchain) { $python = Join-Path $PSScriptRoot 'build\ci-env\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $python = (Get-Command python.exe -ErrorAction Stop).Source
    }

    $needsBuild = $devArguments.Count -gt 0 -and $devArguments[0] -in @('build', 'run', 'debug', 'package')
    if ($ciToolchain) { $needsBuild = $true }
    if ($controlArguments -contains '--help' -or $controlArguments -contains '-h' -or
        $controlArguments -contains '--dry-run' -or $controlArguments -contains '--no-build') {
        $needsBuild = $false
    }
    $toolchainReady = $env:VSCMD_ARG_TGT_ARCH -eq 'x64' -and
        (Get-Command cl.exe -ErrorAction SilentlyContinue) -and
        (Get-Command cmake.exe -ErrorAction SilentlyContinue) -and
        (Get-Command ninja.exe -ErrorAction SilentlyContinue)
    if ($needsBuild -and -not $toolchainReady) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
            throw 'Visual Studio C++ tools were not found. Install the Desktop development with C++ workload.'
        }
        $installations = @(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
        if ($LASTEXITCODE -ne 0 -or $installations.Count -eq 0) {
            throw 'No Visual Studio installation with the x64 C++ tools was found.'
        }
        $installation = $installations[0]
        $developerShell = Join-Path $installation 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
        Import-Module $developerShell
        Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
        foreach ($tool in @('cl.exe', 'cmake.exe', 'ninja.exe')) {
            if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
                throw "Missing $tool. Install the Visual Studio C++ desktop workload and C++ CMake tools."
            }
        }
    }

    $entrypoint = if ($ciToolchain) { 'scripts\ci.py' } else { 'scripts\dev.py' }
    & $python (Join-Path $PSScriptRoot $entrypoint) @devArguments
    exit $LASTEXITCODE
}
catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
