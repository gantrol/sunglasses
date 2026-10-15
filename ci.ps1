# Local CI uses a dedicated locked environment; existing .venv is untouched.
$ErrorActionPreference = 'Stop'
$ciArguments = @($args)
$ciExitCode = 1
$ciMutex = $null
$ciAcquired = $false
$ciPreviousEnvironment = $env:UV_PROJECT_ENVIRONMENT
$ciPreviousDownloads = $env:UV_PYTHON_DOWNLOADS
$ciPreviousPreference = $env:UV_PYTHON_PREFERENCE
Push-Location $PSScriptRoot
try {
    $bootstrapPython = (Get-Command python.exe -ErrorAction Stop).Source
    $ciScript = Join-Path $PSScriptRoot 'scripts\ci.py'
    if ($ciArguments -contains '--dry-run' -or $ciArguments -contains '--help' -or $ciArguments -contains '-h') {
        & $bootstrapPython -X utf8 $ciScript @ciArguments
        $ciExitCode = $LASTEXITCODE
    }
    else {
        # Validate arguments (including Store identity) before downloads or builds.
        $planOutput = & $bootstrapPython -X utf8 $ciScript @ciArguments --dry-run --plan-json
        if ($LASTEXITCODE) { throw 'Invalid CI arguments.' }
        $ciPlan = $planOutput | ConvertFrom-Json
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            $key = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($PSScriptRoot.ToLowerInvariant()))).Replace('-', '')
        } finally { $hasher.Dispose() }
        $ciMutex = New-Object System.Threading.Mutex($false, "Local\SunglassesCI-$key")
        try { $ciAcquired = $ciMutex.WaitOne(0) }
        catch [System.Threading.AbandonedMutexException] { $ciAcquired = $true }
        if (-not $ciAcquired) { throw 'Another CI run is using this workspace.' }
        Get-Command uv, node, ffmpeg, ffprobe -ErrorAction Stop | Out-Null
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $PSScriptRoot 'build\ci-env'
        $env:UV_PYTHON_DOWNLOADS = 'never'
        $env:UV_PYTHON_PREFERENCE = 'only-system'
        & uv sync --locked --group build --no-install-project --no-progress
        if ($LASTEXITCODE) { throw 'Locked CI dependency sync failed.' }
        $ciPython = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
        if ($ciPlan.build) {
            # Reuse the existing VS discovery. CI launches only the Python runner;
            # native compilation remains a stage after the inexpensive checks.
            & (Join-Path $PSScriptRoot 'dev.ps1') ci-toolchain @ciArguments
        }
        else {
            & $ciPython -X utf8 $ciScript @ciArguments
        }
        $ciExitCode = $LASTEXITCODE
    }
}
catch {
    Write-Error $_ -ErrorAction Continue
}
finally {
    $env:UV_PROJECT_ENVIRONMENT = $ciPreviousEnvironment
    $env:UV_PYTHON_DOWNLOADS = $ciPreviousDownloads
    $env:UV_PYTHON_PREFERENCE = $ciPreviousPreference
    if ($ciAcquired) { $ciMutex.ReleaseMutex() }
    if ($ciMutex) { $ciMutex.Dispose() }
    Pop-Location
}
exit $ciExitCode
