# Set error action to stop on error
$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "Starting SDXL Generation Task..."
Write-Host "========================================"

# Check if python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Ensure we are in the script's directory (Project Root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Construct the path to the main python script
$MainScriptPath = Join-Path "src" "main.py"

# Execute the python script
try {
    python $MainScriptPath
}
catch {
    Write-Error "An error occurred during execution."
    Write-Error $_
}

Write-Host "========================================"
Write-Host "Done."
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")