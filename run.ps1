# Set error action to stop on error
$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "Starting DeepSese Framework..."
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

# Execute the python script with all arguments passed to this ps1
# $args contains the parameters passed to run.ps1 (e.g., --t2i --nums 10)
try {
    # We use & operator to execute the command and pass $args array
    python $MainScriptPath $args
}
catch {
    Write-Error "An error occurred during execution."
    Write-Error $_
}

Write-Host "========================================"
Write-Host "Done."
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")