# Sampling Proxy Startup Script for Windows
# This script activates the virtual environment and starts the proxy server

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPath = Join-Path $ScriptDir "sampling-proxy"

# Check if virtual environment exists
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "Please create it first with:" -ForegroundColor Yellow
    Write-Host "  python -m venv sampling-proxy" -ForegroundColor Yellow
    Write-Host "  sampling-proxy\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

# Check if activation script exists
if (-not (Test-Path $ActivateScript)) {
    Write-Host "Activation script not found at: $ActivateScript" -ForegroundColor Red
    Write-Host "Virtual environment may be corrupted. Please recreate it." -ForegroundColor Yellow
    exit 1
}

# Activate the virtual environment
try {
    . $ActivateScript
    Write-Host "Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
} catch {
    Write-Host "Failed to activate virtual environment: $_" -ForegroundColor Red
    Write-Host "Note: You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process" -ForegroundColor Yellow
    exit 1
}

# Check if activation was successful
if ($env:VIRTUAL_ENV) {
    Write-Host "Starting Sampling Proxy..." -ForegroundColor Green
    
    # Pass all command line arguments to the Python script
    $ArgsList = $args
    if ($ArgsList.Count -gt 0) {
        & python sampling-proxy.py $ArgsList
    } else {
        & python sampling-proxy.py
    }
} else {
    Write-Host "Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}