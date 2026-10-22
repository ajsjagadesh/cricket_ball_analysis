$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.venv')) {
    python -m venv .venv
}

. "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
