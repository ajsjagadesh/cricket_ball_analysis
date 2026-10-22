$ErrorActionPreference = 'Stop'

. "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
python -m cricket_vision.cli check-env
