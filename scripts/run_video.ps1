param(
    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$Output = 'output/annotated.mp4'
)

$ErrorActionPreference = 'Stop'

. "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
python -m cricket_vision.cli analyze --video $Video --output $Output
