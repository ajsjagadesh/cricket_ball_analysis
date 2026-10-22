param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [string]$Output = "models/bowling_action.pt",
    [int]$Epochs = 40,
    [int]$HiddenSize = 32
)

$ErrorActionPreference = 'Stop'
python -m cricket_vision.cli train-action --data $Data --output $Output --epochs $Epochs --hidden-size $HiddenSize
if ($LASTEXITCODE -ne 0) { throw 'Action model training failed.' }
