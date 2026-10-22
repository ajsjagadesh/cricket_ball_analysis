param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [string]$BaseModel = "yolo11n.pt",
    [int]$Epochs = 100,
    [int]$ImageSize = 1280
)

$ErrorActionPreference = 'Stop'
python -m cricket_vision.cli train-ball --data $Data --base-model $BaseModel --epochs $Epochs --imgsz $ImageSize
if ($LASTEXITCODE -ne 0) { throw 'Ball detector training failed.' }
