$ErrorActionPreference = "Stop"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}
if (!(Test-Path "models\best.pt")) {
    throw "models\best.pt not found."
}
if (!(Test-Path "data\demo.mp4")) {
    throw "data\demo.mp4 not found. Copy the prepared defense video first."
}

python -m src.detect `
  --source data\demo.mp4 `
  --model models\best.pt `
  --output outputs\videos\core_result.mp4 `
  --conf 0.25 `
  --history 12 `
  --min-votes 4 `
  --min-hits 3 `
  --no-ocr `
  --show
