param(
    [string]$Python = '3.11',
    [string]$Report = 'reports/phase-1/squeezenet11-torchvision-onnx-cpu.json'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    if (-not (Test-Path .repro-venv\Scripts\python.exe)) {
        uv venv --python $Python .repro-venv
    }
    $python = (Resolve-Path .repro-venv\Scripts\python.exe).Path
    $env:PYTHONPATH = (Join-Path $root 'src')
    uv pip install --python $python --index-url https://pypi.org/simple -r scripts\requirements-reference-public.txt
    if ($LASTEXITCODE -ne 0) { throw "public dependency installation failed with exit code $LASTEXITCODE" }
    & $python scripts\export_torchvision_onnx.py
    if ($LASTEXITCODE -ne 0) { throw "public model export failed with exit code $LASTEXITCODE" }
    & $python scripts\run_torchvision_gate.py --report $Report
    if ($LASTEXITCODE -ne 0) { throw "CPU release gate failed with exit code $LASTEXITCODE" }
    & $python scripts\render_report.py $Report
    if ($LASTEXITCODE -ne 0) { throw "report rendering failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
