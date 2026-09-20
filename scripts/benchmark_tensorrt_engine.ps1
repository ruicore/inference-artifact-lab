param(
    [string]$Image = 'nvcr.io/nvidia/tensorrt@sha256:814325e2b8a653f354c30bbcf5ecc8d4c780cf878a88a320ae648fbfdd9dd82d',
    [string]$Engine = 'artifacts/squeezenet1.1-fp32.engine',
    [string]$Log = 'reports/phase-1/tensorrt-trtexec-benchmark.log'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$logPath = Join-Path $root $Log
$logDir = Split-Path -Parent $logPath
New-Item -ItemType Directory -Force $logDir | Out-Null
docker run --rm --gpus all -v "$root`:/workspace" -w /workspace $Image trtexec `
    --loadEngine=$Engine --warmUp=200 --iterations=100 --duration=0 --noDataTransfers 2>&1 |
    Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) { throw "TensorRT benchmark failed with exit code $LASTEXITCODE" }
