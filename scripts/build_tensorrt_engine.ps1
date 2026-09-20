param(
    [string]$Image = 'nvcr.io/nvidia/tensorrt@sha256:814325e2b8a653f354c30bbcf5ecc8d4c780cf878a88a320ae648fbfdd9dd82d',
    [string]$Onnx = 'artifacts/squeezenet1.1-torchvision.onnx',
    [string]$Engine = 'artifacts/squeezenet1.1-fp32.engine'
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mount = "$root`:/workspace"
docker run --rm --gpus all -v $mount -w /workspace $Image trtexec `
    --onnx=$Onnx `
    --saveEngine=$Engine `
    --noTF32 `
    --memPoolSize=workspace:512 `
    --dumpProfile

if ($LASTEXITCODE -ne 0) {
    throw "TensorRT engine build failed with exit code $LASTEXITCODE"
}

Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root $Engine)
