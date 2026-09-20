param(
    [string]$Image = 'nvcr.io/nvidia/tensorrt:25.02-py3',
    [string]$Onnx = 'artifacts/squeezenet1.1-torchvision.onnx',
    [string]$Engine = 'artifacts/squeezenet1.1-torchvision.plan'
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mount = "$root`:/workspace"
docker run --rm --gpus all -v $mount -w /workspace $Image trtexec `
    --onnx=$Onnx `
    --saveEngine=$Engine `
    --shapes=data:1x3x224x224 `
    --dumpProfile

if ($LASTEXITCODE -ne 0) {
    throw "TensorRT engine build failed with exit code $LASTEXITCODE"
}

Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root $Engine)
