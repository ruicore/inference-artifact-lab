param([string]$Python = '3.11')

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$scratchParent = Join-Path $root '.manifest'
New-Item -ItemType Directory -Force -Path $scratchParent | Out-Null
if ((Get-Item -LiteralPath $scratchParent).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
    throw 'reproduction scratch parent must not be a reparse point'
}
$scratch = Join-Path $scratchParent ('cpu-repro-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
$previousPythonPath = $env:PYTHONPATH
try {
    Push-Location $root
    try {
        $files = @(git ls-files --cached)
        if ($LASTEXITCODE -ne 0 -or $files.Count -eq 0) { throw 'unable to enumerate public project files' }
        foreach ($relative in $files) {
            $source = Join-Path $root $relative
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
            $target = Join-Path $scratch $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $source -Destination $target
        }
    } finally {
        Pop-Location
    }
    Push-Location $scratch
    try {
        uv venv --python $Python .venv
        if ($LASTEXITCODE -ne 0) { throw 'fresh Python environment creation failed' }
        $pythonExe = Join-Path $scratch '.venv/Scripts/python.exe'
        $env:PYTHONPATH = Join-Path $scratch 'src'
        uv pip install --python $pythonExe -r scripts/requirements-reference-public.txt
        if ($LASTEXITCODE -ne 0) { throw 'public dependency installation failed' }
        & $pythonExe scripts/export_torchvision_onnx.py
        if ($LASTEXITCODE -ne 0) { throw 'public model export failed' }
        & $pythonExe scripts/run_torchvision_gate.py | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'public reference gate failed' }
        $referenceReport = Get-Content -Raw reports/phase-1/squeezenet11-torchvision-onnx-cpu.json | ConvertFrom-Json
        if ($referenceReport.status -ne 'pass' -or
            (@($referenceReport.checks | Where-Object { $_.id -eq 'fixture.binding' -and $_.status -eq 'pass' })).Count -ne 1 -or
            (@($referenceReport.checks | Where-Object { $_.id -eq 'runtime.equivalence' -and $_.status -eq 'pass' })).Count -ne 1) {
            throw 'reference report did not meet the declared CPU decision'
        }
        & $pythonExe -m inference_artifact_lab examples/squeezenet11-torchvision.manifest.json `
            --runtime onnx-cpu --inputs-npy artifacts/squeezenet11-fixture.npy `
            --reference-npy artifacts/squeezenet11-reference.npy `
            --report artifacts/cli-onnx-cpu.json | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'package ONNX CPU gate failed' }
        $cliReport = Get-Content -Raw artifacts/cli-onnx-cpu.json | ConvertFrom-Json
        if ($cliReport.status -ne 'pass' -or $cliReport.manifest_digest -ne $referenceReport.manifest_digest) {
            throw 'package and reference gate decisions disagree'
        }
        foreach ($report in @($referenceReport, $cliReport)) {
            if ($report.scope.runtime.profile -ne 'onnx-cpu-windows-py311' -or
                $report.scope.runtime.acceptance_role -ne 'cpu_baseline') {
                throw 'CPU report is missing its explicit acceptance scope'
            }
        }
        & $pythonExe scripts/render_report.py reports/phase-1/squeezenet11-torchvision-onnx-cpu.json | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path reports/phase-1/squeezenet11-torchvision-onnx-cpu.md)) {
            throw 'human report rendering failed'
        }
        Write-Output "clean CPU reproduction PASS manifest_digest=$($cliReport.manifest_digest)"
    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $parentPath = [System.IO.Path]::GetFullPath($scratchParent)
    $scratchPath = [System.IO.Path]::GetFullPath($scratch)
    if ($scratchPath.StartsWith($parentPath + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
        -not ((Get-Item -LiteralPath $scratchParent).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -and
        (Test-Path -LiteralPath $scratchPath -PathType Container) -and
        -not ((Get-Item -LiteralPath $scratchPath).Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        Remove-Item -LiteralPath $scratchPath -Recurse -Force
    }
}
