$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseRoot = Join-Path $root "release"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$releaseDir = Join-Path $releaseRoot "k_quant_deck_$stamp"
$zipPath = "$releaseDir.zip"

$excludeDirs = @(".git", ".cache", "__pycache__", "release", "dist", "build", ".venv", "venv")
$excludeFiles = @(".env")

if (!(Test-Path $releaseRoot)) {
    New-Item -ItemType Directory -Path $releaseRoot | Out-Null
}

New-Item -ItemType Directory -Path $releaseDir | Out-Null

Get-ChildItem -Path $root -Force | ForEach-Object {
    $name = $_.Name
    if ($excludeDirs -contains $name) { return }
    if ($excludeFiles -contains $name) { return }
    if ($name -like "*.env" -and $name -ne ".env.example") { return }
    if ($name -like "*.txt" -and $name -like "*흠흠*") { return }
    if ($name -like "*token*" -or $name -like "*secret*") { return }
    if ($name -like "*.pyc") { return }

    $target = Join-Path $releaseDir $name
    Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
}

Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath -Force

Write-Host "Release folder: $releaseDir"
Write-Host "Release zip:    $zipPath"
Write-Host "Check that local env files and .cache are not inside the zip before sharing."
