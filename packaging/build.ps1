# Build CandleCast: PyInstaller one-folder app (dist\CandleCast) + Inno Setup installer
# (dist\CandleCast-Setup-<version>.exe) + a portable zip (dist\CandleCast-<version>-portable.zip).
#
# Run from the repository root:  powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build.ps1
# Needs: .venv with the project installed (pip install -e ".[dev]" pyinstaller) and Inno Setup 6.
param([switch]$SkipInstaller, [switch]$SkipZip)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$version = (& $py -c "import candlecast; print(candlecast.__version__)").Trim()
Write-Host "Building CandleCast $version" -ForegroundColor Cyan

Push-Location $root
try {
    Remove-Item -Recurse -Force build, dist\CandleCast -ErrorAction SilentlyContinue
    & $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\candlecast.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

    if (-not $SkipZip) {
        $zip = "dist\CandleCast-$version-portable.zip"
        Remove-Item $zip -ErrorAction SilentlyContinue
        Compress-Archive -Path dist\CandleCast -DestinationPath $zip -CompressionLevel Optimal
        Write-Host "Portable zip: $zip"
    }
    if (-not $SkipInstaller) {
        $iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                  "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found - install it or pass -SkipInstaller" }
        & $iscc "/DAppVersion=$version" packaging\candlecast.iss
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
        Write-Host "Installer: dist\CandleCast-Setup-$version.exe"
    }
} finally {
    Pop-Location
}
