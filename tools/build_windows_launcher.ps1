param([switch]$Shortcut)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Le compilateur .NET Framework Windows est introuvable.' }
$source = Join-Path $PSScriptRoot 'SmartCameraLauncher.cs'
$executable = Join-Path $projectRoot 'SmartCamera.exe'
$icon = Join-Path $projectRoot 'assets/smart-camera.ico'
& $compiler /nologo /target:winexe /optimize+ /reference:System.Windows.Forms.dll "/win32icon:$icon" "/out:$executable" $source
if ($LASTEXITCODE -ne 0) { throw 'La compilation du lanceur a échoué.' }
if ($Shortcut) {
    $shell = New-Object -ComObject WScript.Shell
    $linkPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Smart Camera.lnk'
    if (Test-Path -LiteralPath $linkPath) { throw "Le raccourci existe déjà : $linkPath" }
    $link = $shell.CreateShortcut($linkPath)
    $link.TargetPath = $executable
    $link.WorkingDirectory = $projectRoot
    $link.Description = 'Caméra Pi et analyse IA — application Windows'
    $link.Save()
}
Write-Output $executable
