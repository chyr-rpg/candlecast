; Inno Setup script for the CandleCast Windows installer. Build with packaging\build.ps1.
#define AppName "CandleCast"
#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#define AppExe "CandleCast.exe"

[Setup]
AppId={{6B3D2E7A-5C1F-4E8B-9A2D-C4A1F0E7B932}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=CandleCast contributors
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=CandleCast-Setup-{#AppVersion}
SetupIconFile=..\src\candlecast\assets\candlecast.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; per-user install by default (no admin prompt); "install for all users" stays available
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\CandleCast\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; the program folder only; forecasts, price data, keys and model weights in %LOCALAPPDATA%\CandleCast are kept
Type: filesandordirs; Name: "{app}"
