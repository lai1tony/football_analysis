#define MyAppName "Football Analysis"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Football Analysis"
#define MyAppExeName "start_app.bat"

[Setup]
AppId={{B9AB7F9B-4E23-4C1A-8B86-FA1234567890}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\FootballAnalysis
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=FootballAnalysisSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: ".git\*,.venv\*,node_modules\*,dist\*,*.pyc,__pycache__\*,.mypy_cache\*,.pytest_cache\*,data\.myenv\*,data\.cache\*,data\.playwright-cli\*,.playwright-cli\*"

[Icons]
Name: "{group}\Football Analysis"; Filename: "{app}\start_app.bat"; WorkingDir: "{app}"
Name: "{commondesktop}\Football Analysis"; Filename: "{app}\start_app.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
Name: "runinstall"; Description: "Run dependency installer after copying files"; GroupDescription: "Post-install:"

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File \"{app}\installer\install.ps1\""; WorkingDir: "{app}"; Flags: postinstall; Tasks: runinstall
Filename: "{app}\start_app.bat"; Description: "Start Football Analysis"; WorkingDir: "{app}"; Flags: postinstall skipifsilent nowait
