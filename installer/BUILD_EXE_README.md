# 编译 EXE 安装器

当前发布包已包含 Inno Setup 工程：`installer\build_installer_inno.iss`。

在 Windows 上安装 Inno Setup 6 后，右键该 `.iss` 文件选择 Compile，或命令行执行：

```powershell
ISCC.exe installer\build_installer_inno.iss
```

输出文件为：

```text
dist\FootballAnalysisSetup.exe
```

如果没有 Inno Setup，也可以直接分发 ZIP 包，解压后运行：

```powershell
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```
