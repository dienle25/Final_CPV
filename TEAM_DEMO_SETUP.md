# Final CPV Classroom Demo

## Windows requirements

- Windows 10 or Windows 11
- Python 3.12 x64
- Git
- Internet connection for the first dependency installation

## Download

```powershell
git clone https://github.com/dienle25/Final_CPV.git
cd Final_CPV
```

Members may also use **Code → Download ZIP**, extract the archive, and open
PowerShell in the extracted folder.

## First-time setup

Verify Python:

```powershell
py -3.12 --version
```

Install the project environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Run the readiness check:

```powershell
.\CHECK_DEMO.bat
```

## Start the classroom application

```powershell
.\RUN_CLASSROOM_DEMO.bat
```

Open:

```text
http://127.0.0.1:8501
```

## Registered student references

- CE190256: 15 images
- CE190579: 21 images
- CE190625: 15 images
- CE191641: 15 images

CE182206 currently has one legacy reference image, so preflight may display a
warning for that ID. A warning is not a startup failure.

## Privacy

This private repository contains student IDs and facial reference images.
Do not make the repository public or redistribute the data outside the team.
