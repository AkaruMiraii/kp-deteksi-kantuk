@echo off
setlocal
set "PKG=%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
if "%~1"=="" (
  start "" "%PKG%runtime\python\pythonw.exe" -E -s "%PKG%app\kp_poc_entry.py"
) else (
  "%PKG%runtime\python\python.exe" -E -s "%PKG%app\kp_poc_entry.py" %*
)
endlocal
