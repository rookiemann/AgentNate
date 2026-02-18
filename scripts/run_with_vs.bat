@echo off
REM Activate VS Build Tools environment and run Python command
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set PATH=E:\AgentNate\envs\exllamav2\Scripts;%PATH%
%*
