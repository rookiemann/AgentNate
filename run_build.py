"""Run vLLM build via CMD with MSVC environment."""
import subprocess
import sys
import tempfile
import os

LOG = r'E:\AgentNate\build_vllm.log'

# Write batch commands to a temp file
bat_content = r'''@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=8.6
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=8
echo === Environment ===
where cl.exe
where nvcc
nvcc --version
cd /d E:\AgentNate\vllm-source
echo === Starting build (incremental) ===
"E:\AgentNate\envs\vllm\Scripts\python.exe" -m pip install -e . --no-build-isolation -v
echo === Exit code: %ERRORLEVEL% ===
exit /b %ERRORLEVEL%
'''

bat_file = os.path.join(tempfile.gettempdir(), 'vllm_build.bat')
with open(bat_file, 'w') as f:
    f.write(bat_content)

print(f"Batch file: {bat_file}")
print(f"Log file: {LOG}")
print("Starting build...")

with open(LOG, 'w') as logf:
    logf.write("=== Build started ===\n")
    logf.flush()
    proc = subprocess.Popen(
        ['cmd.exe', '/c', bat_file],
        stdout=logf,
        stderr=subprocess.STDOUT,
        cwd=r'E:\AgentNate\vllm-source',
    )
    proc.wait()
    logf.write(f"\n=== Process exit code: {proc.returncode} ===\n")

print(f"Build complete. Exit code: {proc.returncode}")
sys.exit(proc.returncode)
