@echo off
echo [1] Starting build... > E:\AgentNate\build_vllm.log 2>&1
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >> E:\AgentNate\build_vllm.log 2>&1
echo [2] vcvars done, errorlevel=%ERRORLEVEL% >> E:\AgentNate\build_vllm.log 2>&1

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=8.6
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=8

echo [3] Checking tools >> E:\AgentNate\build_vllm.log 2>&1
where cl.exe >> E:\AgentNate\build_vllm.log 2>&1
where nvcc >> E:\AgentNate\build_vllm.log 2>&1
nvcc --version >> E:\AgentNate\build_vllm.log 2>&1

echo [4] Starting pip install >> E:\AgentNate\build_vllm.log 2>&1
cd /d E:\AgentNate\vllm-source
"E:\AgentNate\envs\vllm\Scripts\python.exe" -m pip install -e . --no-build-isolation -v >> E:\AgentNate\build_vllm.log 2>&1
echo [5] pip exit code: %ERRORLEVEL% >> E:\AgentNate\build_vllm.log 2>&1
