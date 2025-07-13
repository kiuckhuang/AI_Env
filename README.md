# AI Development Environment Setup Guide (Windows 11 + RTX 5090)

This document provides a step-by-step guide to configure an optimal AI development environment on Windows 11 with NVIDIA RTX 5090 GPU.

---

## 🔧 System Requirements
- Windows 11 (22H2 or later)
- NVIDIA RTX 5090 GPU
- Minimum 32GB RAM (64GB+ recommended)
- 1TB+ free disk space

---

## 📦 Software Installation Steps

### 1. NVIDIA Drivers & CUDA Toolkit
**Install latest drivers from:**
[https://www.nvidia.com/en-us/drivers/](https://www.nvidia.com/en-us/drivers/)  
*(Ensure you select the version compatible with your RTX 5090)*

**CUDA Toolkit (optional for self-compiling):**
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)  
*Recommended: CUDA 12.8*

---

### 2. Chocolatey Package Manager
**Install Chocolatey using:**
[https://github.com/chocolatey/choco/releases - Use latest version](https://github.com/chocolatey/choco/releases)  
*Current version *[v2.5.0](https://github.com/chocolatey/choco/releases/download/2.5.0/chocolatey-2.5.0.0.msi)*

---

### 3. Python/Nodejs Environment (optional)
**Install Python 3.12:**
```cmd
choco install python312
choco install nodejs
choco install ffmpeg
```

---

### 4. PyTorch Installation (optional)
**Install PyTorch 2.7.1 with CUDA support:**
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verification:**
Run `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`  
Expected output:
```
2.7.1+cu128
True
```

---

### 5. Git Installation
[https://git-scm.com/downloads/win](https://git-scm.com/downloads/win)  
*Ensure "Use Git from Windows Command Prompt" is selected during installation*

---

## 🧠 AI Development Tools

### LM Studio (LLM Chat/RAG/MCP)
**Download:**
[https://lmstudio.ai/download](https://lmstudio.ai/download)  
*Suggested models for testing:*
- `qwen/qwen3-14b`
- `Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf`

**Installation Steps:**
1. Download and extract the installation package
2. Launch LM Studio application
3. Load suggested models from Model Hub

---

### ComfyUI (Text-to-Image/Editing)
**Download latest release:**
[https://github.com/comfyanonymous/ComfyUI/releases - Use latest version](https://github.com/comfyanonymous/ComfyUI/releases)  
*Current version:* [v0.3.44](https://github.com/comfyanonymous/ComfyUI/releases/download/v0.3.44/ComfyUI_windows_portable_nvidia.7z)

**Installation Steps:**
1. Extract the 7z file to your working directory
2. Edit `update/update_comfyui_and_python_dependencies.bat`
   *Replace URL line from:*  
   ```cmd
   ..\python_embeded\python.exe -s -m pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128 -r ../ComfyUI/requirements.txt pygit2
   ```

   *Replace above URL line with:*  
   ```cmd
   ..\python_embeded\python.exe -s -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -r ../ComfyUI/requirements.txt pygit2
   ```
   
3. Run:  
   ```cmd
   update/update_comfyui_and_python_dependencies.bat
   run_nvidia_gpu_fast.bat
   ```

**Install ComfyUI-Manager:**
[https://github.com/Comfy-Org/ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager)  
*Enables one-click model installation*

---

## 🛠️ Post-Installation Verification
1. Open **NVIDIA Control Panel** to verify GPU detection
2. Run `nvidia-smi` in CMD to check driver version
3. Test PyTorch CUDA support with:
```python
import torch
print(torch.cuda.get_device_name(0))
```

---

## 📌 Notes
- Always verify driver/CUDA compatibility with your specific GPU model
- For development, consider using virtual environments with `python -m venv`
- Backup important configurations regularly

