<div><h2>[ICML'25] Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models</h2></div>
<br>

**Tung M. Luu, Younghwan Lee, Donghoon Lee, Sunho Kim,
Min Jun Kim, Chang D. Yoo**
<br>
KAIST, South Korea
<br>
[[arXiv]](https://www.arxiv.org/abs/2506.12822) [[Website]](https://erlvlm2025.github.io/) 


## Overview
This is the official implementation of **ERL-VLM** for MetaWorld tasks.

**🔥 New: LLaVA Integration** - This implementation now uses **LLaVA** (Large Language and Vision Assistant) instead of Gemini for vision-language feedback, providing local and more flexible VLM capabilities through Ollama.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for LLaVA 7B model)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models and data

### Software Requirements  
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **CUDA**: 11.7+ (for GPU acceleration)
- **Python**: 3.9
- **Ollama**: Latest version (for LLaVA inference)

## Installation

### Step 1: Basic Installation
```bash
conda create --name erlvlm python=3.9
conda activate erlvlm
```

### Step 2: Install PyTorch with CUDA Support
```bash
# Install PyTorch with CUDA 11.7 (adjust for your CUDA version)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

### Step 3: Fix Requirements and Install Dependencies
```bash
# Fix pyparsing line in requirements.txt if needed
sed -i 's/pyparsing @ file:\/\/\/.*$/pyparsing==3.0.9/' requirements.txt

# Remove torch/torchvision from requirements to avoid conflicts
sed -i '/^torch==/d' requirements.txt
sed -i '/^torchvision==/d' requirements.txt

# Install remaining dependencies
pip install -r requirements.txt --no-deps
```

### Step 4: Install Project and Custom Packages
```bash
# Install main project
pip install -e .

# Install custom dmc2gym package
cd custom_dmc2gym
pip install -e .
cd ..
```

### Step 5: Install MuJoCo 210 (Required for mujoco-py)
```bash
# Create MuJoCo directory
mkdir -p ~/.mujoco

# Download and extract MuJoCo 210
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
cd -
```

### Step 6: Setup Environment Variables (Automatic)
```bash
# Create conda environment activation script for automatic setup
mkdir -p ~/miniconda3/envs/erlvlm/etc/conda/activate.d

# Create environment setup script
cat > ~/miniconda3/envs/erlvlm/etc/conda/activate.d/mujoco_env.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
EOF

# Make executable
chmod +x ~/miniconda3/envs/erlvlm/etc/conda/activate.d/mujoco_env.sh

# Reactivate environment to load variables
conda deactivate
conda activate erlvlm
```

## Setup Vision-Language Model (VLM)

This project now uses **LLaVA** instead of Gemini for vision-language feedback. The system uses Ollama to run LLaVA models locally.

### Step 1: Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service (if not auto-started)
ollama serve
```

### Step 2: Download LLaVA Model
```bash
# Download and install LLaVA model (this may take several minutes)
ollama pull llava

# Verify installation
ollama list
```

### Step 3: Test LLaVA Interface
```bash
# Test the LLaVA interface
python llava/llava_interface.py
```

**Important Notes:**
- Ollama must be running (`ollama serve`) before starting training
- The first time you run training, it may download the LLaVA model automatically
- LLaVA requires significant GPU memory (~8GB+ recommended for 7B model)
- If you encounter issues, check that Ollama is accessible at `http://localhost:11434`

## Run experiments

### Prerequisites for Running Experiments
**Before running any experiments, ensure:**
1. Ollama service is running: `ollama serve` 
2. LLaVA model is installed: `ollama list` should show `llava`
3. Environment variables are set (automatic if you followed installation steps)

### For Headless Servers (No Display)
```bash
# Start Ollama service (in separate terminal or background)
ollama serve &

# Test run with debug mode (disables wandb)
MUJOCO_GL=egl python train_PEBBLE_VLM.py env=metaworld_drawer-open-v2 debug=True

# Full training runs
MUJOCO_GL=egl bash scripts/open_drawer/run_ERLVLM.sh
MUJOCO_GL=egl bash scripts/soccer/run_ERLVLM.sh  
MUJOCO_GL=egl bash scripts/sweep_into/run_ERLVLM.sh
```

### For Servers with Display
```bash
# Start Ollama service (in separate terminal or background)
ollama serve &

# Test run with debug mode
python train_PEBBLE_VLM.py env=metaworld_drawer-open-v2 debug=True

# Full training runs  
bash scripts/open_drawer/run_ERLVLM.sh
bash scripts/soccer/run_ERLVLM.sh
bash scripts/sweep_into/run_ERLVLM.sh
```

### Manual Environment Variables (if automatic setup doesn't work)
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
```

## Troubleshooting

### Common Issues and Solutions

**1. "LLaVA model not initialized" Error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Ensure LLaVA model is installed
ollama pull llava
```

**2. "Ollama not available" or Connection Errors**
```bash
# Check Ollama status
ollama list

# Restart Ollama service
pkill ollama
ollama serve

# Test Ollama API
curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{"model": "llava", "prompt": "test", "stream": false}'
```

**3. GPU Memory Issues**
- LLaVA requires significant GPU memory (~8GB+ for 7B model)
- Consider using a smaller model: `ollama pull llava:7b` 
- Monitor GPU usage: `nvidia-smi`
- Reduce batch sizes in training configuration

**4. NumPy/Package Version Conflicts**
```bash
# Reinstall with correct versions
pip install numpy==1.24.3
pip install transformers==4.37.0
```

**5. MuJoCo/Rendering Issues**
```bash
# For headless servers, always use:
MUJOCO_GL=egl python train_PEBBLE_VLM.py ...

# Check MuJoCo installation
ls ~/.mujoco/mujoco210/
```

**6. Training Starts but No VLM Queries**
- Verify training logs show: "LLaVA model initialized for VLM queries"
- Check that `vlm_feedback=True` in your configuration
- Look for "[LLaVA]" prefixed messages in output

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@InProceedings{
    luu2025erlvlm,
    title={Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models},
    author={Tung Minh Luu , Younghwan Lee, Donghoon Lee, Sunho Kim, Min Jun Kim, Chang D. Yoo},
    booktitle={Proceedings of the 42th International Conference on Machine Learning},
    year={2025}
}
```

## Acknowledgements
- This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-II220184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).

- This repo contains code adapted from [RbRL](https://github.com/Dev1nW/Rating-based-Reinforcement-Learning), 
[RL-VLM-F](https://github.com/yufeiwang63/RL-VLM-F). We thank the authors and contributors for open-sourcing their code.

## License

MIT
