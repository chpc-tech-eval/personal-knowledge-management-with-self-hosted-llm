### How to setup Conda on youur cluster

We use Conda because it makes it easier to install CUDA-compatible Python packages (for example, PyTorch and bitsandbytes) that must match the system toolchain.

1. Install conda via the package manager:

```bash
sudo dnf install conda
```

2. Create your virtual environment (example name: `llm`):

```bash
conda create -n llm python=3.12.12 -y
conda activate llm
```

3. Initialize Conda for your shell if needed (replace `bash` with your shell):

```bash
conda init bash
```

4. update the current env with packages from environment.yml
conda env update -f environment.yml --prune

5. Authenticate with Hugging Face (if required by the model):

```bash
hf auth login
```

6. If you add packages and want to share the environment, export it:

```bash
conda env export > environment.yml
```

7. Deactivate the environment when finished:

```bash
conda deactivate
```
