# UNSB MRI Enhancement — ChatClinic Plugin

**Team:** BISPL (Bio-Imaging Signal Processing Lab, KAIST)

Ultra-Low-Field (64mT) → 3T Brain MRI Enhancement using Unpaired Neural Schrödinger Bridge (UNSB)

## Plugin Structure

```
plugins/unsb_mri_enhance/
├── tool.json                    # ChatClinic tool manifest
├── run.py                       # Entrypoint (--input / --output)
├── environments.yml
├── README.md
├── checkpoints/
│   └── iter_65000_net_G.pth     # Generator checkpoint (~56MB)
├── models/                      # Network definitions
├── util/                        # Utilities
└── options/                     # Option parsing
```

## Execution

ChatClinic runner calls:
```bash
python3 run.py --input input.json --output output.json
```

### Manual Test

```bash
echo '{
  "question": "Enhance this MRI",
  "analysis_source": {
    "file_name": "/path/to/test_brain.png",
    "modality": "medical-image"
  }
}' > /tmp/test_input.json

python3 run.py --input /tmp/test_input.json --output /tmp/test_output.json
cat /tmp/test_output.json
```

## Dependencies

- Python >= 3.9
- PyTorch (CUDA)
- torchvision
- numpy
- Pillow

## Model Info

| Item | Value |
|------|-------|
| Architecture | ResNet-9blocks + conditional time embedding |
| Input/Output | 3ch RGB, 256×256 |
| Inference | 5-step SDE loop |
| Checkpoint | iter_65000_net_G.pth (~56MB) |
| Training | 64mT→3T unpaired brain MRI  |

## Output Artifacts

| Artifact | Description |
|----------|-------------|
| `enhanced_image` | Enhanced 3T-like MRI image |
| `comparison` | Side-by-side: 64mT input (left) vs enhanced (right) |


## Environment Setup
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate
conda activate unsb4090

# Or update existing environment
conda env update -f environment.yml
```

> **Note:** `tool_runner.py` uses system `python3` by default.
> If a separate conda environment is required, activate it before running ChatClinic,
> or modify the entrypoint to use the conda python path directly.