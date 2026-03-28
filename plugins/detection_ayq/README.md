# AYQ Detection - ChatClinic Plugin

**Team:** BISPL (BioImaging, Signal Processing, & machine Learning) Lab, KAIST AI

Medical object detection tool based on the following tools and papers:
- MMDetection: Open MMLab Detection Toolbox and Benchmark  
[![arXiv](https://img.shields.io/badge/arXiv-1906.07155-b31b1b.svg)](https://arxiv.org/abs/1906.07155)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://mmdetection.readthedocs.io/en/latest/)
- DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection  
[![arXiv](https://img.shields.io/badge/arXiv-2203.03605-b31b1b.svg)](https://arxiv.org/abs/2203.03605)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://github.com/IDEA-Research/DINO)
- Align Your Query: Representation Alignment for Multimodality Medical Object Detection  
[![arXiv](https://img.shields.io/badge/arXiv-2510.02789-b31b1b.svg)](https://arxiv.org/abs/2510.02789)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://araseo.github.io/alignyourquery/)


## Plugin Structure

```
plugins/detection_ayq/
├── tool.json                    # ChatClinic tool manifest
├── run.py                       # Entrypoint (--input / --output)
├── requirements.txt
├── README.md
├── assets/
├── ayq_runtime/
│   ├── mmdet/                   # Modified MMDetection Toolbox
│   └── text_embeddings/         # Precomputed text embeddings
├── checkpoints/
│   └── dino_ayq.pth             # Detector checkpoint (DINO+AYQ)
├── configs/
│   └── dino_ayq_config.py       # Detector config file
├── demo/
│   └── image_demo.py            # Called by run.py for detection
├── runtime_outputs/             # Detection outputs
├── samples/                     # Sample images
└── scripts/
```

## Example Usage

![main figure](assets/ChatClinic_Detection_AYQ.png)

## Environment Setup: Conda Environment
Set up the conda environment for the AYQ Detection plugin.
```bash
# Create conda environment
conda create -n ayq python=3.10 -y
conda activate ayq
pip install -U pip "setuptools<81" wheel
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"

pip install -r /path/to/chatclinic/plugins/detection_ayq/requirements.txt
pip install 'numpy<2'
pip install "opencv-python==4.11.0.86"
```

## Environment Setup: Environment Variables
Set environment variables like below.
```bash
# Inside backend terminal, force AYQ Detection plugin to use the ayq conda environment.
...
export AYQ_PYTHON_EXECUTABLE="/path/to/anaconda3/envs/ayq/bin/python"
export AYQ_DEVICE=cpu
source plugins/detection_ayq/scripts/export_local_runtime_env.sh
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

> **Note:**
> We have modified the entrypoint to use the separate conda environment directly, which is why we are setting these environment variables.