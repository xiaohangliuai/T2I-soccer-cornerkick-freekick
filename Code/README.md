# Soccer Tactics Text-to-Image Generation Project

## Project Overview
This project focuses on generating realistic soccer tactical images based on text descriptions. The system takes textual descriptions of soccer tactics (with a focus on corner kick scenarios) and generates corresponding realistic images showing player positions, formations, and tactical setups.

## Implementation Details

### Programming Language
- **Python 3.8+**

### Computer Platform
- **Windows**

### Libraries and Frameworks
- **PyTorch**: Main framework for deep learning models
- **Segment Anything Model (SAM)**: For image segmentation and player detection
- **Stable Diffusion XL (SDXL)**: For text-to-image generation
- **OpenCV (cv2)**: For image processing and manipulation
- **NumPy**: For numerical operations and array manipulation
- **Matplotlib**: For visualization

## Files Description

### 1. `generate_prompts.py`
This script generates text prompts for corner kick scenarios. It creates detailed descriptions of player positions, formations, and tactical setups for both attacking (white) and defending (blue) teams. The prompts are saved to a caption file that can be used for text-to-image generation.

**Key Features:**
- Defines various attacker positions and defender formations
- Generates systematic descriptions of corner kick scenarios
- Creates captions for each image with detailed tactical information

### 2. `generate_images.py`
This script handles the image generation process by manipulating player positions on a soccer field. It uses the Segment Anything Model (SAM) to identify and segment players, then repositions them according to tactical requirements.

**Key Features:**
- Player segmentation using SAM
- Tactical positioning of players according to rules:
  - Players should be in specific field areas
  - Concentration of players near the penalty spot
  - Proper defensive positioning
  - Avoiding player overlap
- Image saving with proper formatting

### 3. `SDXL_Soccer_generator.ipynb`
This Jupyter notebook implements the Stable Diffusion XL model for text-to-image generation of soccer tactical scenes. It processes the text prompts and generates corresponding images.

**Key Features:**
- Implementation of SDXL model for image generation
- Processing of tactical text descriptions
- Generation of realistic soccer scenes

### 4. `genImages.ipynb`
This notebook provides additional image generation capabilities, potentially with different approaches or settings compared to the main SDXL implementation.

## Installation Requirements

### Required Packages
```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
segment-anything # Meta's SAM model
diffusers>=0.20.0 # For Stable Diffusion XL
transformers>=4.30.0
```

## Setup Instructions

1. Clone the repository:
```
git clone [repository-url]
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the SAM model checkpoint:
```
# Download the SAM ViT-H checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

4. Prepare your dataset structure:
```
Dataset/
├── Corner-kick/
│   ├── ck.jpg
│   └── captions.txt
└── GenerateImages/
    └── Corner-Kick/
        └── V2/
```

## Usage

1. Generate text prompts:
```
python generate_prompts.py
```

2. Generate segmented and positioned player images:
```
python generate_images.py
```

3. Run the SDXL model for text-to-image generation:
   - Open `SDXL_Soccer_generator.ipynb` in Jupyter Notebook or Google Colab
   - Follow the instructions in the notebook to generate images

## Notes
- The generated images focus on corner kick scenarios but can be extended to other tactical situations
- The system combines image manipulation and AI-based generation for realistic results
- Player positions follow specific tactical rules to ensure realism 