# Soccer Tactics Text-to-Image Generation

## Project Overview

Soccer Tactics Text-to-Image Generation is a computer vision and generative AI project that converts natural-language descriptions of soccer tactics—primarily corner-kick scenarios—into realistic match images. The project uses Meta’s Segment Anything Model (SAM) to extract players from an existing soccer image and reposition attackers and defenders according to tactical constraints, such as penalty-area concentration, defensive marking, and collision avoidance. These synthetic layouts are paired with automatically generated tactical captions to create a training dataset. Stable Diffusion and SDXL are then fine-tuned with LoRA to generate soccer scenes that more accurately reflect requested player formations, marking strategies, and goalkeeper positioning. The repository also includes inference scripts, model checkpoints, training statistics, and comparisons between the base and fine-tuned models.
