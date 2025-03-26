import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn.functional as F

class SoccerCornerKickDataset(Dataset):
    def __init__(self, caption_file, image_dir, tokenizer_1, tokenizer_2, resolution=768):
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution
        self.image_paths = []
        self.prompts = []
        
        with open(caption_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        image_name, caption = parts
                        image_path = os.path.join(image_dir, image_name)
                        if os.path.exists(image_path):
                            self.image_paths.append(image_path)
                            self.prompts.append(caption)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # [C, H, W]
        
        # Tokenize prompts
        text_input_1 = self.tokenizer_1(
            prompt, padding="max_length", max_length=self.tokenizer_1.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        text_input_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids_1": text_input_1.input_ids[0],
            "input_ids_2": text_input_2.input_ids[0],
        }

def train_lora():
    # Set paths
    DATA_DIR = "Dataset/Corner-kick"
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
    OUTPUT_DIR = "soccer-corner-lora"
    MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training parameters
    RESOLUTION = 768
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    MAX_TRAIN_STEPS = 1000
    SAVE_STEPS = 200
    MIXED_PRECISION = "fp16"
    
    # LoRA parameters
    RANK = 8
    LORA_ALPHA = 16
    
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )
    
    # Load tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer_2")
    
    # Create dataset and dataloader
    train_dataset = SoccerCornerKickDataset(
        CAPTION_FILE, IMAGE_DIR, tokenizer_1, tokenizer_2, RESOLUTION
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    print(f"Created dataset with {len(train_dataset)} images")
    
    # Load models with memory optimizations
    text_encoder_1 = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder_2", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME, subfolder="vae", torch_dtype=torch.float16
    )
    
    # Set models to eval mode and freeze them
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    
    for param in text_encoder_1.parameters():
        param.requires_grad = False
    for param in text_encoder_2.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    
    # Load UNet and apply LoRA
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME, subfolder="unet", torch_dtype=torch.float16
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj", 
            "to_k", "to_q", "to_v", "to_out.0"
        ],
        bias="none",
        lora_dropout=0.0,
        init_lora_weights=True
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    # Move frozen models to device
    vae.to(accelerator.device)
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # Set up noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(total=MAX_TRAIN_STEPS)
    
    for epoch in range(100):
        for batch in train_dataloader:
            if global_step >= MAX_TRAIN_STEPS:
                break
                
            # Extract inputs
            pixel_values = batch["pixel_values"].to(accelerator.device)
            input_ids_1 = batch["input_ids_1"].to(accelerator.device)
            input_ids_2 = batch["input_ids_2"].to(accelerator.device)
            
            # Convert images to latent space
            with torch.no_grad(), torch.amp.autocast('cuda'):
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=latents.device
            )
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            with torch.no_grad(), torch.amp.autocast('cuda'):
                encoder_hidden_states_1 = text_encoder_1(input_ids_1)[0]
                encoder_hidden_states_2 = text_encoder_2(input_ids_2)[0]
                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
            
            # Forward pass and calculate loss
            with accelerator.accumulate(unet):
                # Set up SDXL-specific conditioning
                # Original size, target size for the time_ids
                original_size = (1024, 1024) 
                target_size = (RESOLUTION, RESOLUTION)
                
                time_ids = torch.zeros(
                    (latents.shape[0], 6),
                    dtype=latents.dtype,
                    device=latents.device
                )

                # Original size is 1024x1024
                time_ids[:, 0] = 1024  # orig_height
                time_ids[:, 1] = 1024  # orig_width

                # No cropping
                time_ids[:, 2] = 0     # crop_top
                time_ids[:, 3] = 0     # crop_left

                # Target size is your training resolution
                time_ids[:, 4] = RESOLUTION  # target_height
                time_ids[:, 5] = RESOLUTION  # target_width

                # Create empty text_embeds with exactly the expected shape
                text_embeds = torch.zeros(
                    (latents.shape[0], 1280),
                    dtype=latents.dtype,
                    device=latents.device
                )

                # Package them in the format SDXL expects
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids
                }
                
                # Run model forward
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            global_step += 1
            
            # Save checkpoint
            if global_step % SAVE_STEPS == 0 and accelerator.is_main_process:
                # Unwrap model
                unwrapped_unet = accelerator.unwrap_model(unet)
                
                # Save LoRA weights
                lora_state_dict = {
                    k: v for k, v in unwrapped_unet.state_dict().items()
                    if "lora_" in k
                }
                
                checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    lora_state_dict,
                    os.path.join(checkpoint_dir, "pytorch_lora_weights.bin")
                )
                
                print(f"Saved checkpoint at step {global_step}")
                
            if global_step >= MAX_TRAIN_STEPS:
                break
    
    # Save final model
    if accelerator.is_main_process:
        # Unwrap model
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Save LoRA weights
        lora_state_dict = {
            k: v for k, v in unwrapped_unet.state_dict().items()
            if "lora_" in k
        }
        
        final_dir = os.path.join(OUTPUT_DIR, "final_lora_weights")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(
            lora_state_dict,
            os.path.join(final_dir, "pytorch_lora_weights.bin")
        )
        
        print("Training completed and model saved!")

if __name__ == "__main__":
    train_lora()


