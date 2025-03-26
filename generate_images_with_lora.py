import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import os
from PIL import Image

def generate_images_with_lora():
    # Paths
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_WEIGHTS_PATH = "soccer-corner-lora/final_lora_weights/pytorch_lora_weights.bin"
    OUTPUT_DIR = "generated_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading base model: {BASE_MODEL}")
    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Use DPM++ scheduler for better results
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        algorithm_type="sde-dpmsolver++", 
        use_karras_sigmas=True
    )
    
    pipe = pipe.to("cuda")
    
    print(f"Loading LoRA weights from: {LORA_WEIGHTS_PATH}")
    # Load LoRA weights
    lora_state_dict = torch.load(LORA_WEIGHTS_PATH, map_location="cpu")
    
    # Print keys for debugging
    print("Keys in saved state dict:")
    for key in list(lora_state_dict.keys())[:5]:  # Show first 5 keys
        print(key)
    print(f"Total keys: {len(lora_state_dict)}")
    
    # Try a simpler approach - load weights directly with adapter method
    print("Applying LoRA weights...")
    
    # For PEFT-trained LoRA weights
    from peft import LoraConfig, get_peft_model
    
    # Recreate the same LoRA config used in training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj", 
            "to_k", "to_q", "to_v", "to_out.0"
        ],
        bias="none",
        lora_dropout=0.0
    )
    
    # Apply LoRA to UNet
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    
    # Load weights
    pipe.unet.load_state_dict(lora_state_dict, strict=False)
    print("Applied LoRA weights using PEFT")
    
    # Set prompts to generate images
    prompts = [
        "Professional soccer match photograph, corner kick situation in a stadium with crowded stands, television broadcast view, clear white field markings on green pitch, mixture of attacking and defending players paired up in man-marking positions, high-definition broadcast quality showing player expressions and jersey details, officials in bright uniforms positioned to monitor the play",
        "Professional soccer match photograph, corner kick situation in a stadium with crowded stands, television broadcast view, clear white field markings on green pitch, attacking players making runs into dangerous areas while defenders track their movement, officials in bright uniforms positioned to monitor the play, high-definition broadcast quality showing player expressions and jersey details"
        "Professional soccer match photograph, corner kick situation in a stadium with crowded stands, television broadcast view, clear white field markings on green pitch, defensive line positioned at the edge of the six-yard box with attackers nearby, officials in bright uniforms positioned to monitor the play, packed stadium with passionate fans in the background",
        "Professional soccer match photograph, corner kick situation in a stadium with crowded stands, television broadcast view, clear white field markings on green pitch, players clustered near the penalty spot awaiting the corner delivery, advertising boards visible along the sidelines, high-definition broadcast quality showing player expressions and jersey details",

    ]
    
    # Generation parameters
    num_images_per_prompt = 1
    guidance_scale = 7.5
    num_inference_steps = 30
    
    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
        
        # Add negative prompt to avoid low-quality results
        negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy"
        
        # Set SDXL-specific parameters
        height = 768
        width = 768
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # Save the image
        image_path = os.path.join(OUTPUT_DIR, f"soccer_corner_{i+1}.png")
        image.save(image_path)
        print(f"Image saved to {image_path}")
        
    print("Image generation complete!")

if __name__ == "__main__":
    generate_images_with_lora()