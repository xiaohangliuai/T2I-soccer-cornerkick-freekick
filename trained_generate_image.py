import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Load base model
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
pipe.to("cuda")

# Load LoRA weights and fix the key names
lora_path = "soccer-corner-lora/final_lora_weights/pytorch_lora_weights.bin"
state_dict = torch.load(lora_path, map_location="cpu")

# Create a new state dict with fixed keys
fixed_state_dict = {}
for key, value in state_dict.items():
    # Remove the 'base_model.model.' prefix from the keys
    if key.startswith("base_model.model."):
        new_key = key.replace("base_model.model.", "")
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value

# Load the fixed state dict
pipe.unet.load_state_dict(fixed_state_dict, strict=False)

# Generate image
prompt = "Corner kick scene from overlooking lens with blue team forming compact cluster in middle of penalty area. " \
"White team players positioned with two players wide left, three in central area, two at far post. Blue goalkeeper in yellow positioned at goal line."
image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

# Save the generated image
image.save("generated_corner_kick1.png")