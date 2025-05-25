import random
import os
import re

image_dir = "Dataset/GenerateImages/Corner-Kick/V2"
caption_file = "Dataset/Corner-kick/captions.txt"

# Attacker positions (white team)
attacker_positions = [
    "with two at near post, three in middle, two at far post",
    "with three players near penalty spot, two at far post, two at edge of box",
    "with two players wide left, three in central area, two at far post",
    "with three attackers on far side, three in central area, two hovering at edge of box",
    "with three attackers clustered near penalty spot, two at far post, two providing width",
    "with three players at penalty spot, two at far post, two at near post",
    "with three players at center, two at far post, two providing support from distance"
]

# Defender formations (blue team)
defender_formations = [
    "forming defensive wall clustered centrally",
    "forming staggered line",
    "forming compact cluster in middle of penalty area",
    "defending in mixed zonal/man formation with concentration on right side",
    "forming tight marking pairs with white attackers",
    "creating central barrier, with three providing width near posts",
    "positioning in tight cluster on right side of penalty area"
]

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

def numerical_sort(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# Sort the files numerically
image_files.sort(key=numerical_sort)

with open(caption_file, 'w') as f:
    for i, image_name in enumerate(image_files):
        # Use modulo to cycle through the arrays if we have more images than preset descriptions
        index = i % len(defender_formations)
        defender_formation = defender_formations[index]
        attacker_position = attacker_positions[index]
        
        # Generate more focused prompt with overlooking lens
        prompt = f"Corner kick scene from overlooking lens with blue team {defender_formation}. White team players positioned {attacker_position}. Blue goalkeeper in yellow positioned at goal line."
        
        f.write(f"{image_name}\t{prompt}\n")

print(f"Successfully generated {len(image_files)} prompts in {caption_file}")