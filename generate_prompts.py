import random
import os
import re
from PIL import Image

image_dir = "Dataset/Corner-kick/images"
caption_file = "Dataset/Corner-kick/captions.txt"

# Base prompt components
base_prompt = "Professional soccer match photograph, corner kick situation in a stadium with crowded stands, television broadcast view, clear white field markings on green pitch"

# Position variations - these describe the player positions
position_variations = [
    "players clustered near the penalty spot awaiting the corner delivery",
    "players in blue jerseys and players in white jerseys positioned strategically in the penalty area",
    "players marking each other tightly in the goal area, preparing for an aerial challenge",
    "defensive line positioned at the edge of the six-yard box with attackers nearby",
    "players spread across the penalty area with focused attention on the corner taker",
    "mixture of attacking and defending players paired up in man-marking positions",
    "players jostling for position near the goal as the set piece is about to be taken",
    "tactical defensive formation with players creating a barrier in front of goal",
    "attacking players making runs into dangerous areas while defenders track their movement"
]

# Match elements - these describe the broadcast and stadium elements
match_elements = [
    "scoreboard visible displaying the current match status",
    "evening match under bright stadium lights",
    "advertising boards visible along the sidelines",
    "officials in bright uniforms positioned to monitor the play",
    "packed stadium with passionate fans in the background",
    "broadcast camera angle from elevated position showing tactical formations clearly",
    "stadium atmosphere captured with crowd anticipation visible",
    "high-definition broadcast quality showing player expressions and jersey details"
]

# Create a function to generate a unique prompt for each image
def generate_prompt():

    position = random.choice(position_variations)
    element1 = random.choice(match_elements)
    element2 = random.choice([e for e in match_elements if e != element1])
    
    prompt = f"{base_prompt}, {position}, {element1}, {element2}"
    return prompt

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

def numerical_sort(filename):

    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# Sort the files numerically
image_files.sort(key=numerical_sort)

with open(caption_file, 'w') as f:
    for image_name in image_files:
        
        prompt = generate_prompt()
        f.write(f"{image_name}\t{prompt}\n")

print(f"Successfully generated {len(image_files)} prompts in {caption_file}")