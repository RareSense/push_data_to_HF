
import os
import json
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfFolder

# Hugging Face Token
HF_TOKEN = "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"  # Replace with your Hugging Face token
HfFolder.save_token(HF_TOKEN)

# Paths
face_images_dir = "vi"  # Directory containing images
json_file_path = "output_captions.json"  # Path to output_captions.json
output_dir = "output_images"  # Directory to save target images (optional)
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load JSON captions
with open(json_file_path, 'r') as file:
    json_data = [json.loads(line) for line in file]

# Create a mapping of target images to captions
caption_map = {os.path.basename(item["target"]): item["caption"] for item in json_data}

# Step 2: Prepare data for the Hugging Face Dataset
rows = []  # List to store rows of the table

for file_name in os.listdir(face_images_dir):
    if file_name.endswith(".jpg"):
        file_base = os.path.splitext(file_name)[0]  # Get the base name without extension
        if file_base.isdigit():  # If it's a whole number (e.g., 1, 2, 3...)
            face_image = os.path.join(face_images_dir, file_name)  # Full path to face image
            # Find target images for this face image
            target_images = [
                f for f in os.listdir(face_images_dir)
                if f.startswith(file_base + ".") and f.endswith(".jpg")
            ]
            for target_image in target_images:
                target_image_path = os.path.join(face_images_dir, target_image)  # Full path
                caption = caption_map.get(target_image)  # Get caption

                # Skip rows where the caption is not found
                if caption:
                    rows.append({
                        "face_images": face_image,
                        "target_images": target_image_path,
                        "captions": caption
                    })

# Step 3: Create a Hugging Face Dataset
features = Features({
    "face_images": HFImage(),
    "target_images": HFImage(),
    "captions": Value("string"),
})

hf_dataset = Dataset.from_dict(
    {
        "face_images": [row["face_images"] for row in rows],
        "target_images": [row["target_images"] for row in rows],
        "captions": [row["captions"] for row in rows],
    },
    features=features,
)

# Step 4: Push Dataset to Hugging Face Hub
dataset_name = "sahal42/Faces_Dataset"  # Desired dataset name
hf_dataset.push_to_hub(dataset_name, token=HF_TOKEN, private=False)

print(f"Dataset pushed successfully to Hugging Face Hub: {dataset_name}")
