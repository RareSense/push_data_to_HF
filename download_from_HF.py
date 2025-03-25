from datasets import load_dataset
import os
from tqdm import tqdm

# Function to download dataset images
def download_images_from_hf(dataset_name, save_dir, token=None):
    dataset = load_dataset(dataset_name, split="train", token=token)
    os.makedirs(save_dir, exist_ok=True)
    
    for i, data in enumerate(tqdm(dataset, desc='Saving Images')):
        try:
            # Save target image
            target_image_path = os.path.join(save_dir, f"target_{i}.jpg")
            data["target"].save(target_image_path, format="JPEG")

            # Save source image
            source_image_path = os.path.join(save_dir, f"source_{i}.jpg")
            data["source"].save(source_image_path, format="JPEG")

            # Save caption
            caption_text = data["caption"]
            with open(os.path.join(save_dir, "captions.txt"), "a", encoding="utf-8") as f:
                f.write(f"{i}: {caption_text}\n")
        except Exception as e:
            print(f"Error saving image {i}: {e}")

    print("✅ Download complete!")

# Example usage
dataset_name = "raresense/textile_swatch_to_product"
download_images_from_hf(dataset_name, save_dir="downloaded_images", token="token_id")
