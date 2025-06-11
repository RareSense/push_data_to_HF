from huggingface_hub import HfApi
import os

# 1. Set your token (you can also export HUGGINGFACE_TOKEN env var)
token = os.getenv("HUGGINGFACE_TOKEN", "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")

# 2. Initialize API client
api = HfApi()

# 3. Upload the file
api.upload_folder(
    folder_path="/home/bilal/sahal/Flux_fill_Incontext/pattern_transfer_10k/checkpoint-10000/transformer",
    repo_id="SnapwearAI/print-mockup",
    repo_type="model",          # <username>/<repo_name>
    token=token
)

print("✅ Model file uploaded!")