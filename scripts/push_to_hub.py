"""
push_to_hub.py
==============
Upload trained adapter checkpoints to Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py --repo_id <your-hf-username>/qwen2.5-coder-3b-text2sql-sft
    python scripts/push_to_hub.py --repo_id <your-hf-username>/qwen2.5-coder-3b-text2sql-sft --private
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to Hugging Face Hub")
    parser.add_argument("--checkpoint_dir", default="checkpoints/sft", help="Path to checkpoint folder")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo ID (e.g. username/repo-name)")
    parser.add_argument("--private", action="store_true", help="Create as a private repository")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set in .env or shell.")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    print(f"Uploading adapter from: {args.checkpoint_dir}")
    print(f"Target HF Repository  : {args.repo_id} (private={args.private})")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Exclude intermediate checkpoint folders, only upload top-level adapter files
    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*/**", "checkpoint-*"],
    )

    print(f"\nSuccessfully uploaded to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

