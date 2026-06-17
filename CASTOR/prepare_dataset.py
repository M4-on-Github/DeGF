#!/usr/bin/env python3
"""
Build questions.jsonl from an image directory.

Supports flat layout (image_dir/file.jpg) and categorized layout
(image_dir/category/file.jpg). The same prompt is applied to every image.

Usage:
    python CASTOR/prepare_dataset.py \\
        --image-dir  CASTOR/shipwreck_wiki_images/sorted_images \\
        --output     CASTOR/shipwreck_wiki_images/questions.jsonl \\
        --prompt-file CASTOR/prompts/prompt.txt
"""
import argparse
import json
import os

VALID_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def prepare(image_dir: str, output_file: str, prompt: str):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    entries = []
    for item in sorted(os.listdir(image_dir)):
        item_path = os.path.join(image_dir, item)
        if os.path.isdir(item_path):
            imgs = sorted(f for f in os.listdir(item_path) if f.lower().endswith(VALID_EXT))
            for fname in imgs:
                entries.append(f"{item}/{fname}")
        elif item.lower().endswith(VALID_EXT):
            entries.append(item)

    print(f"Found {len(entries)} images in {image_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, rel_path in enumerate(entries):
            f.write(json.dumps({
                "question_id": i,
                "image": rel_path,
                "text": prompt,
            }) + "\n")

    print(f"Written {len(entries)} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", default=None,
                               help="Text file whose contents become the prompt")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = "Describe this image in detail."

    prepare(args.image_dir, args.output, prompt)


if __name__ == "__main__":
    main()
