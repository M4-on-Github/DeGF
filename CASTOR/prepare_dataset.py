#!/usr/bin/env python3
"""
Build questions.jsonl from an image directory.

Supports flat layout (image_dir/file.jpg) and categorized layout
(image_dir/category/file.jpg). The same prompt is applied to every image.

Usage:
    python CASTOR/prepare_dataset.py \\
        --image-dir   /path/to/sorted_images \\
        --output      /data/$USER/castor_results/questions.jsonl \\
        --prompt-file CASTOR/prompts/promptv4.1.txt

The image directory is normally resolved by CASTOR/benchybench_paths.sh and
passed in as --image-dir; this script does not locate it itself.

An identical copy of this file lives in DeGF, ONLY and QWEN-Maritime. The
copies exist so each repo remains usable standalone; a shared import would
need to know the BenchyBench root before it could be loaded. They had already
drifted cosmetically once, so BenchyBench/tests/test_prepare_dataset.py runs
the same contract against every copy and asserts they produce identical output.
Do not edit one in isolation.
"""
import argparse
import json
import os


class ImageDatasetBuilder:
    """Discovers images under a directory and emits one question per image.

    Two directory layouts are supported and both appear in practice: a flat
    directory of images, and one subdirectory per casualty class. The class
    layout is what CASTOR uses, and the subdirectory name becomes part of the
    `image` value ("aground/00017.jpg") — that string is the join key against
    human_gt.csv, so it must stay relative to the image root rather than
    absolute.

    ORDERING IS PART OF THE CONTRACT. question_id is assigned positionally,
    so traversal is sorted at every level. If ordering ever became dependent
    on filesystem enumeration order, question_ids would shift between runs and
    answers could no longer be joined against previously generated results —
    silently, with nothing raising.
    """

    #: Compared against a lower-cased filename, so lower-case entries here
    #: match upper-case files too. The dataset does contain e.g. 00039.JPG.
    VALID_EXT = (".jpg", ".jpeg", ".png")

    def __init__(self, image_dir, valid_ext=None):
        self.image_dir = image_dir
        self.valid_ext = tuple(e.lower() for e in (valid_ext or self.VALID_EXT))

    def is_image(self, filename):
        return filename.lower().endswith(self.valid_ext)

    def discover(self):
        """Return image paths relative to image_dir, in deterministic order.

        Raises FileNotFoundError if the directory is absent — worth failing
        loudly, since an empty result would otherwise look like a successful
        run over zero images.
        """
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError("Image directory not found: %s" % self.image_dir)

        entries = []
        for item in sorted(os.listdir(self.image_dir)):
            item_path = os.path.join(self.image_dir, item)
            if os.path.isdir(item_path):
                for fname in sorted(f for f in os.listdir(item_path) if self.is_image(f)):
                    entries.append("%s/%s" % (item, fname))
            elif self.is_image(item):
                entries.append(item)
        return entries

    def build_records(self, prompt, entries=None):
        """Pair each discovered image with the prompt and a positional id."""
        if entries is None:
            entries = self.discover()
        return [{"question_id": i, "image": rel, "text": prompt}
                for i, rel in enumerate(entries)]

    @staticmethod
    def write(output_file, records):
        """Write records as JSONL, creating the output directory if needed.

        The output normally lands under /data/$USER/castor_results/, which may
        not exist on a first run.
        """
        parent = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(parent, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


class PromptSource:
    """Resolves the prompt from a file, an inline string, or the default.

    A file wins over an inline string; argparse already makes them mutually
    exclusive, so the order only decides what happens if this is called
    directly.
    """

    DEFAULT = "Describe this image in detail."

    def __init__(self, prompt=None, prompt_file=None):
        self.prompt = prompt
        self.prompt_file = prompt_file

    def resolve(self):
        if self.prompt_file:
            with open(self.prompt_file, encoding="utf-8") as f:
                # Stripped so a trailing newline does not become part of the
                # prompt the model sees.
                return f.read().strip()
        if self.prompt:
            return self.prompt
        return self.DEFAULT


def prepare(image_dir: str, output_file: str, prompt: str):
    """Build and write questions.jsonl. Facade over ImageDatasetBuilder."""
    builder = ImageDatasetBuilder(image_dir)
    entries = builder.discover()
    print("Found %d images in %s" % (len(entries), image_dir))
    builder.write(output_file, builder.build_records(prompt, entries))
    print("Written %d entries to %s" % (len(entries), output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", default=None,
                              help="Text file whose contents become the prompt")
    args = parser.parse_args()

    prompt = PromptSource(prompt=args.prompt, prompt_file=args.prompt_file).resolve()
    prepare(args.image_dir, args.output, prompt)


if __name__ == "__main__":
    main()
