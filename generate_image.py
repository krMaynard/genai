#!/usr/bin/env python3
"""Generate images locally using the Gemini image generation API."""

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-3-pro-image-preview"

DEFAULT_PROMPT = """
Create a clear, professional flowchart diagram illustrating a language learning workflow. Please use the following structure, actors, and flow:

- LLM generates or summarizes interesting text
- LLM translates text
- TTS model creates audio
- LLM creates vocabulary list
- Image model creates illustrations with vocabulary words

Repeat in a cycle.

Future state: immersive environment personalized to language learning (e.g., Roblox)
"""


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        print("Run: export GOOGLE_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    return key


def generate_image(client: genai.Client, prompt: str, output_dir: str) -> str | None:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        ),
    )

    if not response.candidates or not response.candidates[0].content:
        print("Error: No content generated. The request may have been blocked.", file=sys.stderr)
        return None

    image_filename = None
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"Model: {part.text.strip()}")
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            random_number = random.randint(0, 10000)
            image_filename = os.path.join(output_dir, f"{MODEL_NAME}_{random_number}.png")
            image.save(image_filename)
            if sys.platform == "darwin":
                subprocess.run(["open", image_filename], check=False)
            print(f"Saved: {image_filename}")

    return image_filename


def main():
    parser = argparse.ArgumentParser(description="Generate images with Gemini")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Image generation prompt")
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory to save images (default: current dir)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client = genai.Client(api_key=get_api_key())

    print(f"Generating {args.count} image(s) in parallel…")
    with ThreadPoolExecutor(max_workers=args.count) as executor:
        futures = {
            executor.submit(generate_image, client, args.prompt, args.output_dir): i + 1
            for i in range(args.count)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Image {idx} failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
