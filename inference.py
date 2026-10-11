"""Predict class-ID arrays; output has no geographic referencing."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_arrays, load_checkpoint, prepare_images
from crater3_net import build_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New .npz prediction file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint["config"]
    images, _, names, _ = load_arrays(args.data, cfg["channel_order"], require_masks=False)
    if names != checkpoint["class_names"]:
        raise ValueError("class_names do not match the checkpoint.")
    model = build_model(cfg["model_name"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    results = []
    for start in range(0, len(images), args.batch_size):
        batch = prepare_images(images[start:start + args.batch_size])
        results.append(predict(model, batch).numpy().astype(np.uint8))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predictions=np.concatenate(results),
        class_names=np.asarray(names))
    print(f"Saved {len(images)} masks of shape 512x512 to {output}")


if __name__ == "__main__":
    main()
