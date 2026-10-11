"""Evaluate one explicit held-out split using the supplied metric implementation."""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from common import load_arrays, load_checkpoint, write_json
from crater3_net import CraterDataset, DiceLoss, build_model, evaluate
from train import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset", choices=("valid", "test"), default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New JSON metrics file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint["config"]
    images, masks, names, _ = load_arrays(args.data, cfg["channel_order"])
    if names != checkpoint["class_names"]:
        raise ValueError("class_names do not match the checkpoint.")
    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    validate_split(split, len(images))
    model = build_model(cfg["model_name"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    loader = DataLoader(CraterDataset(images, masks, indices=split[args.subset]),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = evaluate(model, loader, DiceLoss(), args.device)
    result.update(class_names=names, subset=args.subset,
        sample_count=len(split[args.subset]),
        note="Metrics describe this run, not a reproduction claim for manuscript tables.")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
