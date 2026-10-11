"""Train the supplied model on an explicit train/validation/test split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import load_arrays, read_config, write_json
from crater3_net import CraterDataset, TrainingConfig, build_model, fit, set_seed


def validate_split(split, count):
    flattened = []
    for name in ("train", "valid", "test"):
        values = split[name]
        if not values or any(type(i) is not int or i < 0 or i >= count for i in values):
            raise ValueError(f"Invalid or empty {name} split.")
        flattened.extend(values)
    if len(flattened) != count or len(set(flattened)) != count:
        raise ValueError("Splits must be disjoint and cover all samples exactly once.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Numeric .npz dataset")
    parser.add_argument("--split", required=True, help="JSON index lists: train, valid, test")
    parser.add_argument("--config", default="configs/crater3_net.json")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = read_config(args.config)
    images, masks, names, ids = load_arrays(args.data, cfg["channel_order"])
    with open(args.split, encoding="utf-8") as stream:
        split = json.load(stream)
    validate_split(split, len(images))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(cfg["seed"])
    model = build_model(cfg["model_name"])
    loaders = []
    for i, name in enumerate(("train", "valid")):
        dataset = CraterDataset(images, masks, indices=split[name], augment=i == 0)
        loaders.append(DataLoader(dataset, batch_size=cfg["batch_size"],
            shuffle=i == 0, num_workers=0,
            generator=torch.Generator().manual_seed(cfg["seed"] + i)))
    config = TrainingConfig(**{key: cfg[key] for key in
        ("epochs", "learning_rate", "lr_factor", "lr_patience", "seed")})
    result = fit(model, *loaders, config=config, device=args.device,
        callback=lambda record: print(json.dumps(record), flush=True))
    checkpoint = {"state_dict": result["state_dict"], "config": cfg,
        "class_names": names, "best_epoch": result["best_epoch"],
        "best_valid_loss": result["best_valid_loss"]}
    torch.save(checkpoint, output / "best.pt")
    write_json(output / "history.json", result["history"])
    write_json(output / "split.json", split)
    write_json(output / "run_metadata.json", {"config": cfg, "class_names": names,
        "sample_ids": ids, "torch_version": str(torch.__version__),
        "preprocessing": "nearest resize to 512x512; per-sample/channel min-max 0..1"})
    print(f"Saved best validation-loss checkpoint to {output / 'best.pt'}")


if __name__ == "__main__":
    main()
