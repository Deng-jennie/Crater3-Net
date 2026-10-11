"""Data/metadata adapter for the supplied model; not a GIS preprocessing pipeline."""
import json
from pathlib import Path

import numpy as np
import torch

from crater3_net import CraterDataset, MODEL_NAMES


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = json.load(stream)
    if cfg["model_name"] not in MODEL_NAMES:
        raise ValueError("Unknown model_name.")
    order = cfg["channel_order"]
    expected = {"DEM", "slope", "plan curvature"}
    if "3" in cfg["model_name"]:
        if len(order) != 3 or set(order) != expected:
            raise ValueError("Three-channel models require DEM, slope and plan curvature.")
    elif order != ["DEM"]:
        raise ValueError("This adapter uses DEM only for one-channel models.")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be positive.")
    return cfg


def load_arrays(path, channel_order, require_masks=True):
    with np.load(path, allow_pickle=False) as archive:
        images = archive["images"].copy()
        order = archive["channel_order"].tolist()
        if order != channel_order:
            raise ValueError(f"Channel order {order} does not match {channel_order}.")
        if images.ndim != 4 or images.shape[1] != len(channel_order):
            raise ValueError("images must have shape (N,C,H,W).")
        masks = archive["masks"].copy() if require_masks else None
        names = archive["class_names"].tolist()
        if len(names) != 5 or len(set(names)) != 5 or names[0] != "background":
            raise ValueError("class_names must list 5 distinct names; ID 0 is background.")
        if not len(images) or not np.isfinite(images).all():
            raise ValueError("images must be nonempty and finite; resolve NoData first.")
        if require_masks:
            CraterDataset(images, masks)
        ids = archive["sample_ids"].tolist() if "sample_ids" in archive else None
        if ids is not None and (len(ids) != len(images) or len(set(ids)) != len(ids)):
            raise ValueError("sample_ids must be unique and match sample count.")
    return images, masks, names, ids


def prepare_images(images):
    images = torch.as_tensor(images, dtype=torch.float32)
    images = torch.nn.functional.interpolate(images, size=(512, 512), mode="nearest")
    minimum = images.amin(dim=(-2, -1), keepdim=True)
    span = images.amax(dim=(-2, -1), keepdim=True) - minimum
    return (images - minimum) / span.clamp_min(torch.finfo(images.dtype).eps)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not {"state_dict", "config", "class_names"}.issubset(checkpoint):
        raise ValueError("Expected a checkpoint produced by this repository's train.py.")
    return checkpoint


def write_json(path, value):
    def clean(item):
        if isinstance(item, torch.Tensor):
            return clean(item.tolist())
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
