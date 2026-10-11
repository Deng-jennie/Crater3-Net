
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


MODEL_NAMES = ("Crater3-Net", "Crater1-Net", "U3-Net", "U1-Net")
INPUT_CHANNELS = ("DEM", "slope", "plan curvature")
WIDTHS = (16, 32, 64, 128, 256)
DROPOUT_RATES = (0.1, 0.1, 0.2, 0.2, 0.3)


def set_seed(seed: int = 666) -> None:
    """Seed PyTorch and disable cuDNN benchmarking."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InputBlock(nn.Module):
    """One independent 3 x 3 convolution per terrain channel."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(1, WIDTHS[0], kernel_size=3, padding=1)
            for _ in range(in_channels)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(WIDTHS[0], WIDTHS[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATES[0]),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [
            F.relu(branch(x[:, i:i + 1]), inplace=False)
            for i, branch in enumerate(self.branches)
        ]
        return self.refine(torch.stack(features, dim=0).sum(dim=0))


class ConvBlock(nn.Sequential):
    """Two 3 x 3 convolutions with the dropout shown in Figure 4."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )


class CraterUNet(nn.Module):
    """Shared U-shaped topology; configurations differ in inputs/upsampling."""

    def __init__(self, model_name: str = "Crater3-Net") -> None:
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}.")
        self.model_name = model_name
        self.in_channels = 3 if "3" in model_name else 1
        self.num_classes = 5
        self.use_unpool = model_name.startswith("Crater")
        self.input_block = InputBlock(self.in_channels)
        self.encoder = nn.ModuleList(
            ConvBlock(WIDTHS[i - 1], WIDTHS[i], DROPOUT_RATES[i])
            for i in range(1, len(WIDTHS))
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upsamplers = nn.ModuleList(
            nn.Identity() if self.use_unpool else
            nn.ConvTranspose2d(
                WIDTHS[i + 1], WIDTHS[i + 1], kernel_size=2, stride=2
            )
            for i in range(len(WIDTHS) - 2, -1, -1)
        )
        self.decoder = nn.ModuleList(
            ConvBlock(WIDTHS[i] + WIDTHS[i + 1], WIDTHS[i], DROPOUT_RATES[i])
            for i in range(len(WIDTHS) - 2, -1, -1)
        )
        self.classifier = nn.Conv2d(WIDTHS[0], self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits of shape (N, 5, H, W)."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected (N, {self.in_channels}, H, W) input.")
        if min(x.shape[-2:]) < 16 or any(s % 16 for s in x.shape[-2:]):
            raise ValueError("Spatial dimensions must be multiples of 16.")
        skips, pooling_indices = [], []
        x = self.input_block(x)
        for block in self.encoder:
            skips.append(x)
            x, indices = self.pool(x)
            pooling_indices.append(indices)
            x = block(x)
        for block, upsampler, skip, indices in zip(
            self.decoder, self.upsamplers,
            reversed(skips), reversed(pooling_indices)
        ):
            if self.use_unpool:
                repeats = x.shape[1] // indices.shape[1]
                expanded_indices = indices.repeat(1, repeats, 1, 1)
                output_size = (x.shape[0], x.shape[1], *skip.shape[-2:])
                x = self.unpool(x, expanded_indices, output_size=output_size)
            else:
                x = upsampler(x)
            x = block(torch.cat((skip, x), dim=1))
        return self.classifier(x)


class Crater3Net(CraterUNet):
    def __init__(self) -> None:
        super().__init__("Crater3-Net")


class Crater1Net(CraterUNet):
    def __init__(self) -> None:
        super().__init__("Crater1-Net")


class U3Net(CraterUNet):
    def __init__(self) -> None:
        super().__init__("U3-Net")


class U1Net(CraterUNet):
    def __init__(self) -> None:
        super().__init__("U1-Net")


def build_model(model_name: str = "Crater3-Net") -> CraterUNet:
    constructors = dict(zip(MODEL_NAMES, (Crater3Net, Crater1Net, U3Net, U1Net)))
    if model_name not in constructors:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}.")
    return constructors[model_name]()


class CraterDataset(Dataset):
    """In-memory terrain samples; no data locations or file readers."""

    def __init__(self, images, masks, indices=None, augment: bool = False):
        self.images = torch.as_tensor(images).detach().cpu()
        self.masks = torch.as_tensor(masks).detach().cpu()
        if self.images.ndim != 4 or self.images.shape[1] not in (1, 3):
            raise ValueError("images must have shape (N, 1 or 3, H, W).")
        expected = (len(self.images), *self.images.shape[-2:])
        if tuple(self.masks.shape) != expected:
            raise ValueError("masks must have shape (N, H, W), matching images.")
        if not len(self.images) or not torch.isfinite(self.images).all():
            raise ValueError("images must be nonempty and finite.")
        if not torch.isfinite(self.masks).all() or not torch.equal(
            self.masks, self.masks.long().to(self.masks.dtype)
        ):
            raise ValueError("Masks must contain finite integer class IDs.")
        if self.masks.min() < 0 or self.masks.max() > 4:
            raise ValueError("Mask IDs must be in 0..4.")
        self.indices = list(range(len(self.images))) if indices is None else list(indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.indices[index]
        image = self.images[sample].float()
        mask = self.masks[sample].long()
        image = F.interpolate(
            image.unsqueeze(0), size=(512, 512), mode="nearest"
        ).squeeze(0)
        minimum = image.amin(dim=(-2, -1), keepdim=True)
        span = image.amax(dim=(-2, -1), keepdim=True) - minimum
        image = (image - minimum) / span.clamp_min(torch.finfo(image.dtype).eps)
        mask = F.interpolate(
            mask[None, None].float(), size=(512, 512), mode="nearest"
        )[0, 0].long()
        if self.augment:
            for axis in (-2, -1):
                if torch.rand(()).item() < 0.5:
                    image, mask = image.flip(axis), mask.flip(axis)
        target = F.one_hot(mask, num_classes=5).permute(2, 0, 1).float()
        return image, target


def make_dataloaders(images, masks, batch_size: int = 16, seed: int = 666):
    """Random sample-level 8:1:1 split; augment the training set only.

    Each sample must be a distinct crater. Supply your own loaders if samples
    share a crater or a spatial group, to avoid splitting that group.
    """
    if len(images) < 10:
        raise ValueError("At least 10 samples are required for an 8:1:1 split.")
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    n_train, n_valid = int(0.8 * len(order)), int(0.1 * len(order))
    groups = (order[:n_train], order[n_train:n_train + n_valid],
              order[n_train + n_valid:])
    datasets = [
        CraterDataset(images, masks, indices=group.tolist(), augment=i == 0)
        for i, group in enumerate(groups)
    ]
    return tuple(
        DataLoader(
            dataset, batch_size=batch_size, shuffle=i == 0, num_workers=0,
            generator=torch.Generator().manual_seed(seed + i)
        )
        for i, dataset in enumerate(datasets)
    )


class DiceLoss(nn.Module):
    """One-hot/sigmoid Dice objective, following the supplied loss setup.

    Five output channels include background. Absent target classes contribute
    zero, matching the source library's multilabel Dice convention.
    """

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.shape != target.shape:
            raise ValueError("Dice targets must be one-hot and match logits.")
        probabilities = logits.sigmoid()
        dims = (0, 2, 3)
        intersection = (probabilities * target).sum(dims)
        denominator = probabilities.sum(dims) + target.sum(dims)
        score = 2.0 * intersection / denominator.clamp_min(1e-7)
        present = target.sum(dims) > 0
        return ((1.0 - score) * present).mean()


def confusion_matrix(prediction: Tensor, reference: Tensor) -> Tensor:
    """Rows are reference classes; columns are predicted classes."""
    encoded = 5 * reference.reshape(-1).long() + prediction.reshape(-1).long()
    return torch.bincount(encoded, minlength=25).reshape(5, 5)


def segmentation_metrics(matrix: Tensor) -> dict:
    """Exclude background from means, not from foreground FP/FN counts.

    mIoU includes foreground classes present in reference OR prediction.
    mPA includes only foreground classes present in reference.
    An undefined mean (background-only reference/prediction) is NaN.
    """
    matrix = matrix.double()
    true_positive = matrix.diag()[1:]
    reference = matrix.sum(dim=1)[1:]
    prediction = matrix.sum(dim=0)[1:]
    union = reference + prediction - true_positive
    nan = torch.full_like(true_positive, float("nan"))
    iou = torch.where(union > 0, true_positive / union.clamp_min(1), nan)
    cpa = torch.where(reference > 0, true_positive / reference.clamp_min(1), nan)
    return {
        "miou": iou.nanmean().item(),
        "mpa": cpa.nanmean().item(),
        "class_iou": iou.detach().cpu().tolist(),
        "class_pixel_accuracy": cpa.detach().cpu().tolist(),
    }


def train_one_epoch(model, loader, optimizer, loss_function, device) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        total_samples += len(images)
    if not total_samples:
        raise ValueError("Training loader is empty.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, loss_function, device) -> dict:
    """Return pooled-pixel metrics and separate mean per-crater metrics."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    matrix = torch.zeros((5, 5), dtype=torch.long, device=device)
    crater_scores = []
    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        logits = model(images)
        loss = loss_function(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss.")
        total_loss += loss.item() * len(images)
        total_samples += len(images)
        predictions, references = logits.argmax(dim=1), targets.argmax(dim=1)
        for prediction, reference in zip(predictions, references):
            sample_matrix = confusion_matrix(prediction, reference)
            matrix += sample_matrix
            scores = segmentation_metrics(sample_matrix)
            crater_scores.append((scores["miou"], scores["mpa"]))
    if not total_samples:
        raise ValueError("Evaluation loader is empty.")
    result = segmentation_metrics(matrix)
    per_crater = torch.tensor(crater_scores, dtype=torch.float64).nanmean(dim=0)
    result.update(
        loss=total_loss / total_samples,
        mean_crater_miou=per_crater[0].item(),
        mean_crater_mpa=per_crater[1].item(),
        confusion_matrix=matrix.cpu(),
    )
    return result


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 150
    learning_rate: float = 0.001
    lr_factor: float = 0.5
    lr_patience: int = 3
    seed: int = 666


def fit(
    model: CraterUNet,
    train_loader,
    valid_loader,
    config: Optional[TrainingConfig] = None,
    device=None,
    callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Train, restore the lowest-validation-loss state, and return history.

    Seed BEFORE build_model() as well to make initialization reproducible.
    Test data are deliberately not consumed during fitting.
    """
    config = config or TrainingConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    loss_function = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    history, best_state = [], None
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation = evaluate(model, valid_loader, loss_function, device)
        scheduler.step(validation["loss"])
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        record = {
            "epoch": epoch, "learning_rate": lr, "train_loss": train_loss,
            "valid_loss": validation["loss"], "valid_miou": validation["miou"],
            "valid_mpa": validation["mpa"],
        }
        history.append(record)
        if callback is not None:
            callback(deepcopy(record))
    model.load_state_dict(best_state)
    return {
        "model_name": model.model_name, "best_epoch": best_epoch,
        "best_valid_loss": best_loss, "state_dict": best_state, "history": history,
    }


@torch.no_grad()
def predict(model: CraterUNet, images: Tensor) -> Tensor:
    """Return mutually exclusive class IDs for already-preprocessed inputs."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        return model(images.to(device=device, dtype=torch.float32)).argmax(dim=1).cpu()
    finally:
        model.train(was_training)
