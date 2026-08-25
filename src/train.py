import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

from dataset import HandwritingDataset, collate_fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trocr(model, processor):
    tokenizer = processor.tokenizer

    decoder_start = tokenizer.cls_token_id

    if decoder_start is None:
        decoder_start = tokenizer.bos_token_id

    if decoder_start is None:
        raise RuntimeError(
            "Could not determine decoder_start_token_id."
        )

    model.config.decoder_start_token_id = decoder_start
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.sep_token_id is not None:
        model.config.eos_token_id = tokenizer.sep_token_id


def evaluate_loss(model, loader, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            pixel_values = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            batch_size = pixel_values.size(0)

            total_loss += (
                outputs.loss.item()
                * batch_size
            )

            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]

    set_seed(
        int(train_cfg["seed"])
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TrOCR FINE-TUNING")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # ---------------------------------------------------------
    # Load processor + model
    # ---------------------------------------------------------

    model_name = cfg["model_name"]

    print("\nLoading processor...")

    processor = TrOCRProcessor.from_pretrained(
        model_name
    )

    print("Loading model...")

    model = VisionEncoderDecoderModel.from_pretrained(
        model_name
    )

    configure_trocr(
        model,
        processor,
    )

    model.to(device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = HandwritingDataset(
        "data/processed/train.csv",
        processor,
        max_target_length=int(
            train_cfg["max_target_length"]
        ),
    )

    val_dataset = HandwritingDataset(
        "data/processed/val.csv",
        processor,
        max_target_length=int(
            train_cfg["max_target_length"]
        ),
    )

    print("\nDataset:")
    print(
        "  Train:",
        len(train_dataset),
    )

    print(
        "  Validation:",
        len(val_dataset),
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    loader_args = {
        "num_workers": int(
            train_cfg["num_workers"]
        ),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            train_cfg["batch_size"]
        ),
        shuffle=True,
        **loader_args,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            train_cfg["eval_batch_size"]
        ),
        shuffle=False,
        **loader_args,
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=float(
            train_cfg["learning_rate"]
        ),
        weight_decay=float(
            train_cfg["weight_decay"]
        ),
    )

    epochs = int(
        train_cfg["epochs"]
    )

    accumulation = int(
        train_cfg[
            "gradient_accumulation_steps"
        ]
    )

    steps_per_epoch = max(
        (
            len(train_loader)
            + accumulation
            - 1
        )
        // accumulation,
        1,
    )

    total_steps = (
        steps_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_steps
        * float(
            train_cfg["warmup_ratio"]
        )
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---------------------------------------------------------
    # Mixed precision
    # ---------------------------------------------------------

    use_amp = (
        bool(train_cfg["fp16"])
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # ---------------------------------------------------------
    # Checkpoint directory
    # ---------------------------------------------------------

    save_dir = Path(
        train_cfg["save_dir"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    print("\nTraining configuration:")
    print(
        "  Epochs:",
        epochs,
    )

    print(
        "  Batch size:",
        train_cfg["batch_size"],
    )

    print(
        "  Learning rate:",
        train_cfg["learning_rate"],
    )

    print(
        "  FP16:",
        use_amp,
    )

    print(
        "  Gradient accumulation:",
        accumulation,
    )

    print("\nStarting training...\n")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            pixel_values = batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=True,
            )

            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss = (
                    outputs.loss
                    / accumulation
                )

            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            scaler.scale(
                loss
            ).backward()

            should_step = (
                step % accumulation == 0
                or step == len(train_loader)
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
                * accumulation
            )

        # -----------------------------------------------------
        # Epoch metrics
        # -----------------------------------------------------

        train_loss = (
            running_loss
            / max(
                len(train_loader),
                1,
            )
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # -----------------------------------------------------
        # Save epoch checkpoint
        # -----------------------------------------------------

        epoch_dir = (
            save_dir
            / f"epoch-{epoch}"
        )

        model.save_pretrained(
            epoch_dir
        )

        processor.save_pretrained(
            epoch_dir
        )

        # -----------------------------------------------------
        # Save best model
        # -----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_dir = (
                save_dir / "best"
            )

            model.save_pretrained(
                best_dir
            )

            processor.save_pretrained(
                best_dir
            )

            print(
                "  New best checkpoint:",
                best_dir,
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "Best model:",
        save_dir / "best",
    )


if __name__ == "__main__":
    main()