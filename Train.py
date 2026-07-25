import os
import argparse
from datetime import datetime
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from model.find_model import find_model


def train(args):
    # Create checkpoint directory
    ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    os.makedirs(ckpt_dir, exist_ok=True)

    utils.print_args(args)
    utils.save_log(ckpt_dir, "train", args)

    # Initialize model and dataset
    model, dataset = find_model(
        model_name=args.model,
        phase="train",
        dataset_dir=args.dataset_dir,
        use_transform=True,
        learning_rate=args.learning_rate,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    start_epoch = 1

    # Resume training (optional)
    if False:
        last_epoch = model.load(ckpt_dir, epoch=450)
        if last_epoch > 1:
            start_epoch = last_epoch + 1
            print(f"Resuming training from epoch {last_epoch}")

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable multi-GPU training when available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    max_batches = len(data_loader)

    for epoch in range(start_epoch, args.max_epochs + 1):
        start_time = time()
        losses_list = []

        for batch, data in enumerate(data_loader, start=1):

            if isinstance(model, nn.DataParallel):
                model.module.set_inputs(data)
                model.module.train_on_batch()
            else:
                model.set_inputs(data)
                model.train_on_batch()

            if batch == 1 or batch % 10 == 0 or batch == max_batches:

                if isinstance(model, nn.DataParallel):
                    losses_list.append(model.module.get_losses())
                else:
                    losses_list.append(model.get_losses())

                utils.print_losses(
                    epoch,
                    args.max_epochs,
                    batch,
                    max_batches,
                    losses_list,
                    title=f"[{args.model}] TRAIN",
                    mode="last",
                )

        elapsed_time = time() - start_time

        utils.print_losses(
            epoch,
            args.max_epochs,
            batch,
            max_batches,
            losses_list,
            title=f"[{args.model}: {elapsed_time:.3f}s] TRAIN MEAN LOSS",
            mode="mean",
        )

        utils.save_losses(ckpt_dir, epoch, losses_list, mode="mean")

        outputs = (
            model.module.get_outputs()
            if isinstance(model, nn.DataParallel)
            else model.get_outputs()
        )

        utils.save_outputs(
            save_dir=os.path.join(args.save_dir, args.model, "train"),
            filename=f"{epoch:04d}.png",
            outputs=outputs,
            max_display=3,
        )

        if epoch % 500 == 0:
            if isinstance(model, nn.DataParallel):
                model.module.save(ckpt_dir, epoch)
            else:
                model.save(ckpt_dir, epoch)

    print(f"Training completed successfully: {datetime.now()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DeRainDrop")

    parser.add_argument(
        "--train_continue",
        action="store_false",
        dest="train_continue",
        help="Resume training from the latest checkpoint.",
    )

    parser.add_argument(
        "--model",
        default="proposed",
        type=str,
        help="Model name.",
    )

    parser.add_argument(
        "--max_epochs",
        default=2000,
        type=int,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Training batch size.",
    )

    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="Number of data loading workers.",
    )

    parser.add_argument(
        "--lr",
        default=2e-4,
        type=float,
        dest="learning_rate",
        help="Learning rate.",
    )

    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        type=str,
        help="Directory for saving checkpoints.",
    )

    parser.add_argument(
        "--dataset_dir",
        default="./data/train",
        type=str,
        help="Path to the training dataset.",
    )

    parser.add_argument(
        "--save_dir",
        default="./results",
        type=str,
        help="Directory for saving output images.",
    )

    args = parser.parse_args()
    train(args)