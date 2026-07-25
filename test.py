import os
import time
import argparse

import cv2
import torch

import utils
from model.find_model import find_model


def test(args):
    """Run inference on the test dataset."""

    testsets = ["test_b"]

    ckpt_dir = os.path.join(args.ckpt_dir, args.model)

    # Load model
    model, _ = find_model(args.model, "test")

    epoch = model.load(ckpt_dir, epoch=args.ckpt_epoch)
    print(f"Loaded {args.model} checkpoint from epoch {epoch}")

    total_params = sum(
        p.numel() for p in model.generator.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {total_params:,}")

    inference_times = []

    for testset in testsets:

        img_dir = os.path.join(args.dataset_dir, testset, "data")
        img_files = utils.get_image_file_list(img_dir)

        for idx, filename in enumerate(img_files, start=1):

            img = cv2.imread(os.path.join(img_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_tensor = utils.numpy2tensor(img)

            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            output = model.test_one_image(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_time = time.time() - start_time

            inference_times.append(elapsed_time)

            print(
                f"{testset} {idx}/{len(img_files)} | "
                f"{filename} | "
                f"Inference: {elapsed_time:.4f} s | "
                f"Average: {sum(inference_times)/len(inference_times):.4f} s"
            )

            # Optional resizing strategies
            if args.resize == "square":

                resized_img = cv2.resize(img, (512, 512))
                input_tensor = utils.numpy2tensor(resized_img)
                output = model.test_one_image(input_tensor)

            elif args.resize == "expand":

                rows, cols = img.shape[:2]
                expanded_img = utils.expand_size(img, 256)

                input_tensor = utils.numpy2tensor(expanded_img)
                output = model.test_one_image(input_tensor)

                for key in output:
                    output[key] = utils.restore_size(output[key], rows, cols)

            elif args.resize == "original":

                input_tensor = utils.numpy2tensor(img)
                output = model.test_one_image(input_tensor)

            # Save visualization
            save_dir = os.path.join(
                args.save_dir,
                args.model,
                testset,
                f"{epoch}_{args.resize}",
            )

            utils.save_outputs(
                save_dir=save_dir,
                filename=f"{filename[:-4]}.png",
                outputs=output,
                max_display=3,
            )

            # Save final restored image
            output_dir = os.path.join(save_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(
                os.path.join(output_dir, f"{filename[:-4]}.png"),
                cv2.cvtColor(
                    output["output"].squeeze() * 255,
                    cv2.COLOR_RGB2BGR,
                ),
            )

    print("Testing completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="D2WGAN")

    parser.add_argument(
        "--model",
        default="proposed",
        type=str,
        help="Model name.",
    )

    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        type=str,
        help="Checkpoint directory.",
    )

    parser.add_argument(
        "--ckpt_epoch",
        default=400,
        type=int,
        help="Checkpoint epoch to evaluate.",
    )

    parser.add_argument(
        "--dataset_dir",
        default="./data",
        type=str,
        help="Dataset directory.",
    )

    parser.add_argument(
        "--save_dir",
        default="./results",
        type=str,
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--resize",
        default="original",
        choices=["original", "square", "expand"],
        help="Input resizing strategy.",
    )

    args = parser.parse_args()

    test(args)