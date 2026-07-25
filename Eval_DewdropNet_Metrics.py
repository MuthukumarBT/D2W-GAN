import os
import cv2
import numpy as np
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import niqe
import argparse


# Function to calculate PSNR
def calculate_psnr(gt, output):
    mse = np.mean((gt - output) ** 2)
    if mse == 0:
        return 100  # No noise, perfect match
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Function to calculate SSIM
def calculate_ssim(gt, output):
    # Convert images to float type in the range [0, 1]
    gt_float = img_as_float(gt)
    output_float = img_as_float(output)
    ssim1 = ssim(gt_float, output_float, multichannel=True)
    return ssim1

# Function to calculate NIQE
def calculate_niqe(gt, output):
    # NIQE requires images to be in grayscale and in the range [0, 1]
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) / 255.0

    # NIQE expects a 2D grayscale image, not a 3D RGB image
    try:
        gt_niqe = niqe.niqe(gt_gray)
        output_niqe = niqe.niqe(output_gray)
    except Exception as e:
        # Catch any exception (error) and skip it
        print(f"An error occurred: {e}")
        # You can log the error or just pass to continue the execution
        gt_niqe = np.nan
        output_niqe = np.nan

    return gt_niqe, output_niqe


# Main function
def main(output_folder, gt_folder):
    # Get list of image files in the output folder (assuming _rain suffix)
    output_images = sorted(os.listdir(output_folder))

    # Initialize lists to store metric values
    psnr_values = []
    ssim_values = []
    gt_niqe_values = []
    output_niqe_values = []
    ceiq_values = []

    # Loop through each output image (assumed to have '_rain' suffix)
    for output_image in output_images:
        if output_image.endswith("_rain.png"):  # Assuming images are JPEG; change extension as needed
            # Create the corresponding ground truth filename by replacing '_rain' with '_clean'
            # gt_image = output_image.replace("_rain.png", "_clean.jpg") # test_b
            gt_image = output_image.replace("_rain", "_clean")  # test_a

            # Build the paths for both GT and output images
            gt_path = os.path.join(gt_folder, gt_image)
            output_path = os.path.join(output_folder, output_image)

            # Check if the corresponding GT image exists
            if os.path.exists(gt_path):
                # Read the ground truth and output images
                gt = cv2.imread(gt_path)  # Ground truth image
                output = cv2.imread(output_path)  # Network output image

                # Convert to RGB if needed (assuming images are in BGR by default in OpenCV)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                # Calculate PSNR
                psnr_value = calculate_psnr(gt, output)
                psnr_values.append(psnr_value)

                # Calculate SSIM
                ssim_value = calculate_ssim(gt, output)
                ssim_values.append(ssim_value)

                # # Calculate NIQE for both images (ground truth and network output)
                gt_niqe, output_niqe = calculate_niqe(gt, output)
                gt_niqe_values.append(gt_niqe)
                output_niqe_values.append(output_niqe)



                # Print the results for each image pair
                print(f"Image Pair: {output_image}")
                print(f"PSNR: {psnr_value} dB")
                print(f"SSIM: {ssim_value}")
                print(f"GT NIQE: {gt_niqe}")
                print(f"Output NIQE: {output_niqe}")
                print("-" * 40)
            else:
                print(f"Warning: Corresponding GT image for {output_image} not found.")

    # Optionally, you can calculate the average PSNR, SSIM, and NIQE over all images
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_gt_niqe = np.nanmean(gt_niqe_values)
    avg_output_niqe = np.nanmean(output_niqe_values)
    # Print the averages
    print("\nAverages for the entire dataset:")
    print(f"Average PSNR: {avg_psnr} dB")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average GT NIQE: {avg_gt_niqe}")
    print(f"Average Output NIQE: {avg_output_niqe}")


# Command line argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Raindrop Removal Models")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Path to the output folder containing the result images")
    parser.add_argument('--gt_folder', type=str, required=True,
                        help="Path to the ground truth folder containing the clean images")

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.output_folder, args.gt_folder)