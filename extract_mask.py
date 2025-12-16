import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2  # For show_mask function
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Helper function to display and save mask
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # Model configuration
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"  # Correct path from current working directory
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    print("Loading model...")
    start_load = time.time()
    # Build SAM2 model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_load = time.time()
    print(f"Model load time: {end_load - start_load:.4f} seconds")

    # Load image
    image_path = "notebooks/images/cars.jpg"
    image = Image.open(image_path)
    image_np = np.array(image.convert("RGB"))

    print("Starting inference...")
    
    # Measure Image Encoding Time
    start_encoding = time.time()
    # Set image for predictor (Image Encoder)
    predictor.set_image(image_np)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_encoding = time.time()
    print(f"Image Encoding time: {end_encoding - start_encoding:.4f} seconds")

    # Define input point (x, y) and label (1 for foreground)
    input_point = np.array([[750, 750]])
    input_label = np.array([1])

    # Measure Mask Prediction Time
    start_prediction = time.time()
    # Predict masks (Prompt Encoder + Mask Decoder)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_prediction = time.time()
    print(f"Mask Prediction time: {end_prediction - start_prediction:.4f} seconds")

    # Select the best mask
    sorted_ind = np.argsort(scores)[::-1]
    best_mask = masks[sorted_ind[0]]
    best_score = scores[sorted_ind[0]]

    print(f"Predicted mask with score: {best_score:.3f}")

    # Save the original image with the predicted mask overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    show_mask(best_mask, plt.gca(), random_color=False, borders=True)
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Predicted Mask (Score: {best_score:.3f})", fontsize=18)
    plt.axis("off")
    output_image_path = "cars_segment_mask.png"
    plt.savefig(output_image_path)
    plt.close()  # Close the plot to prevent it from being displayed if run in a non-interactive environment

    print(f"Segment mask saved to {output_image_path}")


if __name__ == "__main__":
    main()
