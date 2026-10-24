"""CSE480/CSE482 Lab 01: four OpenCV image-processing tasks.

Run every task:  python lab01.py
Run one task:    python lab01.py --task 1
Results are written to outputs/ next to this script by default.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_image(folder, filename):
    path = folder / filename
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(folder, filename, image):
    path = folder / filename
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def task1(input_dir, output_dir):
    """Keep red-dominant apple pixels in colour on a grayscale background."""
    original = load_image(input_dir, "apple.png")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blue, green, red = cv2.split(original)

    # Bright red alone also includes yellow/white pixels. Require red to
    # exceed both other channels as well. Signed values avoid uint8 overflow.
    _, red_threshold = cv2.threshold(red, 130, 255, cv2.THRESH_BINARY)
    red16 = red.astype(np.int16)
    dominant = (red16 > green.astype(np.int16) + 20) & (
        red16 > blue.astype(np.int16) + 20
    )
    mask = cv2.bitwise_and(red_threshold, dominant.astype(np.uint8) * 255)

    foreground = cv2.bitwise_and(original, original, mask=mask)
    background = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
    result = cv2.bitwise_or(foreground, background)

    save_image(output_dir, "task1_grayscale.png", gray)
    save_image(output_dir, "task1_red_mask.png", mask)
    save_image(output_dir, "task1_selective_colour.png", result)
    return original, result


def task2(input_dir, output_dir):
    """Blend equally sized images with several transparency settings."""
    base = load_image(input_dir, "cameraman.jpg")
    ghost = load_image(input_dir, "zebra.png")
    height, width = base.shape[:2]
    ghost = cv2.resize(ghost, (width, height), interpolation=cv2.INTER_AREA)

    save_image(output_dir, "task2_resized_zebra.png", ghost)
    for alpha in (0.35, 0.50, 0.65):
        beta = 1.0 - alpha
        # Each output pixel = alpha * cameraman + beta * zebra.
        # Weights summing to one keep the overall brightness comparable.
        result = cv2.addWeighted(base, alpha, ghost, beta, 0)
        save_image(output_dir, f"task2_blend_alpha_{alpha:.2f}.png", result)
        if alpha == 0.65:
            preview = result
    return base, preview


def task3(input_dir, output_dir):
    """Blur a filled circular mask, then multiply it into the rover image."""
    original = load_image(input_dir, "marsrover.png")
    height, width = original.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(0.45 * min(height, width))
    cv2.circle(mask, (width // 2, height // 2), radius, 255, thickness=-1)
    soft_mask = cv2.GaussianBlur(
        mask, (0, 0), sigmaX=min(height, width) / 8.0
    )

    # A fractional mask smoothly scales brightness. A bitwise mask would
    # treat every nonzero value as selected and lose the soft transition.
    weights = soft_mask.astype(np.float32) / float(soft_mask.max())
    result = np.clip(
        original.astype(np.float32) * weights[:, :, np.newaxis], 0, 255
    ).astype(np.uint8)

    save_image(output_dir, "task3_circle_mask.png", mask)
    save_image(output_dir, "task3_soft_mask.png", soft_mask)
    save_image(output_dir, "task3_vignette.png", result)
    return original, result


def task4(input_dir, output_dir):
    """Reorder the zebra's blue, green and red channels."""
    original = load_image(input_dir, "zebra.png")
    blue, green, red = cv2.split(original)

    # cv2.merge expects BGR: the last entry becomes the displayed red channel.
    # Here the original green values become red, and red becomes green.
    result = cv2.merge((blue, red, green))
    swap_blue_red = cv2.merge((red, green, blue))
    rotate_channels = cv2.merge((green, red, blue))

    save_image(output_dir, "task4_swap_green_red.png", result)
    save_image(output_dir, "task4_swap_blue_red.png", swap_blue_red)
    save_image(output_dir, "task4_rotate_channels.png", rotate_channels)
    return original, result


def comparison_panel(image, title, width=520, height=300):
    """Fit an image below a caption without changing its aspect ratio."""
    panel = np.full((height + 40, width, 3), 245, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    fitted = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (width - fitted.shape[1]) // 2
    y = 40 + (height - fitted.shape[0]) // 2
    panel[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
    cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "1", "2", "3", "4"), default="all")
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = (task1, task2, task3, task4)
    titles = ("Selective colour", "Ghost overlay (alpha=0.65)",
              "Vignette", "Green / red channel swap")
    rows = []
    for number, (task, title) in enumerate(zip(tasks, titles), start=1):
        if args.task not in ("all", str(number)):
            continue
        original, result = task(args.input_dir, args.output_dir)
        rows.append(np.hstack((
            comparison_panel(original, f"Task {number}: original"),
            comparison_panel(result, title),
        )))
        print(f"Task {number} complete: {title}")

    comparison_name = "comparison.png" if args.task == "all" else f"task{args.task}_comparison.png"
    save_image(args.output_dir, comparison_name, np.vstack(rows))
    print(f"Saved images to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
