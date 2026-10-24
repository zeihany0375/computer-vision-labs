# LAB01 - Image Processing with OpenCV

CSE480/CSE482 Lab Assignment 01, Mechatronics Engineering and Automation Program.

This lab implements selective colour isolation, a ghostly image overlay, vignette
filtering, and colour channel swapping using Python, OpenCV, and NumPy.

## Contents

- `lab01.py`: all four tasks, with individual task selection.
- `images/`: the four input images required by the assignment.
- `docs/assignment.pdf`: original task requirements.
- `docs/lab_report.pdf`: methods, result figures, observations, and conclusions.
- `requirements.txt`: Python dependencies.

## Setup and run

From this repository, open a terminal in `LAB01`:

```powershell
cd LAB01
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\lab01.py
```

On macOS or Linux, use `.venv/bin/python` in place of
`.\.venv\Scripts\python.exe`.

Run a single task, or select other input and output folders:

```powershell
.\.venv\Scripts\python.exe .\lab01.py --task 1
.\.venv\Scripts\python.exe .\lab01.py --input-dir images --output-dir outputs
```

`--task` accepts `all`, `1`, `2`, `3`, or `4`; the default is `all`.
The default input directory is `images/` beside the script, and the default
output directory is `outputs/`. The report describes the original experiment;
the input images have been organized into `images/` for this repository.

## Experiments

| Task | Method | Main output |
| --- | --- | --- |
| 1 | Grayscale conversion, BGR splitting, a red threshold and dominance mask, and bitwise compositing | `task1_selective_colour.png` |
| 2 | Resize zebra to cameraman dimensions and blend with `cv2.addWeighted` at alpha 0.35, 0.50, and 0.65 | `task2_blend_alpha_*.png` |
| 3 | Blur a filled circular mask and multiply normalized weights into the rover image | `task3_vignette.png` |
| 4 | Split and merge zebra channels in three alternative orders | `task4_*.png` |

Running all tasks creates 14 PNG files, including intermediate masks, output
variations, and `comparison.png`, which shows the original and main result for
each task. Generated outputs are ignored by Git because they can be recreated;
the report already includes the result figures.

The red mask selects pixels by colour rather than segmenting an entire apple,
so some orange or yellow areas remain coloured. The overlay blends the full
zebra background, and resizing changes its aspect ratio. The zebra is mostly
neutral, so channel swapping produces relatively subtle colour changes.

## Validation

The original lab and this packaged implementation were run successfully with
OpenCV 5.0.0 and NumPy 2.5.4. All 14 generated images matched the original saved
outputs exactly. Task 1 uses a binary mask and leaves unselected pixels grayscale;
the vignette mask has 256 intensity levels. Dependencies are listed without
version pins; use the versions above when reproducing the validated environment.
