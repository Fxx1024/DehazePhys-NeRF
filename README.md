# DehazePhys-NeRF

Inference and image-quality evaluation for DehazePhys-NeRF.

## Environment

The reference environment uses Python 3.7.16, PyTorch 1.13.1, torchvision
0.14.1, CUDA 11.7, NumPy 1.21.5 and Pillow 9.3.0. An NVIDIA GPU with 8 GB
of memory supports the default ray chunk size of 2048.

```bash
conda create -n dehazephys-inference python=3.7.16
conda activate dehazephys-inference
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

## Models

[Download the scene models from Google Drive](https://drive.google.com/drive/folders/1C1Y2CIRqKksdyr379Sqwk7VvkAa7Qogc).
Each scene has an independent ZIP archive of approximately 35 MB. Download one
scene or all ten, verify the archive against `MODEL_SHA256SUMS.txt`, then extract
each archive into the repository directory. Existing scene directories can coexist.
`--scene all` renders every scene currently extracted under `models/`.
The expected layout is:

```text
models/
  Bear/
    weights.npz
    cameras.npz
    scene.json
  Elephant/
  ...
```

Each scene contains an array-only model, its test-camera geometry, and rendering
settings. Model arrays are loaded with `allow_pickle=False`. The camera bundle
is sufficient to render the supplied test views; source RGB and depth datasets
are not needed for this operation. Verify the downloaded archive against the
published SHA-256 checksum before extraction.

| Scene | Test views |
| --- | ---: |
| Bear | 47 |
| Elephant | 79 |
| Lion | 47 |
| RocksLegoFog | 10 |
| GrassDrumsFog | 10 |
| DesertFicusFog | 10 |
| Counter | 30 |
| Bonsai | 37 |
| Garden | 24 |
| Bicycle | 25 |
| **Total** | **319** |

## Render

```bash
# All scenes
python render.py --models models --scene all --output results --chunk 2048

# One scene
python render.py --models models --scene Bear --output results --chunk 2048

# One view, or a selected range
python render.py --scene Bear --start 0 --count 1
```

The command writes complete images to `results/<scene>/raw/` and images ready
for evaluation to `results/<scene>/evaluation/`. File names are `view_000.png`,
`view_001.png`, and so on. For Elephant the evaluation crop is
`[0, 124, 320, 504]`; for Lion it is `[0, 16, 285, 418]`. Coordinates follow
`[left, top, right, bottom]`, with right and bottom excluded. Other scenes use
the full image.

Use `--chunk 1024` or `--chunk 512` if GPU memory is limited. The reference
commands use FP32 without TF32 or mixed precision. Different devices, numerical
libraries and chunk sizes may cause small quantization differences. CPU execution
is available with `--device cpu` but is substantially slower.

To render another camera, provide intrinsics, camera-to-world extrinsics and
image sizes in the same normalized scene coordinates as `cameras.npz`. Scene
models are not general-purpose single-image dehazing models.

## Evaluate

Install the optional metric dependencies:

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --predictions results/Bear/evaluation --references references/Bear --output results/Bear/metrics.csv
```

Provide clear reference images with exactly the same `view_XXX.png` names,
ordering, resolution and crop as the predictions. Reference images are not
included in the model archive. The evaluator requires matching filenames and
does not resize images or silently pair differently named files.

The metrics are RGB PSNR, PIQA SSIM and PIQA LPIPS with the AlexNet backbone,
using RGB tensors in `[0, 1]`. LPIPS may download its public backbone weights on
first use. For PSNR alone, no optional metric dependencies are required:

```bash
python evaluate.py --predictions results/Bear/evaluation --references references/Bear --metrics psnr
```

Each CSV contains one row per image and an arithmetic-mean row. JPEG decoding
versions can affect comparisons against reference images originally stored as
JPEG; preserve the reference decoder when comparing very small metric changes.

## Release validation

All 319 test-camera ray bundles were checked against the reference implementation.
The corresponding forward-output check covered 40,832 rays with identical values.
One complete image per scene was rendered and compared pixel by pixel; all ten
were identical to the reference outputs in the reference environment. The three
metric entry points were also executed successfully.

## License and acknowledgments

Code is distributed under Apache-2.0. Numerical routines and radiance-field
components derive from NeRF-Factory and MultiNeRF. See `LICENSE` and `NOTICE`
for attribution and terms.
