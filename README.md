# Hanzi OCR

An OCR project for Chinese characters (Hanzi), trained on synthetic images rendered from real fonts. The project is divided into three planned phases:

1. **Single characters** (in progress): classify an image of one character. The vocabulary comes from the [HSK 3.0 word lists](https://github.com/krmanik/HSK-3.0).
2. **Lines of text** (data generation done): read a line of Chinese, which usually mixes Hanzi, punctuation, Latin script and Arabic numerals. Training text comes from Chinese Wikipedia.
3. **Multiple lines**: read whole blocks of Chinese text.

## How it works

```
HSK word lists ──> make_char_list ──> character list
                                          │
fonts ────────────────────────> hanzi_image_generation ──> images + manifest
                                                                │
                                                     train_hanzi_model ──> checkpoint.pt + le.json
                                                                                  │
                                                                     api (FastAPI) ──> predictions
```

The model is a CNN that takes a 64×64 grayscale image and returns a probability for each character in the vocabulary. To measure how well it generalizes, whole fonts are held out for validation instead of random images.

## Project structure

```
src/hanzi_ocr/
├── hanzi_model.py           # CNN model
├── pipelines.py             # Augmentation and normalization, shared by training and inference
├── SyntheticHanziDataset.py # PyTorch dataset over the generated images
├── train_hanzi_model.py     # Training entry point
├── train_nn.py              # Training loop with early stopping
├── model_inference.py       # Load the model, preprocess images, predict
├── model_storage.py         # Download model files from S3
├── api.py                   # HTTP API
└── utils.py                 # Shared helpers (paths, text rendering, hanzi detection)
scripts/
├── make_char_list.py        # Build the character list from the HSK word lists
├── hanzi_image_generation.py
├── download_zh_wiki.py      # Download Chinese Wikipedia (phase 2)
└── line_image_generation.py # Render lines of Wikipedia text (phase 2)
tests/                       # pytest suite
data/                        # Not tracked by git, see below
```

## Setup

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the CPU build of PyTorch, which works everywhere. To train on an AMD GPU instead,
sync the `rocm` group, which replaces it with a ROCm build:

```bash
uv sync --no-group cpu --group rocm
```

The two are mutually exclusive, so `cpu` has to be disabled when `rocm` is enabled. Exporting
`UV_NO_GROUP=cpu` once makes every later command shorter, since `uv run` keeps the environment in
whichever state its flags ask for:

```bash
export UV_NO_GROUP=cpu
uv run --group rocm python -m hanzi_ocr.train_hanzi_model ...
```

Without those flags `uv run` syncs back to the CPU build, because `cpu` is the default group.
See [Training on a GPU](#training-on-a-gpu) for the rest of what an AMD card needs.

## Preparing the data

Everything is stored under `data/`, which is not tracked by git.

1. Download the HSK word lists (`HSK 1.tsv` to `HSK 7-9.tsv`) from [krmanik/HSK-3.0](https://github.com/krmanik/HSK-3.0) into `data/HSK_vocab_lists/`.
2. Put the fonts to render with (`.ttf` or `.otf`) into `data/fonts/`. Fonts must support Chinese characters, and characters missing from a font are logged instead of rendered. Anything with `reserved` in its name, such as a `data/fonts/reserved/` folder, is skipped.
3. Build the character list:

   ```bash
   uv run python scripts/make_char_list.py
   ```

4. Render one image per character and font:

   ```bash
   uv run python scripts/hanzi_image_generation.py --hanzi_styles s
   ```

   `--hanzi_styles` takes `s` (simplified), `t` (traditional) or both (`st`). Variable fonts are rendered once per weight listed in `--font_weights` (by default `Light Regular Bold`), and fonts with a single weight are rendered once. The options used are saved as `generation_parameters.json` next to the manifest. **This deletes any existing images and manifests in the output folders before generating new ones.** Characters a font cannot render are listed in `data/hanzi_images_manifest/error_manifest.csv`.

Every script accepts `--help` to list its options and defaults.

## Training

```bash
uv run python -m hanzi_ocr.train_hanzi_model \
  --model_name my_model \
  --holdout_fonts ZCOOLKuaiLe-Regular \
  --epochs 30
```

- `--holdout_fonts` lists the fonts used only for validation. Names ignore capitalization and the file extension.
- `--early_stopping_monitor` chooses what early stopping and the saved checkpoint follow: `valid_metric` (default) or `train_loss`.
- `--num_workers` (default 3) sets how many processes load and augment training images in parallel. Each worker gets its own augmentation seed, derived from `--random_seed`, so runs are reproducible as long as the number of workers stays the same. When training on the CPU, the workers compete with the model for cores, so a few workers usually work better than many. Use `0` to load images in the main process, which helps when debugging.
- `--device` chooses where training runs: `auto` (the default, picks a GPU when there is one), `cpu`, `cuda` or `mps`. The device that was actually used is recorded in `train_hyperparameters.json`.
- Other options include `--optimizer` (`adam`, `adamw`, `nag`, `momentum`), `--scheduler` (`performance`, `onecycle`), `--metric` (`accuracy`, `f1`, `f1 macro`), `--lr`, `--batch_size` and `--patience_epochs` for early stopping.

Note that `--scheduler performance` uses `ReduceLROnPlateau` with its own default patience of 10
epochs, so `--patience_epochs` should be comfortably higher than that or early stopping will end the
run before the learning rate is ever reduced.

### Training on a GPU

PyTorch reports both CUDA and ROCm cards through `torch.cuda`, so `--device auto` picks either one up
without any change. On an AMD card, sync the `rocm` group as described in [Setup](#setup).

Cards that ROCm does not officially support can still work by presenting themselves as a supported
architecture. A Radeon RX 6600 (`gfx1032`), for example, needs:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

which makes the runtime use the `gfx1030` kernels. Check that the build actually ships them before
trusting it, because the override cannot invent kernels that aren't in the wheel:

```bash
uv run --no-group cpu --group rocm python -c "import torch; print(torch.cuda.get_arch_list())"
```

If the architecture you are overriding to is missing from that list, the card won't work with that
wheel. The `rocm6.4` index pinned in `pyproject.toml` includes `gfx1030`; other ROCm channels vary,
and the newest torch release is usually unavailable there, so the ROCm and CPU groups sit on
different torch versions. Checkpoints are plain `state_dict`s and move between them fine.

Should a run crash or hang on the GPU, `dmesg` is the place to look: `amdgpu` ring timeouts or GPU
resets point at a driver fault rather than anything tunable in this project. A smaller
`--batch_size`, `MIOPEN_FIND_MODE=NORMAL` or `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` are
worth trying first, and `--device cpu` always works.

The results are saved in `data/model_weights/<model_name>/`:

| File | Contents |
|---|---|
| `checkpoint.pt` | Weights of the best epoch on the validation metric |
| `le.json` | Codepoint of each class, in the order of the model's outputs |
| `train_hyperparameters.json` | The options used for this run |
| `data_generation_parameters.json` | The options used to generate the training images, copied from the manifest folder |
| `training_history.json` | Loss and metrics per epoch |
| `training_history.png` | Plot of the loss and metrics, marking the epoch of the saved checkpoint |

`checkpoint.pt` and `le.json` must always be kept together: each model's `le.json` only matches its own checkpoint.

## Running the API

The API loads the model once at startup, either from a local folder or from S3, and is configured through environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | | Local folder with `checkpoint.pt` and `le.json`. If set, S3 is not used. |
| `MODEL_BUCKET` | | S3 bucket with the model. Required when `MODEL_DIR` is not set. |
| `MODEL_NAME` | `hanziOCR` | Folder of the model inside the bucket |
| `MODEL_VERSION` | `V1` | Version folder inside `MODEL_NAME` |
| `DEVICE` | `cpu` | Device for inference, e.g. `cpu` or `cuda` |

When using S3, the files are read from `s3://<MODEL_BUCKET>/<MODEL_NAME>/<MODEL_VERSION>/checkpoint.pt` and `.../le.json`. To deploy a new model, upload it under a new version and change `MODEL_VERSION`. Credentials are picked up by boto3 as usual, from your AWS configuration locally or from an IAM role on AWS.

To run it locally:

```bash
MODEL_DIR=data/model_weights/my_model uv run uvicorn hanzi_ocr.api:app --host 0.0.0.0 --port 8080
```

Interactive documentation is available at `http://localhost:8080/docs`.

### Endpoints

- `GET /health` returns `{"status": "ok"}` once the model is loaded.
- `POST /predict` takes an image file (PNG or JPEG) and an optional `top_k` (default 5), and returns the `top_k` most likely characters:

  ```bash
  curl -F "file=@character.png" "http://localhost:8080/predict?top_k=3"
  ```

  Example response:

  ```json
  [
    {"probability": 0.91, "simplified": "中", "traditional": "中"},
    {"probability": 0.04, "simplified": "申", "traditional": "申"},
    {"probability": 0.01, "simplified": "史", "traditional": "史"}
  ]
  ```

  Each prediction includes both the simplified and the traditional form of the character. The conversion between the two is done one character at a time, so characters with more than one traditional form (such as 发 → 發 or 髮) only show one of them.

  Invalid images or an invalid `top_k` return a `400` error.

The model was trained on synthetic images: one dark character, centered, on a light background. Images that look like that work best.

## Tests

```bash
uv run pytest
```

Tests never write to the real `data/` folder. A few of them read a font from `data/fonts/`.

## Roadmap

- Docker image for the API (in progress)
- Deployment to AWS (planned)
- Phase 2: line recognition, using the images from `line_image_generation.py`

## Data sources

- HSK 3.0 word lists: [krmanik/HSK-3.0](https://github.com/krmanik/HSK-3.0)
- Chinese Wikipedia: [YCWTG/wikipedia_zh_cleaned_latest](https://huggingface.co/datasets/YCWTG/wikipedia_zh_cleaned_latest) on Hugging Face
- Fonts are not distributed with this repository. Check each font's license before using it.
