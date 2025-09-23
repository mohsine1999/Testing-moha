# KuaiRec Sequential Training Pipeline

This repository contains utilities to convert the [KuaiRec](https://kuairec.com/) dataset into [RecBole](https://recbole.io/) atomic files and to train sequential recommenders such as **SASRec**, **SASRecF**, and **FDSA** on top of them.

## 1. Prepare the dataset

1. Download and extract `KuaiRec.zip` so that the directory layout matches
   `KuaiRec/data/{small_matrix.csv,big_matrix.csv,user_features.csv,...}`.
2. Run the converter to materialise RecBole atomic files (`.inter`, `.user`, `.item`):

   ```bash
   python -m kuairec_pipeline.data_prep /path/to/KuaiRec/data \
       --output-dir data --dataset-name kuairec_small --matrix-size small \
       --with-side-info --min-watch-ratio 0.1
   ```

   *Use `--matrix-size big` to consume `big_matrix.csv`. Omitting `--with-side-info`
   generates only the interaction file.*

## 2. Train models with RecBole

Once the RecBole-formatted files are available in `data/kuairec_small/` the training entry points can be launched as follows:

- **Baseline SASRec** (interaction-only):

  ```bash
  python -m kuairec_pipeline.train_sasrec \
      --data-path data --dataset-name kuairec_small \
      --output-dir saved --epochs 100 --learning-rate 0.001
  ```

  Uses `configs/kuairec_sasrec.yaml` and mirrors the original baseline provided with this repository.

- **Feature-aware SASRecF** (item side-information appended to the sequence):

  ```bash
  python -m kuairec_pipeline.train_sasrecf \
      --data-path data --dataset-name kuairec_small \
      --output-dir saved --config-file configs/kuairec_sasrecf.yaml
  ```

  Requires the `.item` file produced by running the data preparation step with `--with-side-info`. The default configuration concatenates video categories, upload types, durations, tags, and daily engagement statistics to the item embeddings.

- **FDSA** (feature-level attention over item attributes):

  ```bash
  python -m kuairec_pipeline.train_fdsa \
      --data-path data --dataset-name kuairec_small \
      --output-dir saved --config-file configs/kuairec_fdsa.yaml
  ```

  Also depends on the `.item` file and leverages the same feature set as SASRecF while allowing deeper attention over the attribute channels.

All scripts accept the same command-line overrides (learning rate, epochs, batch sizes, etc.). Passing a different RecBole YAML config via `--config-file` enables custom hyper-parameters.

## 3. Python environment

Install the Python dependencies in a virtual environment (Python ≥ 3.9):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`RecBole` will automatically pull the matching version of PyTorch. If GPU training
is required ensure a compatible CUDA wheel of PyTorch is installed before running
`pip install recbole`.

## 4. Outputs

- `data/<dataset_name>/<dataset_name>.inter`: sequential interactions sorted by
  timestamp with watch/play metadata.
- `data/<dataset_name>/<dataset_name>.user`: optional user side features (if
  `--with-side-info` is passed).
- `data/<dataset_name>/<dataset_name>.item`: optional item features summarising
  the latest per-video statistics.
- `saved/`: RecBole checkpoints and logs for the chosen training run (SASRec, SASRecF, or FDSA).

## 5. Reproducing experiments

For reproducibility the default config enforces deterministic behaviour via
`seed: 2020` and `reproducibility: true`. Adjust `device: cpu` to `cuda` when
running on GPUs.
