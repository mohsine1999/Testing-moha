# KuaiRec SASRec Training Pipeline

This repository contains utilities to convert the [KuaiRec](https://kuairec.com/) dataset into [RecBole](https://recbole.io/) atomic files and to train the sequential recommender **SASRec** on top of them.

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

## 2. Train SASRec with RecBole

Once the RecBole-formatted files are available in `data/kuairec_small/` the
training entry point can be launched with:

```bash
python -m kuairec_pipeline.train_sasrec \
    --data-path data --dataset-name kuairec_small \
    --output-dir saved --epochs 100 --learning-rate 0.001
```

The command uses the baseline hyperparameters defined in
`configs/kuairec_sasrec.yaml`. Any parameter exposed by RecBole can be overridden
on the command line, e.g.:

```bash
python -m kuairec_pipeline.train_sasrec --learning-rate 0.0005 --epochs 50
```

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
- `saved/`: RecBole checkpoints and logs for the SASRec training run.

## 5. Reproducing experiments

For reproducibility the default config enforces deterministic behaviour via
`seed: 2020` and `reproducibility: true`. Adjust `device: cpu` to `cuda` when
running on GPUs.
