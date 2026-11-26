# Are friend communities also taste communities?

DTU 02807 Computational Tools for Data Science — project on the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/).

The whole analysis is in one notebook, `notebooks/Computational_final.ipynb`. This README explains how to set the project up on your own computer and run it.

## Requirements

- **Python 3.11** (the package versions in `requirements.txt` are pinned for it)
- About **15 GB of free disk space**: ≈ 4 GB for the downloaded archive, ≈ 9 GB for the extracted files, plus the cached tables
- **8 GB of RAM** is enough (the large files are streamed, never loaded whole)
- Jupyter (JupyterLab, Jupyter Notebook or VS Code with the Jupyter extension)

## Project structure

```text
project/
├── notebooks/
│   └── Computational_final.ipynb   <- the full analysis (run this)
├── src/                            <- Python modules used by the notebook
├── tests/                          <- unit tests (pytest)
├── data/                           <- NOT in git, you create it (see below)
│   ├── raw/                        <- the 3 Yelp JSON files go here
│   ├── processed/                  <- created by the notebook
│   └── results/                    <- created by the notebook
├── figures/                        <- created by the notebook
├── requirements.txt
└── README.md
```

## 1. Get the code

```bash
git clone https://github.com/cniq182/Computational-tools-4-datasciense.git
cd Computational-tools-4-datasciense
```

## 2. Create a Python environment and install the packages

With `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab ipykernel
```

or with conda:

```bash
conda create -n dtu02807 python=3.11
conda activate dtu02807
pip install -r requirements.txt
pip install jupyterlab ipykernel
```

Install exactly the versions in `requirements.txt`. The community detection results depend on the library versions, and the notebook stops if they don't match the reference numbers (see *Troubleshooting*).

## 3. Download the data

The dataset is **not included in this repository**: it is too large for GitHub, and Yelp's Terms of Use do not allow redistributing it. Each team member downloads it separately.

1. Go to <https://business.yelp.com/data/resources/open-dataset/>, accept the Terms of Use and download the **JSON** version (`yelp_dataset.tar`, ≈ 4 GB).
2. Extract the archive:

   ```bash
   tar -xf yelp_dataset.tar
   ```

3. Only **3 of the 5 files** are needed. **Keep their original names**:

   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_review.json`
   - `yelp_academic_dataset_user.json`

   The other files (`..._checkin.json`, `..._tip.json`) are not used and can be deleted.

## 4. Put the data in place

Create `data/raw/` in the root of the repository and move the 3 files into it:

```bash
mkdir -p data/raw
mv /path/to/yelp_academic_dataset_{business,review,user}.json data/raw/
```

```text
data/
└── raw/
    ├── yelp_academic_dataset_business.json
    ├── yelp_academic_dataset_review.json
    └── yelp_academic_dataset_user.json
```

`data/processed/`, `data/results/` and `figures/` are created automatically on the first run.

**Data stored somewhere else?** Instead of moving the files, point the environment variable `YELP_DATA_RAW` to the folder that contains them before starting Jupyter:

```bash
export YELP_DATA_RAW="/path/to/yelp_dataset"     # Windows (PowerShell): $env:YELP_DATA_RAW="C:\path\to\yelp_dataset"
```

The `data/` folder is listed in `.gitignore`. **Never commit it.**

## 5. Run the notebook

```bash
jupyter lab
```

Open `notebooks/Computational_final.ipynb`, select the kernel of the environment from step 2, and run all cells (*Run → Run All Cells*).

- The first run takes about **10 minutes**. About 2 minutes of that is streaming the raw files, which are then cached in `data/processed/`.
- Each section reads the files written by the sections before it, so after the setup cells (section 0) you can re-run a single section once its inputs exist.

## 6. Run the tests (optional)

From the repository root:

```bash
pytest -q tests
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Yelp JSON files not found` | The 3 files are not in `data/raw/` (or under the folder in `YELP_DATA_RAW`), or they were renamed. Keep the original `yelp_academic_dataset_*.json` names. |
| `FileNotFoundError: Project root not found` | Open the notebook from inside the repository (root or `notebooks/`). |
| `AssertionError: regenerated data differ from the report` | Your package versions differ from `requirements.txt`. Reinstall with `pip install -r requirements.txt` in a fresh environment and re-run the whole notebook. |
| `ModuleNotFoundError` (e.g. `pytest`, `sklearn`) | The notebook kernel is not the environment from step 2. Select the right kernel, or install the requirements into the kernel's environment. |
