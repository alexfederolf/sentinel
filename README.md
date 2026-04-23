# SENTINEL

Based on Kaggle competition: [ESA-ADB Challenge](https://www.kaggle.com/competitions/esa-adb-challenge)

---

## Getting Started

**Requirements:** Python 3.10.6, pyenv

```bash
# 1. Clone the repo
mkdir ~/code/alexfederolf && cd "$_"
git clone git@github.com:alexfederolf/sentinel.git
cd sentinel

# 2. Set up the Python environment
pyenv virtualenv 3.10.6 sentinel
pyenv local sentinel

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Verify the setup
python -c "from sentinel.ml_logic.metrics import corrected_event_f05; print('OK')"
```

**Data:** Download the competition files from [Kaggle](https://www.kaggle.com/competitions/esa-adb-challenge/data) and place them in `data/raw/`:

```
data/raw/
├── train.parquet
├── test.parquet
├── target_channels.csv
└── sample_submission.parquet
```

---

## Running the Notebooks

Run in this order, each notebook saves outputs that the next one loads:

| # | Notebook | Description |
|---|---|---|
| 01 | `notebooks/01-eda.ipynb` | Exploratory data analysis |
| 02 | `notebooks/02-preprocessing.ipynb` | Scaling and windowing |
| 03 | `notebooks/03-baseline_iforest.ipynb` | Isolation Forest |
| 04 | `notebooks/04-baseline_pca.ipynb` | PCA reconstruction — best model |
