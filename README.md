# GPT-2-XL, mT5-XL, and BERT-BASE-Cased Predictive Modeling of SEEG Data

This repository contains code to analyze data from Misra et al.'s 4-word-task experiment and generate predictive LLM models of the SEEG gamma potential.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Running the Regression Pipeline](#running-the-regression-pipeline)
- [Tunable Parameters](#tunable-parameters)
- [Authors](#authors)

---

## Getting Started

### Prerequisites
- Python 3.x
- Node.js / npm
- Access to the Kreiman Lab Dropbox (`Dropbox.com/KreimanLab`)

### Clone the Repository
```bash
git clone https://link-to-project
cd my-project
```

---

## Installation

**Install Node dependencies:**
```bash
npm install
```

**Set up a Python virtual environment and install requirements:**
```bash
python3 -m venv environment_name
source environment_name/bin/activate
pip3 install -r requirements.txt
```

**Download required data files** from the Kreiman Lab Dropbox:
- `Sub-Mat-Converted` files
- `4WT` directory

Then update the path variables in your local config to point to:
- Sub-Mat-Converted files
- `File_1` neural data
- `elecFinal` table

---

## Running the Regression Pipeline

### Step 1 — Download data
Download all `sub-mat-converted` data from the Alliyah folder in the 4WT Collab code folder.

### Step 2 — Configure paths
In `constants.py`, set the `EXPANSION_PATH` constant to point to your local `4WT-analysis` folder where the sub-mat-converted files are stored.

### Step 3 — Run preprocessing
```bash
python3 preprocessing.py
```
This performs gamma band filter extraction, trigger alignment, and sliding window feature extraction.

> ⏱ **Note:** Preprocessing takes ~20 minutes per subject. Consider running it in batches.

### Step 4 — Run regressions
Once all preprocessing files are created, run:
```bash
python3 pooling.py
```

### Step 5 — Explore and visualize results
Use the following notebooks to recreate the figures from the thesis:
- `time_window_analysis.ipynb`
- `TimeWindowNeuroVisualization.ipynb`

### Step 6- Diagnostics and Troubleshooting
- Use run `diagnostic.py` if any issues with pickle file keys in `pooling.py` emerge:
```bash
python3 diagnostic.py
```


## Tunable Parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `step_ms` and window size | `preprocessing.py` → `extract_sliding_window_features()` | Controls sliding window granularity |
| LLM models | `pooling.py` | Swap in any open-source HuggingFace model in place of GPT-2-XL, T5, or BERT-Base-Cased |
| `n_permutations` | `ridge.py` | Number of permutations for ridge regression (default: 100) |
| Plotting & stats | `time_window_analysis.ipynb`, `TimeWindowNeuroVisualization.ipynb` | Modify plots or add statsmodeling for auditory vs. visual variables |

---

## Authors

- [Alliyah Steele](https://github.com/HelloUniversePyC/)

## File Structure
```├── __pycache__
├── helpers
│   └── __pycache__
└── results_objs
    ├── a
    ├── GNS
    ├── GS
    ├── NGNS
    ├── overall
    └── v```
