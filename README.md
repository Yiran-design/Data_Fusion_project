# Data_Fusion_project

A minimal, runnable skeleton for the research proposal:
**Freshness-Aware Semantic Scheduling for Multi-View Edge Inference in IoT Systems**.

This project implements a lightweight pilot experiment with:
- a small real dataset (`sklearn.datasets.load_digits`),
- three synthetic sensing views per sample,
- four scheduling policies (`random`, `aoi`, `semantic`, `joint`),
- fusion-aware edge inference with cache freshness,
- automatic plot and result generation.

## Project structure

```text
Data_Fusion_project/
├── README.md
├── requirements.txt
├── main.py
├── data_utils.py
├── scheduler.py
├── simulator.py
└── plot_results.py
```

## Quick start (local)

```bash
pip install -r requirements.txt
python main.py --episodes 200 --episode-len 5 --budgets 1 2 3 4
```

Outputs will be saved to `outputs/`:
- `results.csv`
- `accuracy_vs_budget.png`
- `avg_aoi_bar_budget2.png`
- `summary_budget2.txt`

## Run in Google Colab

Create a new Colab notebook and run:

```python
!git clone https://github.com/Yiran-design/Data_Fusion_project.git
%cd Data_Fusion_project
!pip install -r requirements.txt
!python main.py --episodes 200 --episode-len 5 --budgets 1 2 3 4
```

If you want to download the figures after running:

```python
from google.colab import files
files.download('outputs/accuracy_vs_budget.png')
files.download('outputs/avg_aoi_bar_budget2.png')
files.download('outputs/results.csv')
```

## Methods

The experiment compares four policies:
1. `random`: random selection under budget.
2. `aoi`: updates the stalest cached view first.
3. `semantic`: prioritizes views with higher confidence margin.
4. `joint`: combines semantic score, freshness need, diversity bonus, and cost penalty.

## Notes

- The experiment is intentionally lightweight so that it can run on CPU in Colab.
- You can increase `--episodes` for more stable results.
- You can later replace the toy view transforms with your own sensor views or image augmentations.
