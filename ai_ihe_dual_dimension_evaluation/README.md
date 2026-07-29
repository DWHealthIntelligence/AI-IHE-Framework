<!-- # Evaluation of AI-Assisted Decision Support Framework for Individualized Health Examination
 -->


## 📌 2. Dual-Dimensional Evaluation Framework for Strategy-Oriented AI-assisted Decision Support 

This repository contains the source code for the **Dual-Dimensional Evaluation Framework** presented in the manuscript *"Development and Evaluation of an AI-Assisted Decision Support Framework for Individualized Health Examination"*. 

The codebase implements a comprehensive analytical pipeline to evaluate both **Strategy-level Usability** and **Decision-making Utility** of the AI-IHE system.

<!-- The evaluation is divided into two primary dimensions: -->
1. **Strategy-level Usability:** Compares the clinical appropriateness of AI-developed strategies against Package-based strategies using an expert-defined reference standard.
2. **Decision-making Utility:** Evaluates whether AI assistance improves the quality (precision, recall) and efficiency (time) of clinical decision-making across healthcare providers with varying levels of experience.

## 📂 Repository Structure
```text
ai_ihe_dual_dimension_evaluation/
├── data/
│   ├── raw/                 # Raw evaluation records (reviews, decisions, timestamps)
│   └── processed/           # Processed datasets (e.g., Expert Gold Standard)
├── results/
│   ├── tables/              # Exported CSV tables
│   └── figures/             # Exported visualization charts
├── src/
│   ├── utils.py                           # Core statistical and metric calculation functions
│   ├── 01_establish_ref_standard.py       # Consensus processing for reference standard
│   ├── 02_strategy_usability.py           # Dimension 1: Usability metrics pipeline
│   ├── 03_decision_utility_quality.py     # Dimension 2: Decision support utility quality metrics & Visualizations
│   └── 04_decision_utility_efficiency.py  # Dimension 2: Decision support utility efficiency (Time) metrics
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
└── .gitignore               # Git ignore file type
```

## 🚀 Reproducibility
**Data Privacy Note:** The original evaluation data involves identifiable health information and cannot be shared publicly due to privacy regulations.

**Code Access:** To ensure complete methodological transparency and code reproducibility for peer review, all scripts used for evaluation experiments analysis are open sourced, any questions about the AI-assisted Individual Health Examination Decision Support System can be requested from the authors of the paper.

## 📊 Execution Pipeline
Step 1: Establish the Expert-Defined Reference Standard

Processes the independent expert reviews and consensus decisions to establish the reference standard.
```bash
# Execute the analytical modules sequentially under the project root directory
python -m src.01_establish_ref_standard
```
*Outputs:* ``data/processed/gold_standard.csv``

Step 2: Strategy-level Usability

Calculates and compares the Precision and Recall of AI-developed vs. Package-based strategies
```bash
python -m src.02_strategy_usability
```
*Outputs:* ``results/tables/table_usability.csv``

Step 3: Decision Support Utility (Quality)

Performs paired analysis of healthcare provider decisions (Manual vs. AI-assisted) across item categories and provider roles.
```bash
python -m src.03_decision_utility_quality
```
*Outputs:* ``results/tables/table_decision_utility_quality.csv``, ``results/figures/confusion_matrix.png``, ``results/figures/individual_metrics.png``

Step 4: Decision Support Utility (Efficiency)

Calculates the reduction in time required for examination strategy development under AI assistance.
```bash
python -m src.04_decision_utility_efficiency
```
*Outputs:* ``results/tables/table_decision_utility_efficiency.csv``
