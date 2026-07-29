"""
Step 2 - Evaluate Strategy-level Usability
Compares AI-developed strategies vs. Package-based strategies against the Gold Standard
Result for Table "Comparison of AI-assisted individualized examination strategies and health examination packages for strategy-level usability evaluation".

@author: DanWu and Agreewithu (Ruixin Dai)
"""


import pandas as pd
from pathlib import Path
from src.utils import (
    calculate_metrics,
    get_confusion_matrix_elements,
    calculate_mean_and_ci,
    format_metric_ci,
    perform_paired_test,
    format_p_value
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_STANDARD_PATH = BASE_DIR / "data" / "processed" / "reference_standard.csv"
STRATEGY_DATA_PATH = BASE_DIR / "data" / "raw" / "strategy_data.csv"
OUTPUT_TABLE_PATH = BASE_DIR / "results" / "tables" / "table_usability.csv"


def evaluate_strategy_usability():
    """
    Main pipeline to evaluate Strategy-level Usability.
    """
    print("--- Step 2: Strategy-level Usability Evaluation (Table 2) ---")
    
    # Ensure Reference Standard exists
    if not REFERENCE_STANDARD_PATH.exists():
        raise FileNotFoundError(f"Reference standard missing. Please run 01_establish_reference_standard.py first.")
    
    df_gold = pd.read_csv(REFERENCE_STANDARD_PATH)
    
    cases = df_gold['case_id'].unique()
    
    # Data storage for case-level metrics
    results = {
        'case_id': [],
        'ai_precision': [], 'ai_recall': [],
        'pkg_precision': [], 'pkg_recall': []
    }
    
    # 1. Calculate Case-Level Metrics
    for case in cases:
        # Get item sets
        gold_set = set(df_gold[df_gold['case_id'] == case]['item_id'])

        df_strat = pd.read_csv(STRATEGY_DATA_PATH)
        
        ai_set = set(df_strat[(df_strat['case_id'] == case) & 
                              (df_strat['strategy_type'] == 'AI-assisted')]['item_id'])
        
        pkg_set = set(df_strat[(df_strat['case_id'] == case) & 
                               (df_strat['strategy_type'] == 'Package-based')]['item_id'])
        
        # Confusion matrices
        tp_ai, fp_ai, fn_ai = get_confusion_matrix_elements(ai_set, gold_set)
        tp_pkg, fp_pkg, fn_pkg = get_confusion_matrix_elements(pkg_set, gold_set)
        
        # Metrics
        p_ai, r_ai = calculate_metrics(tp_ai, fp_ai, fn_ai)
        p_pkg, r_pkg = calculate_metrics(tp_pkg, fp_pkg, fn_pkg)
        
        results['case_id'].append(case)
        results['ai_precision'].append(p_ai)
        results['ai_recall'].append(r_ai)
        results['pkg_precision'].append(p_pkg)
        results['pkg_recall'].append(r_pkg)
        
    df_results = pd.DataFrame(results)
    
    # 2. Compute Statistics and Hypothesis Testing
    statistic_data = []
    
    for metric in ['precision', 'recall']:
        col_ai = f'ai_{metric}'
        col_pkg = f'pkg_{metric}'
        
        # Mean and CI
        stat_ai = calculate_mean_and_ci(df_results[col_ai])
        stat_pkg = calculate_mean_and_ci(df_results[col_pkg])
        
        # Paired Differences (AI - Package)
        diff_array = df_results[col_ai] - df_results[col_pkg]
        stat_diff = calculate_mean_and_ci(diff_array)
        
        # Paired statistical test
        p_val, test_name = perform_paired_test(df_results[col_ai], df_results[col_pkg])
        
        print(f"Strategy Usability Metric: {metric}, Test Type: {test_name}")

        # Format for output table
        row = {
            'Metric': f"{metric.capitalize()}, mean (95% CI), %",
            'Package-based strategies': format_metric_ci(stat_pkg['mean'], stat_pkg['lower_ci'], stat_pkg['upper_ci']),
            'AI-assisted strategies': format_metric_ci(stat_ai['mean'], stat_ai['lower_ci'], stat_ai['upper_ci']),
            'Difference': format_metric_ci(stat_diff['mean'], stat_diff['lower_ci'], stat_diff['upper_ci']),
            'P value': format_p_value(p_val)
        }
        statistic_data.append(row)
        
    # 3. Save and Display Table "Comparison of AI-assisted individualized examination strategies and health examination packages for strategy-level usability evaluation"
    df_table_ai_vs_pkg = pd.DataFrame(statistic_data)
    
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_table_ai_vs_pkg.to_csv(OUTPUT_TABLE_PATH, index=False)
    
    print("\n--- Evaluation Complete. Generated Table **Comparison of AI-assisted individualized examination strategies and health examination packages for strategy-level usability evaluation** ---")
    print(df_table_ai_vs_pkg.to_string(index=False))
    print(f"\nTable saved to: {OUTPUT_TABLE_PATH}")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_strategy_usability()