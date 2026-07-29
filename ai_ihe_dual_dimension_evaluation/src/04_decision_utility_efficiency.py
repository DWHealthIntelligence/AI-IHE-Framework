"""
Step 4 - Evaluate Decision-making Utility (Efficiency), analyzes the time spent per case by healthcare providers.
Result for Table "Efficiency of examination strategy development under manual and AI-assisted conditions"

@author: DanWu and Agreewithu (Ruixin Dai)
"""


import pandas as pd
from pathlib import Path
from src.utils import (
    calculate_mean_and_ci,
    format_metric_ci,
    perform_paired_test,
    format_p_value
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROVIDER_TIME_DATA_PATH = BASE_DIR / "data" / "raw" / "provider_time.csv"
OUTPUT_TABLE_PATH = BASE_DIR / "results" / "tables" / "table_decision_utility_efficiency.csv"


def evaluate_decision_efficiency():
    """
    Main pipeline to evaluate Decision Utility Efficiency.
    """
    print("--- Step 4: Decision-making Utility (Efficiency) ---")
        
    df_time = pd.read_csv(PROVIDER_TIME_DATA_PATH)
    
    table_data = []
    
    # Helper function to compute row statistics
    def compute_efficiency_row(role_label, df_subset):
        # Average the time spent by the specific role's providers for each case.
        manual_times = df_subset[df_subset['condition'] == 'Manual'].groupby('case_id')['time_minutes'].mean()
        ai_times = df_subset[df_subset['condition'] == 'AI-assisted'].groupby('case_id')['time_minutes'].mean()
        
        # Align by case_id for paired testing
        aligned = pd.merge(manual_times, ai_times, on='case_id', suffixes=('_m', '_a')).dropna()
        
        stat_m = calculate_mean_and_ci(aligned['time_minutes_m'])
        stat_a = calculate_mean_and_ci(aligned['time_minutes_a'])
        
        # Difference = AI - Manual
        diff_array = aligned['time_minutes_a'] - aligned['time_minutes_m']
        stat_diff = calculate_mean_and_ci(diff_array)
        
        # Paired test
        p_val, test_name = perform_paired_test(aligned['time_minutes_a'], aligned['time_minutes_m'])

        print(f"Decision Utility Efficiency: Time Diff, Test Type: {test_name}")
        
        return {
            'Healthcare Provider Role': role_label,
            'Manual': format_metric_ci(stat_m['mean'], stat_m['lower_ci'], stat_m['upper_ci']),
            'AI-assisted': format_metric_ci(stat_a['mean'], stat_a['lower_ci'], stat_a['upper_ci']),
            'Difference': format_metric_ci(stat_diff['mean'], stat_diff['lower_ci'], stat_diff['upper_ci']),
            'P value': format_p_value(p_val)
        }

    # 1. Role-specific analysis
    role_mapping = {'N': 'N (1-3)', 'P': 'P (1-3)', 'S': 'S (1-5)'}
    for role_code in ['N', 'P', 'S']:
        sub_df = df_time[df_time['role'] == role_code]
        row = compute_efficiency_row(role_mapping[role_code], sub_df)
        table_data.append(row)
        
    # 2. Overall analysis (All roles)
    row_all = compute_efficiency_row('All', df_time)
    table_data.append(row_all)
    
    # 3. Create DataFrame and Output
    df_table = pd.DataFrame(table_data)
    
    # Prepend the multi-index like header row for clarity
    header_row = pd.DataFrame([{
        'Healthcare Provider Role': '',
        'Manual': 'Time, mean (95% CI), min',
        'AI-assisted': '',
        'Difference': '',
        'P value': ''
    }])
    
    df_final = pd.concat([header_row, df_table], ignore_index=True)
    
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_TABLE_PATH, index=False)
    
    print("\n--- Evaluation Complete. Generated Table **Efficiency of examination strategy development under manual and AI-assisted conditions** ---")
    print(df_table.to_string(index=False))
    print(f"\nTable saved to: {OUTPUT_TABLE_PATH}")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_decision_efficiency()