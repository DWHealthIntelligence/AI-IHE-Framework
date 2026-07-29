"""
Step 1 - Process raw expert panel reviews to establish the expert-defined reference standard for cases.

@author: DanWu and Agreewithu (Ruixin Dai)
"""


import pandas as pd
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent # Project dir
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "raw_expert_reviews.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "gold_standard.csv"


def establish_ref_standard():
    """
    Filter consensus results and build the reference standard.
    """
    print("--- Step 1: Establishing Expert-Defined Reference Standard ---")
        
    # Load raw expert evaluations
    print(f"Loading raw expert evaluations from: {RAW_DATA_PATH}")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    total_initial_items = len(df_raw)
    total_cases = df_raw['case_id'].nunique()
    print(f"Loaded {total_initial_items} reviewed items across {total_cases} cases.")
    
    # Apply consensus filtering: Retain only items approved as 'R' (Recommended) by the expert panel
    df_gold = df_raw[df_raw['consensus_decision'] == 'R'].copy()
    
    # Deduplicate in case of any accidental redundant additions per case
    df_gold = df_gold.drop_duplicates(subset=['case_id', 'item_id'])
    
    # Sort for presentation
    df_gold = df_gold.sort_values(by=['case_id', 'item_category', 'item_id']).reset_index(drop=True)
    
    # Save processed reference standard
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_gold.to_csv(OUTPUT_PATH, index=False)
    print(f"Reference Standard at: {OUTPUT_PATH}")
    
    # Print summary statistics verifying alignment with manuscript
    print("\n--- Summary of Reference Standard ---")
    print(f"Total valid evaluation cases: {df_gold['case_id'].nunique()}")
    print(f"Total reference examination items established: {len(df_gold)}")
    print(f"Average reference items per case: {len(df_gold) / df_gold['case_id'].nunique():.2f}")
    
    print("\nBreakdown by Examination Item Category:")
    category_summary = df_gold['item_category'].value_counts()
    for cat, count in category_summary.items():
        print(f"  - {cat}: {count} items ({count/len(df_gold)*100:.1f}%)")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    establish_ref_standard()