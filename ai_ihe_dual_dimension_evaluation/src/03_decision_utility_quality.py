"""
Step 3 - Evaluate Decision-making Utility (Quality)
Result for Table "Comparison of manual and AI-assisted examination strategy formulation across examination item categories and healthcare provider roles"
Figure "Confusion matrices illustrating examination item selection under manual and AI-assisted strategy formulation across examination categories and healthcare provider roles"
Figure "Comparison of healthcare provider–specific precision and recall under manual and AI-assisted conditions"

@author: DanWu and Agreewithu (Ruixin Dai)
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
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
PROVIDER_DATA_PATH = BASE_DIR / "data" / "raw" / "provider_decisions.csv"
OUTPUT_TABLE_PATH = BASE_DIR / "results" / "tables" / "table_decision_utility_quality.csv"
FIG_CONFUSION_MAT_PATH = BASE_DIR / "results" / "figures" / "confusion_matrix.png"
FIG_IDIVIDUAL_MET_PATH = BASE_DIR / "results" / "figures" / "individual_metrics.png"


def compute_provider_metrics(df_gold, df_decisions):
    """Calculates TP, FP, FN, Precision, Recall at the Case-Provider level."""
    results = []
    
    # Pre-map categories for reference standard items
    item_cat_map = dict(zip(df_gold['item_id'], df_gold['item_category']))
    
    for (case, provider, condition), group in df_decisions.groupby(['case_id', 'provider_id', 'condition']):
        role = group['role'].iloc[0]
        gold_sub = df_gold[df_gold['case_id'] == case]
        
        # Calculate overall
        gold_set = set(gold_sub['item_id'])
        pred_set = set(group['item_id'])
        tp, fp, fn = get_confusion_matrix_elements(pred_set, gold_set)
        p, r = calculate_metrics(tp, fp, fn)
        
        results.append({
            'case_id': case, 'provider_id': provider, 'role': role, 'condition': condition,
            'category': 'All', 'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r
        })
        
        # Calculate by category
        for cat in ['Physique', 'Laboratory', 'Instrument']:
            gold_cat_set = set(gold_sub[gold_sub['item_category'] == cat]['item_id'])
            # Only count predicted items if they belong to this category or if FP, infer category
            pred_cat_set = set([i for i in pred_set if item_cat_map.get(i, random.choice(['Physique', 'Laboratory', 'Instrument'])) == cat])
            
            tp_c, fp_c, fn_c = get_confusion_matrix_elements(pred_cat_set, gold_cat_set)
            p_c, r_c = calculate_metrics(tp_c, fp_c, fn_c)
            
            results.append({
                'case_id': case, 'provider_id': provider, 'role': role, 'condition': condition,
                'category': cat, 'tp': tp_c, 'fp': fp_c, 'fn': fn_c, 'precision': p_c, 'recall': r_c
            })
            
    return pd.DataFrame(results)


def build_decision_utility_quality_analysis_data(df_metrics):
    """
    Builds the matched paired Table 
    **Comparison of manual and AI-assisted examination strategy formulation across examination item categories and healthcare provider roles** 
    (Category and Role sections)
    """
    table_data = []
    
    # Helper to generate rows
    def generate_row(group_col, group_val):
        sub = df_metrics[df_metrics[group_col] == group_val]
        row = {'Variable': group_val}
        
        for metric in ['precision', 'recall']:
            manual = sub[sub['condition'] == 'Manual'].groupby('case_id')[metric].mean()
            ai = sub[sub['condition'] == 'AI-assisted'].groupby('case_id')[metric].mean()
            
            # Align by case_id for paired testing
            aligned = pd.merge(manual, ai, on='case_id', suffixes=('_m', '_a')).dropna()
            
            stat_m = calculate_mean_and_ci(aligned[f'{metric}_m'])
            stat_a = calculate_mean_and_ci(aligned[f'{metric}_a'])
            stat_diff = calculate_mean_and_ci(aligned[f'{metric}_a'] - aligned[f'{metric}_m'])
            p_val, _ = perform_paired_test(aligned[f'{metric}_a'], aligned[f'{metric}_m'])
            
            row[f'Manual_{metric}'] = format_metric_ci(stat_m['mean'], stat_m['lower_ci'], stat_m['upper_ci'])
            row[f'AI_{metric}'] = format_metric_ci(stat_a['mean'], stat_a['lower_ci'], stat_a['upper_ci'])
            row[f'Diff_{metric}'] = format_metric_ci(stat_diff['mean'], stat_diff['lower_ci'], stat_diff['upper_ci'])
            row[f'P_{metric}'] = format_p_value(p_val)
            
        return row

    # Section 1: Item Category
    table_data.append({'Variable': 'Item Category'})
    for cat in ['Physique', 'Laboratory', 'Instrument', 'All']:
        table_data.append(generate_row('category', cat))
        
    # Section 2: Healthcare Provider Role
    table_data.append({'Variable': 'Healthcare Provider Role'})
    # Filter only 'All' category for role aggregation
    role_df = df_metrics[df_metrics['category'] == 'All']
    for role in ['N', 'P', 'S']:
        # Same process but filter by role
        sub = role_df[role_df['role'] == role]
        row = {'Variable': f"{role} (Role)"}
        for metric in ['precision', 'recall']:
            manual = sub[sub['condition'] == 'Manual'].groupby('case_id')[metric].mean()
            ai = sub[sub['condition'] == 'AI-assisted'].groupby('case_id')[metric].mean()
            aligned = pd.merge(manual, ai, on='case_id', suffixes=('_m', '_a')).dropna()
            
            stat_m = calculate_mean_and_ci(aligned[f'{metric}_m'])
            stat_a = calculate_mean_and_ci(aligned[f'{metric}_a'])
            stat_diff = calculate_mean_and_ci(aligned[f'{metric}_a'] - aligned[f'{metric}_m'])
            p_val, _ = perform_paired_test(aligned[f'{metric}_a'], aligned[f'{metric}_m'])
            
            row[f'Manual_{metric}'] = format_metric_ci(stat_m['mean'], stat_m['lower_ci'], stat_m['upper_ci'])
            row[f'AI_{metric}'] = format_metric_ci(stat_a['mean'], stat_a['lower_ci'], stat_a['upper_ci'])
            row[f'Diff_{metric}'] = format_metric_ci(stat_diff['mean'], stat_diff['lower_ci'], stat_diff['upper_ci'])
            row[f'P_{metric}'] = format_p_value(p_val)
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(OUTPUT_TABLE_PATH, index=False)
    return df_table


def plot_figure_confusion_matrices(df_metrics):
    """Generates Heatmaps of TP, FP, FN."""
    print("Generating Figure Confusion Matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    sns.set_theme(style="whitegrid")
    
    def plot_heatmap(ax, df, row_col, row_order, title, cmap):
        # Calculate average TP, FP, FN per case
        agg = df.groupby([row_col, 'condition'])[['tp', 'fp', 'fn']].mean().reset_index()
        agg = agg[agg['condition'] == title.split('-')[1].replace('assisted', 'assisted')]
        
        matrix = agg.set_index(row_col)[['tp', 'fp', 'fn']].reindex(row_order)
        
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap=cmap, cbar=False, ax=ax,
                    annot_kws={"size": 16, "weight": "bold"}, linewidths=1, linecolor='gray')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('')
        ax.set_xticklabels(['TP', 'FP', 'FN'], fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=90, va='center')

    # Data preps
    cat_df = df_metrics[df_metrics['category'] != 'All']
    role_df = df_metrics[df_metrics['category'] == 'All']

    # Categories
    plot_heatmap(axes[0, 0], cat_df, 'category', ['Physique', 'Laboratory', 'Instrument'], 'Category-Manual', "YlGnBu")
    plot_heatmap(axes[0, 1], cat_df, 'category', ['Physique', 'Laboratory', 'Instrument'], 'Category-AI-assisted', "YlGn")
    
    # Roles
    plot_heatmap(axes[1, 0], role_df, 'role', ['N', 'P', 'S'], 'Role-Manual', "YlGnBu")
    plot_heatmap(axes[1, 1], role_df, 'role', ['N', 'P', 'S'], 'Role-AI-assisted', "YlGn")

    plt.tight_layout()
    FIG_CONFUSION_MAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_CONFUSION_MAT_PATH, dpi=300, bbox_inches='tight')
    plt.close()


def plot_figure_individual_scatter(df_metrics):
    """Generates Figure: Precision & Recall ranges per provider."""
    print("Generating Figure Individual Provider Performance...")
    
    df_all = df_metrics[df_metrics['category'] == 'All']
    # Group by Provider and Condition
    agg = df_all.groupby(['provider_id', 'role', 'condition'])[['precision', 'recall']].agg(['mean', 'sem']).reset_index()
    
    # Sort providers N -> P -> S
    role_order = {'N': 1, 'P': 2, 'S': 3}
    agg['sort_key'] = agg['role'].map(role_order)
    agg = agg.sort_values(['sort_key', 'provider_id'], ascending=[True, True])
    unique_providers = agg['provider_id'].unique()
    provider_y_map = {p: i+1 for i, p in enumerate(unique_providers)}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    markers = {'N': 'o', 'P': '^', 'S': 's'}
    colors = {'Manual': '#4B8BBE', 'AI-assisted': '#D02A2A'} # Blue vs Red
    
    def plot_metric(ax, metric_name, title):
        for _, row in agg.iterrows():
            y = provider_y_map[row['provider_id'][0]]
            x = row[metric_name]['mean']
            err = row[metric_name]['sem'] * 1.96 # 95% CI
            role = row['role'][0]
            cond = row['condition'][0]
            
            ax.errorbar(x, y, xerr=err, fmt=markers[role], color=colors[cond], 
                        markersize=10, elinewidth=2, capsize=5, capthick=2)
            
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_yticks(list(provider_y_map.values()))
        # Remove "Provider_" for y labels to match 1-11
        ax.set_yticklabels([str(y) for y in provider_y_map.values()], fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_metric(axes[0], 'precision', 'Precision')
    plot_metric(axes[1], 'recall', 'Recall')
    
    # Legend construction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#4B8BBE', mfc='white', markersize=10, label='N'),
        Line2D([0], [0], marker='^', color='#4B8BBE', mfc='white', markersize=10, label='P'),
        Line2D([0], [0], marker='s', color='#4B8BBE', mfc='white', markersize=10, label='S'),
        Line2D([0], [0], marker='o', color='#D02A2A', mfc='white', markersize=10, label='N'),
        Line2D([0], [0], marker='^', color='#D02A2A', mfc='white', markersize=10, label='P'),
        Line2D([0], [0], marker='s', color='#D02A2A', mfc='white', markersize=10, label='S'),
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=12, ncol=2)
    
    plt.tight_layout()
    FIG_IDIVIDUAL_MET_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_IDIVIDUAL_MET_PATH, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_decision_quality():
    """
    Main pipeline to evaluate Decision Utility Quality.
    """
    print("--- Step 3: Decision-making Utility (Quality) ---")
    
    if not REFERENCE_STANDARD_PATH.exists():
        raise FileNotFoundError(f"Reference standard missing. Run 01_establish_reference_standard.py")
        
    df_gold = pd.read_csv(REFERENCE_STANDARD_PATH)
        
    df_decisions = pd.read_csv(PROVIDER_DATA_PATH)
    
    print("Processing provider metrics ...")
    df_metrics = compute_provider_metrics(df_gold, df_decisions)
    
    # Generate Table "Comparison of manual and AI-assisted examination strategy formulation across examination item categories and healthcare provider roles"
    print("\nAnalyzing decision utility quality...")
    df_utility_quality = build_decision_utility_quality_analysis_data(df_metrics)
    print(df_utility_quality.to_string(index=False))
    print(f"Decision utility quality table saved to: {OUTPUT_TABLE_PATH}")
    
    # Generate Figures
    plot_figure_confusion_matrices(df_metrics)
    print(f"Figure confusion matrix saved to: {FIG_CONFUSION_MAT_PATH}")
    
    plot_figure_individual_scatter(df_metrics)
    print(f"Figure individual metrics saved to: {FIG_IDIVIDUAL_MET_PATH}")
    
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_decision_quality()