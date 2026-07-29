"""
Functions for statistical analysis, performance metrics (Precision, Recall), and hypothesis testing for AI-IHE Dual-Dimension Evaluation.

@author: DanWu and Agreewithu (Ruixin Dai)
"""


import numpy as np
from scipy import stats
from typing import Tuple, List, Set, Dict, Union


def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """
    Calculate Precision and Recall given True Positives, False Positives, and False Negatives.
    
    Formula:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
    
    Returns:
        tuple: (precision, recall) as percentages (0.0 to 100.0). Returns 0.0 if denominator is 0.
    """
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    return precision, recall


def get_confusion_matrix_elements(predicted_items: Set[str], reference_items: Set[str]) -> Tuple[int, int, int]:
    """
    Compare predicted examination items against the experts' gold standard.
    
    Returns:
        tuple: (TP, FP, FN)
    """
    tp = len(predicted_items.intersection(reference_items))
    fp = len(predicted_items - reference_items)
    fn = len(reference_items - predicted_items)
    return tp, fp, fn


def calculate_mean_and_ci(data: Union[List[float], np.ndarray], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate the mean and 95% Confidence Interval (CI) for a sample distribution.
    
    Returns:
        dict: {'mean': float, 'lower_ci': float, 'upper_ci': float}
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    
    if n <= 1:
        return {'mean': mean, 'lower_ci': mean, 'upper_ci': mean}
    
    # Calculate standard error and t-distribution critical value
    se = stats.sem(data)
    t_val = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    
    lower_ci = mean - t_val * se
    upper_ci = mean + t_val * se
    
    return {
        'mean': mean,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }


def format_metric_ci(mean: float, lower_ci: float, upper_ci: float, decimals: int = 1) -> str:
    """
    Format mean and CI.
    """
    fmt = f"{{:.{decimals}f}}"
    mean_str = fmt.format(mean)
    lower_str = fmt.format(lower_ci)
    upper_str = fmt.format(upper_ci)
    return f"{mean_str}({lower_str}-{upper_str})"


def perform_paired_test(group1: Union[List[float], np.ndarray], 
                        group2: Union[List[float], np.ndarray], 
                        alpha: float = 0.05) -> Tuple[float, str]:
    """
    Perform paired hypothesis testing between two matched conditions (e.g., Manual vs AI-assisted).
    
    According to the result of normal distribution test:
    - Paired t-tests were used when paired differences satisfied normality assumptions (Shapiro-Wilk test).
    - Otherwise, Wilcoxon signed-rank tests were applied.
    
    Returns:
        tuple: (p_value, test_name_used)
    """
    diff = np.array(group1) - np.array(group2)
    
    # Check normality of differences using Shapiro-Wilk test
    if np.all(diff == diff[0]) or len(diff) < 3:
        # Default to Wilcoxon if variance is 0 or sample too small for Shapiro
        stat, p_val = stats.wilcoxon(group1, group2, zero_method='zsplit', error='ignore')
        return p_val, "Wilcoxon signed-rank test"
    
    _, shapiro_p = stats.shapiro(diff)
    
    if shapiro_p > alpha:
        # Differences are normally distributed -> Paired t-test
        stat, p_val = stats.ttest_rel(group1, group2)
        test_name = "Paired t-test"
    else:
        # Non-normal distribution -> Wilcoxon signed-rank test
        stat, p_val = stats.wilcoxon(group1, group2)
        test_name = "Wilcoxon signed-rank test"
        
    return p_val, test_name


def format_p_value(p_val: float) -> str:
    """
    Format P-value (e.g., "<.001").
    """
    if p_val < 0.001:
        return "<.001"
    else:
        return f"{p_val:.3f}".lstrip('0')