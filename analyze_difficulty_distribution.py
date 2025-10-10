#!/usr/bin/env python3
"""
Analysis script for Cambridge MCQ difficulty distributions
Analyzes difficulty, discrimination, and facility metrics from train and test datasets
Creates violin plots and statistical summaries
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_json_data(file_path):
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        return None

def extract_metrics(data, dataset_name):
    """Extract difficulty, discrimination, and facility metrics from data"""
    metrics = []
    
    for item in data:
        if 'difficulty' in item and 'discrimination' in item and 'facility' in item:
            metrics.append({
                'dataset': dataset_name,
                'difficulty': item['difficulty'],
                'discrimination': item['discrimination'],
                'facility': item['facility']
            })
    
    return pd.DataFrame(metrics)

def calculate_statistics(df, metric_name):
    """Calculate comprehensive statistics for a metric"""
    stats_dict = {
        'count': len(df),
        'mean': df[metric_name].mean(),
        'median': df[metric_name].median(),
        'std': df[metric_name].std(),
        'min': df[metric_name].min(),
        'max': df[metric_name].max(),
        'q25': df[metric_name].quantile(0.25),
        'q75': df[metric_name].quantile(0.75),
        'skewness': stats.skew(df[metric_name]),
        'kurtosis': stats.kurtosis(df[metric_name])
    }
    return stats_dict

def print_statistics(df, metric_name, dataset_name):
    """Print formatted statistics"""
    stats = calculate_statistics(df, metric_name)
    
    print(f"\n{'='*60}")
    print(f"{metric_name.upper()} Statistics for {dataset_name}")
    print(f"{'='*60}")
    print(f"Count:           {stats['count']:>8.0f}")
    print(f"Mean:            {stats['mean']:>8.2f}")
    print(f"Median:          {stats['median']:>8.2f}")
    print(f"Std Dev:         {stats['std']:>8.2f}")
    print(f"Min:             {stats['min']:>8.2f}")
    print(f"Max:             {stats['max']:>8.2f}")
    print(f"25th Percentile: {stats['q25']:>8.2f}")
    print(f"75th Percentile: {stats['q75']:>8.2f}")
    print(f"Skewness:        {stats['skewness']:>8.2f}")
    print(f"Kurtosis:        {stats['kurtosis']:>8.2f}")

def create_violin_plots(df):
    """Create violin plots for all metrics"""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribution of Metrics: Train vs Test Datasets', fontsize=16, fontweight='bold')
    
    metrics = ['difficulty', 'discrimination', 'facility']
    metric_titles = ['Difficulty', 'Discrimination', 'Facility']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Create violin plot
        sns.violinplot(data=df, x='dataset', y=metric, ax=axes[i], inner='box')
        
        # Customize the plot
        axes[i].set_title(f'{title} Distribution', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Dataset', fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add mean markers
        for j, dataset in enumerate(['train', 'test']):
            subset = df[df['dataset'] == dataset]
            if not subset.empty:
                mean_val = subset[metric].mean()
                axes[i].scatter(j, mean_val, color='red', s=100, marker='D', 
                              zorder=5, label='Mean' if j == 0 else "")
        
        # Add statistics text
        train_data = df[df['dataset'] == 'train'][metric]
        test_data = df[df['dataset'] == 'test'][metric]
        
        if not train_data.empty and not test_data.empty:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(train_data, test_data)
            
            stats_text = f'Train: μ={train_data.mean():.2f}, σ={train_data.std():.2f}\n'
            stats_text += f'Test: μ={test_data.mean():.2f}, σ={test_data.std():.2f}\n'
            stats_text += f't-test: p={p_value:.4f}'
            
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
    
    # Add legend for mean markers
    axes[0].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/nfshomes/minglii/scratch/EduDD/difficulty_distribution_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_distribution_plot(df):
    """Create a combined distribution plot showing all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Difficulty distribution
    sns.histplot(data=df, x='difficulty', hue='dataset', bins=30, alpha=0.7, ax=axes[0,0])
    axes[0,0].set_title('Difficulty Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Difficulty')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Discrimination distribution
    sns.histplot(data=df, x='discrimination', hue='dataset', bins=30, alpha=0.7, ax=axes[0,1])
    axes[0,1].set_title('Discrimination Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Discrimination')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Facility distribution
    sns.histplot(data=df, x='facility', hue='dataset', bins=30, alpha=0.7, ax=axes[1,0])
    axes[1,0].set_title('Facility Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Facility')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    correlation_data = df[['difficulty', 'discrimination', 'facility']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1,1])
    axes[1,1].set_title('Correlation Matrix', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/nfshomes/minglii/scratch/EduDD/comprehensive_distribution_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def perform_statistical_tests(df):
    """Perform statistical tests between train and test datasets"""
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS: Train vs Test")
    print(f"{'='*60}")
    
    metrics = ['difficulty', 'discrimination', 'facility']
    
    for metric in metrics:
        train_data = df[df['dataset'] == 'train'][metric]
        test_data = df[df['dataset'] == 'test'][metric]
        
        if not train_data.empty and not test_data.empty:
            # T-test
            t_stat, p_value = stats.ttest_ind(train_data, test_data)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(train_data, test_data, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = stats.ks_2samp(train_data, test_data)
            
            print(f"\n{metric.upper()}:")
            print(f"  T-test:           t={t_stat:.4f}, p={p_value:.6f}")
            print(f"  Mann-Whitney U:   U={u_stat:.4f}, p={u_p_value:.6f}")
            print(f"  Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p_value:.6f}")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(train_data) - 1) * train_data.var() + 
                                (len(test_data) - 1) * test_data.var()) / 
                               (len(train_data) + len(test_data) - 2))
            cohens_d = (train_data.mean() - test_data.mean()) / pooled_std
            print(f"  Cohen's d:        {cohens_d:.4f}")

def main():
    """Main analysis function"""
    print("Cambridge MCQ Difficulty Distribution Analysis")
    print("=" * 60)
    
    # File paths
    test_file = '/nfshomes/minglii/scratch/EduDD/split_pub/Cambridge_mcq_test_pub_Processed.json'
    train_file = '/nfshomes/minglii/scratch/EduDD/split_pub/Cambridge_mcq_train_pub_Processed.json'
    
    # Load data
    test_data = load_json_data(test_file)
    train_data = load_json_data(train_file)
    
    if test_data is None or train_data is None:
        print("Error: Could not load one or both datasets")
        return
    
    # Extract metrics
    test_df = extract_metrics(test_data, 'test')
    train_df = extract_metrics(train_data, 'train')
    
    # Combine datasets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\nDataset Summary:")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Total samples: {len(combined_df)}")
    
    # Print statistics for each metric and dataset
    metrics = ['difficulty', 'discrimination', 'facility']
    
    for metric in metrics:
        print_statistics(train_df, metric, 'TRAIN')
        print_statistics(test_df, metric, 'TEST')
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS...")
    print(f"{'='*60}")
    
    create_violin_plots(combined_df)
    create_combined_distribution_plot(combined_df)
    
    # Perform statistical tests
    perform_statistical_tests(combined_df)
    
    # Save summary statistics to CSV
    summary_stats = []
    for dataset in ['train', 'test']:
        subset = combined_df[combined_df['dataset'] == dataset]
        for metric in metrics:
            stats_dict = calculate_statistics(subset, metric)
            stats_dict['dataset'] = dataset
            stats_dict['metric'] = metric
            summary_stats.append(stats_dict)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('/nfshomes/minglii/scratch/EduDD/difficulty_statistics_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("Generated files:")
    print("  - difficulty_distribution_analysis.png")
    print("  - comprehensive_distribution_analysis.png") 
    print("  - difficulty_statistics_summary.csv")
    
    return combined_df

if __name__ == "__main__":
    # Set up matplotlib backend for headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    # Run the analysis
    df = main()
