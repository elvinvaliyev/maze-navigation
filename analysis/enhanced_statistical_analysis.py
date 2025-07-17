"""
Enhanced statistical analysis module for maze navigation experiments.

This module provides comprehensive statistical analysis including multi-factor ANOVA,
effect size calculations, correlation analysis, and cluster analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analysis for comprehensive hypothesis testing."""
    
    def __init__(self):
        self.results = None
        self.statistical_results = {}
        self.alpha = 0.05  # Significance level
        
    def load_results(self, results_file: str = "results/comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            # — if the CSV has only reward_improvement, make sure learning_improvement exists —
            if 'learning_improvement' not in self.results.columns and 'reward_improvement' in self.results.columns:
                self.results['learning_improvement'] = self.results['reward_improvement']
            print(f"Loaded {len(self.results)} results for enhanced statistical analysis")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def multi_factor_anova(self) -> Dict:
        """Perform multi-factor ANOVA to test interactions between variables."""
        if self.results is None:
            return {}
        
        print("\n=== MULTI-FACTOR ANOVA ANALYSIS ===")
        
        anova_results = {}
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        
        for metric in metrics:
            print(f"\n{metric.upper()} Analysis:")
            
            # Prepare data for multi-factor ANOVA
            # We'll test the effects of agent, maze, reward_config, swap_prob, and step_budget
            
            # Create groups for each factor
            groups = {}
            factors = ['agent', 'maze', 'reward_config', 'swap_prob', 'step_budget']
            
            for factor in factors:
                groups[factor] = [group[metric].values for name, group in self.results.groupby(factor)]
            
            # Perform one-way ANOVA for each factor
            factor_results = {}
            for factor, group_data in groups.items():
                if len(group_data) > 1:
                    f_stat, p_value = f_oneway(*group_data)
                    factor_results[factor] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'effect_size': self._calculate_eta_squared(group_data)
                    }
                    
                    print(f"  {factor}: F={f_stat:.3f}, p={p_value:.3f}, η²={factor_results[factor]['effect_size']:.3f}")
                    if p_value < self.alpha:
                        print(f"    *** SIGNIFICANT EFFECT ***")
            
            anova_results[metric] = factor_results

            # --- Post-hoc Tukey HSD for agent differences (on avg_reward) ---
            if metric == 'avg_reward':
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey = pairwise_tukeyhsd(endog=self.results['avg_reward'],
                                              groups=self.results['agent'],
                                              alpha=0.05)
                    print("\nTukey HSD post-hoc test for agent avg_reward:")
                    print(tukey.summary())
                except Exception as e:
                    print(f"[WARNING] Tukey HSD test failed: {e}")

            # --- Bootstrapped CI for optimal step budget threshold ---
            if metric == 'avg_reward':
                def compute_optimal_budget(df):
                    # Example: step budget with max avg_reward
                    grouped = df.groupby('step_budget')['avg_reward'].mean()
                    return grouped.idxmax() if not grouped.empty else np.nan
                bootstraps = []
                for _ in range(1000):
                    sample = self.results.sample(frac=1, replace=True)
                    bootstraps.append(compute_optimal_budget(sample))
                ci_low, ci_high = np.percentile(bootstraps, [2.5, 97.5])
                print(f"Bootstrapped CI for optimal step budget: {ci_low:.1f} – {ci_high:.1f}")
        
        return anova_results
    
    def _calculate_eta_squared(self, groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        # Calculate total sum of squares
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        # Calculate between-group sum of squares
        ss_between = 0
        for group in groups:
            group_mean = np.mean(group)
            ss_between += len(group) * (group_mean - grand_mean) ** 2
        
        # Calculate eta-squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        return eta_squared
    
    def correlation_analysis(self) -> Dict:
        """Perform comprehensive correlation analysis between variables."""
        if self.results is None:
            return {}
        
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numerical variables for correlation
        numerical_vars = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement', 
                         'avg_collected_rewards', 'step_budget', 'swap_prob']
        
        correlation_matrix = self.results[numerical_vars].corr()
        
        # Calculate significance of correlations
        correlation_significance = {}
        for var1 in numerical_vars:
            for var2 in numerical_vars:
                if var1 != var2:
                    x = self.results[var1]
                    y = self.results[var2]
                    mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 2:
                        correlation_significance[(var1, var2)] = {
                            'correlation': np.nan,
                            'p_value': np.nan,
                            'significant': False
                        }
                        continue
                    correlation, p_value = stats.pearsonr(x[mask], y[mask])
                    correlation_significance[(var1, var2)] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < self.alpha
                    }
        
        # Find strongest correlations
        strong_correlations = []
        for (var1, var2), data in correlation_significance.items():
            if data['significant'] and abs(data['correlation']) > 0.3:
                strong_correlations.append({
                    'variables': (var1, var2),
                    'correlation': data['correlation'],
                    'p_value': data['p_value']
                })
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print("Strongest significant correlations:")
        for corr in strong_correlations[:10]:  # Top 10
            print(f"  {corr['variables'][0]} ↔ {corr['variables'][1]}: r={corr['correlation']:.3f}, p={corr['p_value']:.3f}")
        
        return {
            'correlation_matrix': correlation_matrix,
            'correlation_significance': correlation_significance,
            'strong_correlations': strong_correlations
        }
    
    def effect_size_analysis(self) -> Dict:
        """Calculate comprehensive effect sizes for all comparisons."""
        if self.results is None:
            return {}
        
        print("\n=== EFFECT SIZE ANALYSIS ===")
        
        effect_sizes = {}
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        
        for metric in metrics:
            metric_effects = {}
            
            # Agent comparisons
            agent_data = {}
            for agent in self.results['agent'].unique():
                agent_data[agent] = self.results[self.results['agent'] == agent][metric].values
            
            # Calculate Cohen's d for all agent pairs
            agent_effects = []
            agents = list(agent_data.keys())
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    agent1, agent2 = agents[i], agents[j]
                    cohens_d = self._calculate_cohens_d(agent_data[agent1], agent_data[agent2])
                    agent_effects.append({
                        'comparison': f"{agent1} vs {agent2}",
                        'cohens_d': cohens_d,
                        'magnitude': self._interpret_effect_size(cohens_d)
                    })
            
            metric_effects['agent_comparisons'] = agent_effects
            
            # Condition comparisons
            for condition in ['maze', 'reward_config', 'swap_prob', 'step_budget']:
                condition_effects = []
                condition_groups = [group[metric].values for name, group in self.results.groupby(condition)]
                
                if len(condition_groups) > 1:
                    # Calculate eta-squared for condition effect
                    eta_squared = self._calculate_eta_squared(condition_groups)
                    condition_effects.append({
                        'condition': condition,
                        'eta_squared': eta_squared,
                        'magnitude': self._interpret_effect_size(np.sqrt(eta_squared))
                    })
                
                metric_effects[f'{condition}_effect'] = condition_effects
            
            effect_sizes[metric] = metric_effects
        
        # Print summary of largest effects
        print("Largest effect sizes:")
        for metric, effects in effect_sizes.items():
            if 'agent_comparisons' in effects:
                largest_agent_effect = max(effects['agent_comparisons'], key=lambda x: abs(x['cohens_d']))
                print(f"  {metric}: {largest_agent_effect['comparison']} (d={largest_agent_effect['cohens_d']:.3f}, {largest_agent_effect['magnitude']})")
        
        return effect_sizes
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                              (len(group2) - 1) * np.var(group2)) / 
                             (len(group1) + len(group2) - 2))
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return cohens_d
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        else:
            return "large"
    
    def cluster_analysis(self) -> Dict:
        """Perform cluster analysis to identify performance patterns."""
        if self.results is None:
            return {}
        
        print("\n=== CLUSTER ANALYSIS ===")
        
        # Prepare features for clustering
        features = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement', 
                   'avg_collected_rewards']
        
        # Aggregate data by agent and condition to create feature matrix
        cluster_data = self.results.groupby(['agent', 'maze', 'reward_config', 'swap_prob', 'step_budget'])[features].mean()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_data)
        
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=min(3, len(features)))
        features_pca = pca.fit_transform(features_scaled)
        
        # Perform clustering
        n_clusters = min(5, len(cluster_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_pca)
        
        # Analyze clusters
        cluster_analysis = {}
        cluster_data_with_labels = cluster_data.reset_index()
        cluster_data_with_labels['cluster'] = clusters
        
        for i in range(n_clusters):
            cluster_subset = cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_subset),
                'avg_reward': cluster_subset['avg_reward'].mean(),
                'avg_exit_rate': cluster_subset['exit_rate'].mean(),
                'avg_survival_rate': cluster_subset['survival_rate'].mean(),
                'avg_learning': cluster_subset['learning_improvement'].mean(),
                'agent_distribution': cluster_subset['agent'].value_counts().to_dict(),
                'maze_distribution': cluster_subset['maze'].value_counts().to_dict()
            }
        
        print(f"Identified {n_clusters} performance clusters:")
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"  {cluster_name}: {cluster_data['size']} cases")
            print(f"    Avg reward: {cluster_data['avg_reward']:.3f}")
            print(f"    Avg exit rate: {cluster_data['avg_exit_rate']:.3f}")
            print(f"    Avg survival rate: {cluster_data['avg_survival_rate']:.3f}")
        
        return {
            'clusters': cluster_analysis,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'cluster_assignments': dict(zip(range(len(cluster_data)), clusters))
        }
    
    def confidence_interval_analysis(self) -> Dict:
        """Calculate confidence intervals for all key metrics."""
        if self.results is None:
            return {}
        
        print("\n=== CONFIDENCE INTERVAL ANALYSIS ===")
        
        ci_results = {}
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        
        for metric in metrics:
            metric_cis = {}
            
            # Overall confidence intervals
            data = self.results[metric].values
            mean_val = np.mean(data)
            std_err = stats.sem(data)
            ci_lower, ci_upper = stats.t.interval(0.95, len(data)-1, loc=mean_val, scale=std_err)
            
            metric_cis['overall'] = {
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std_error': std_err
            }
            
            # Confidence intervals by agent
            agent_cis = {}
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent][metric].values
                mean_val = np.mean(agent_data)
                std_err = stats.sem(agent_data)
                ci_lower, ci_upper = stats.t.interval(0.95, len(agent_data)-1, loc=mean_val, scale=std_err)
                
                agent_cis[agent] = {
                    'mean': mean_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'std_error': std_err
                }
            
            metric_cis['by_agent'] = agent_cis
            ci_results[metric] = metric_cis
        
        # Print summary
        print("95% Confidence Intervals:")
        for metric, cis in ci_results.items():
            overall = cis['overall']
            print(f"  {metric}: {overall['mean']:.3f} [{overall['ci_lower']:.3f}, {overall['ci_upper']:.3f}]")
        
        return ci_results
    
    def create_enhanced_statistical_visualizations(self):
        """Create enhanced statistical visualizations."""
        if self.results is None:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive statistical plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        ax1 = axes[0, 0]
        numerical_vars = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement', 
                         'avg_collected_rewards', 'step_budget', 'swap_prob']
        correlation_matrix = self.results[numerical_vars].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Correlation Matrix')
        
        # 2. Effect size comparison
        ax2 = axes[0, 1]
        metrics = ['avg_reward', 'exit_rate', 'survival_rate']
        effect_sizes = []
        metric_labels = []
        
        for metric in metrics:
            agent_data = {}
            for agent in self.results['agent'].unique():
                agent_data[agent] = self.results[self.results['agent'] == agent][metric].values
            
            # Calculate average effect size between agents
            agents = list(agent_data.keys())
            metric_effects = []
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    cohens_d = self._calculate_cohens_d(agent_data[agents[i]], agent_data[agents[j]])
                    metric_effects.append(abs(cohens_d))
            
            effect_sizes.append(np.mean(metric_effects))
            metric_labels.append(metric.replace('_', ' ').title())
        
        bars = ax2.bar(metric_labels, effect_sizes, alpha=0.7)
        ax2.set_ylabel('Average Effect Size (|Cohen\'s d|)')
        ax2.set_title('Effect Sizes by Metric')
        
        # Add value labels
        for bar, effect in zip(bars, effect_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{effect:.3f}', ha='center', va='bottom')
        
        # 3. Confidence intervals by agent
        ax3 = axes[0, 2]
        agents = self.results['agent'].unique()
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for agent in agents:
            agent_data = self.results[self.results['agent'] == agent]['avg_reward'].values
            mean_val = np.mean(agent_data)
            std_err = stats.sem(agent_data)
            ci_lower, ci_upper = stats.t.interval(0.95, len(agent_data)-1, loc=mean_val, scale=std_err)
            
            means.append(mean_val)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
        
        y_pos = np.arange(len(agents))
        ax3.errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lowers), 
                                        np.array(ci_uppers) - np.array(means)], 
                    fmt='o', capsize=5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(agents)
        ax3.set_xlabel('Average Reward')
        ax3.set_title('95% Confidence Intervals by Agent')
        ax3.grid(True, alpha=0.3)
        
        # 4. ANOVA results visualization
        ax4 = axes[1, 0]
        factors = ['agent', 'maze', 'reward_config', 'swap_prob', 'step_budget']
        f_stats = []
        p_values = []
        
        for factor in factors:
            groups = [group['avg_reward'].values for name, group in self.results.groupby(factor)]
            if len(groups) > 1:
                f_stat, p_val = f_oneway(*groups)
                f_stats.append(f_stat)
                p_values.append(p_val)
            else:
                f_stats.append(0)
                p_values.append(1)
        
        # Plot -log10(p-values) to show significance
        significance = -np.log10(np.array(p_values))
        bars = ax4.bar(factors, significance, alpha=0.7)
        ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('ANOVA Significance by Factor')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        # Color bars based on significance
        for bar, sig in zip(bars, significance):
            if sig > -np.log10(0.05):
                bar.set_color('green')
            else:
                bar.set_color('gray')
        
        # 5. Cluster analysis visualization
        ax5 = axes[1, 1]
        features = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        cluster_data = self.results.groupby(['agent', 'maze', 'reward_config', 'swap_prob', 'step_budget'])[features].mean()
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_data)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        kmeans = KMeans(n_clusters=min(4, len(cluster_data)), random_state=42)
        clusters = kmeans.fit_predict(features_pca)
        
        scatter = ax5.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax5.set_title('Performance Clusters (PCA)')
        plt.colorbar(scatter, ax=ax5)
        
        # 6. Effect size distribution
        ax6 = axes[1, 2]
        all_effect_sizes = []
        
        for metric in ['avg_reward', 'exit_rate', 'survival_rate']:
            agent_data = {}
            for agent in self.results['agent'].unique():
                agent_data[agent] = self.results[self.results['agent'] == agent][metric].values
            
            agents = list(agent_data.keys())
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    cohens_d = self._calculate_cohens_d(agent_data[agents[i]], agent_data[agents[j]])
                    all_effect_sizes.append(abs(cohens_d))
        
        ax6.hist(all_effect_sizes, bins=20, alpha=0.7, edgecolor='black')
        ax6.axvline(x=0.2, color='red', linestyle='--', label='Small effect')
        ax6.axvline(x=0.5, color='orange', linestyle='--', label='Medium effect')
        ax6.axvline(x=0.8, color='green', linestyle='--', label='Large effect')
        ax6.set_xlabel('|Cohen\'s d|')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Effect Size Distribution')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'enhanced_statistical_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_enhanced_statistical_report(self) -> Dict:
        """Generate a comprehensive enhanced statistical analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE ENHANCED STATISTICAL ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all enhanced statistical analyses
        anova_results = self.multi_factor_anova()
        correlation_analysis = self.correlation_analysis()
        effect_sizes = self.effect_size_analysis()
        cluster_analysis = self.cluster_analysis()
        confidence_intervals = self.confidence_interval_analysis()
        
        # Create visualizations
        self.create_enhanced_statistical_visualizations()
        
        # Generate summary
        print("\n" + "="*60)
        print("ENHANCED STATISTICAL SUMMARY")
        print("="*60)
        
        # Count significant effects
        significant_effects = 0
        total_effects = 0
        
        for metric, factor_results in anova_results.items():
            for factor, result in factor_results.items():
                total_effects += 1
                if result['significant']:
                    significant_effects += 1
        
        print(f"\nSignificant effects: {significant_effects}/{total_effects} ({significant_effects/total_effects*100:.1f}%)")
        
        # Strongest correlations
        if correlation_analysis['strong_correlations']:
            strongest_corr = correlation_analysis['strong_correlations'][0]
            print(f"\nStrongest correlation: {strongest_corr['variables'][0]} ↔ {strongest_corr['variables'][1]} (r={strongest_corr['correlation']:.3f})")
        
        # Largest effect sizes
        largest_effects = []
        for metric, effects in effect_sizes.items():
            if 'agent_comparisons' in effects:
                largest_effect = max(effects['agent_comparisons'], key=lambda x: abs(x['cohens_d']))
                largest_effects.append((metric, largest_effect))
        
        if largest_effects:
            largest_effect = max(largest_effects, key=lambda x: abs(x[1]['cohens_d']))
            print(f"\nLargest effect size: {largest_effect[0]} - {largest_effect[1]['comparison']} (d={largest_effect[1]['cohens_d']:.3f})")
        
        # Cluster summary
        if 'clusters' in cluster_analysis:
            print(f"\nIdentified {len(cluster_analysis['clusters'])} performance clusters")
        
        report = {
            'anova_results': anova_results,
            'correlation_analysis': correlation_analysis,
            'effect_sizes': effect_sizes,
            'cluster_analysis': cluster_analysis,
            'confidence_intervals': confidence_intervals
        }
        
        return report 