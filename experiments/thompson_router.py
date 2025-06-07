"""
Thompson Sampling Router for Large Language Model Selection

This module implements a Thompson Sampling-based router that optimally balances
exploration and exploitation for LLM selection under uncertainty.

Mathematical Foundation:
=======================

Thompson Sampling Framework:
1. Prior beliefs: π_i ~ Beta(α_i, β_i) for each model i
2. Posterior updates: α_i ← α_i + successes, β_i ← β_i + failures  
3. Model selection: i* = argmax_i π_i where π_i ~ Beta(α_i, β_i)

Utility Function:
The router maximizes expected utility U_i(x_j) for query j and model i:
    U_i(x_j) = E[Quality_i] - λ × E[Cost_i(x_j)] - γ × Uncertainty_i(x_j)

Where:
- E[Quality_i] = α_i / (α_i + β_i): expected performance from Beta posterior
- E[Cost_i(x_j)] = predicted_tokens_i × cost_per_token_i: expected monetary cost
- Uncertainty_i(x_j): prediction uncertainty from Bayesian model
- λ: cost sensitivity parameter (default: 0.3)  
- γ: risk tolerance parameter (default: 0.25)

Exploration Strategy:
- Adaptive exploration rate: ε_t = max(0.05, ε_0 × decay_rate^t)
- Upper Confidence Bound (UCB) exploration: π_i + c√(ln t / n_i)
- Uncertainty-guided exploration: prefers models with high prediction uncertainty

Cold Start Mechanism:
- Family-informed priors: α_new = performance_family × confidence
- Rapid convergence: leverages family relationships for faster learning
- Progressive refinement: balances family knowledge with model-specific evidence

Key Features:
============
- Multi-objective Optimization: balances quality, cost, and uncertainty
- Adaptive Learning: continuously updates model performance beliefs
- Robust Exploration: prevents premature convergence to suboptimal models
- Economic Awareness: explicitly considers deployment costs

Classes:
========
- ThompsonSamplingRouter: Main routing algorithm implementation
- ColdStartExperiment: Experimental framework for new model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ThompsonSamplingRouter:
    """Thompson Sampling-based LLM router"""
    
    def __init__(self, models: List[str], token_predictor, cost_weight: float = 0.01):
        self.models = models
        self.token_predictor = token_predictor
        self.cost_weight = cost_weight
        
        # Beta distribution parameters (success, failure counts)
        self.alpha = {model: 1.0 for model in models}
        self.beta = {model: 1.0 for model in models}
        
        # Model cost information
        self.model_costs = {
            'gpt-4-1106-preview': 0.03,
            'gpt-4-0613': 0.03,
            'claude-2.1': 0.024,
            'claude-2.0': 0.024,
            'claude-instant-1': 0.00163,
            'gemini-pro': 0.0005,
            'llama-2-70b-chat': 0.00275,
            'llama-2-13b-chat': 0.0013,
            'mixtral-8x7b-instruct': 0.00024,
            'gpt-3.5-turbo-0613': 0.002
        }
        
        # Routing history
        self.routing_history = []
        self.performance_history = []
        self.exploration_rate = 0.1  # Exploration rate
        
    def set_informative_prior(self, model: str, family_performance: Optional[float] = None, 
                             confidence: float = 0.5):
        """Set informative prior distribution (for cold start)"""
        
        if family_performance is not None:
            # Transform beta distribution parameters (Method of Moments)
            mean_perf = max(0.01, min(0.99, family_performance))
            variance = confidence * (1 - confidence)
            
            if 0 < mean_perf < 1 and variance > 0:
                # Calculate α, β from beta distribution's mean and variance
                common_factor = mean_perf * (1 - mean_perf) / variance - 1
                alpha = mean_perf * common_factor
                beta = (1 - mean_perf) * common_factor
                
                self.alpha[model] = max(1.0, alpha)
                self.beta[model] = max(1.0, beta)
                
                print(f"🎯 Set prior distribution for {model}: α={self.alpha[model]:.2f}, β={self.beta[model]:.2f}")
    
    def select_model(self, query_features: np.ndarray, model_predictions: Dict[str, float], 
                    model_uncertainties: Dict[str, float], available_models: Optional[List[str]] = None, 
                    risk_tolerance: float = 0.25) -> str:
        """Select optimal model for query (utilizing Bayesian prediction results)"""
        
        if available_models is None:
            available_models = self.models
            
        utilities = {}
        predictions = {}
        
        # Decide exploration vs exploitation
        should_explore = np.random.random() < self.exploration_rate
        
        for model in available_models:
            if model not in model_predictions:
                continue
                
            # Sample performance (from beta distribution)
            performance_sample = np.random.beta(self.alpha[model], self.beta[model])
            
            # Use Bayesian prediction results
            expected_tokens = model_predictions[model]
            uncertainty = model_uncertainties[model]
            
            # Calculate expected cost
            cost_per_1k = self.model_costs.get(model, 0.01)
            expected_cost = (expected_tokens / 1000) * cost_per_1k
            
            # Apply uncertainty penalty
            uncertainty_penalty = risk_tolerance * uncertainty / 100
            
            if should_explore:
                # Exploration: prefer models with high uncertainty
                exploration_bonus = uncertainty / 100
                utility = performance_sample + exploration_bonus - self.cost_weight * expected_cost
            else:
                # Exploitation: performance first, uncertainty penalty
                utility = performance_sample - self.cost_weight * expected_cost - uncertainty_penalty
            
            utilities[model] = utility
            predictions[model] = {
                'performance_sample': performance_sample,
                'expected_cost': expected_cost,
                'uncertainty': uncertainty,
                'tokens': expected_tokens,
                'exploration_mode': should_explore
            }
        
        # Select model with highest utility
        selected_model = max(utilities, key=utilities.get)
        
        # Save history
        self.routing_history.append({
            'selected_model': selected_model,
            'utilities': utilities.copy(),
            'predictions': predictions.copy(),
            'timestamp': pd.Timestamp.now(),
            'exploration_mode': should_explore
        })
        
        return selected_model
    
    def update(self, model: str, actual_performance: float, actual_cost: Optional[float] = None):
        """Update beta distribution with observed results"""
        
        # Performance criterion: success if above threshold
        performance_threshold = 0.5
        
        if actual_performance > performance_threshold:
            self.alpha[model] += 1
        else:
            self.beta[model] += 1
        
        # Save performance history
        self.performance_history.append({
            'model': model,
            'performance': actual_performance,
            'cost': actual_cost,
            'timestamp': pd.Timestamp.now()
        })
        
        # Adaptive exploration rate adjustment (decrease exploration as performance stabilizes)
        if len(self.performance_history) > 100:
            recent_variance = np.var([h['performance'] for h in self.performance_history[-50:]])
            if recent_variance < 0.01:  # Performance stabilized
                self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
    
    def get_model_confidence(self, model: str) -> Dict:
        """Calculate model confidence"""
        alpha = self.alpha[model]
        beta = self.beta[model]
        
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = np.sqrt(variance)
        
        # 95% confidence interval (beta distribution)
        ci_lower = stats.beta.ppf(0.025, alpha, beta)
        ci_upper = stats.beta.ppf(0.975, alpha, beta)
        
        return {
            'mean_performance': mean,
            'variance': variance,
            'std': std,
            'confidence_interval': [ci_lower, ci_upper],
            'samples': alpha + beta - 2,  # Exclude initial values of 2
            'alpha': alpha,
            'beta': beta
        }
    
    def get_routing_stats(self) -> Dict:
        """Return routing statistics"""
        if not self.routing_history:
            return {}
        
        # Model selection distribution
        model_counts = {}
        exploration_counts = 0
        
        for entry in self.routing_history:
            model = entry['selected_model']
            model_counts[model] = model_counts.get(model, 0) + 1
            if entry.get('exploration_mode', False):
                exploration_counts += 1
        
        total_decisions = len(self.routing_history)
        
        # Performance statistics
        performance_stats = {}
        if self.performance_history:
            recent_performances = [h['performance'] for h in self.performance_history[-100:]]
            performance_stats = {
                'mean_performance': np.mean(recent_performances),
                'std_performance': np.std(recent_performances),
                'recent_trend': self._calculate_trend(recent_performances)
            }
        
        return {
            'total_decisions': total_decisions,
            'model_distribution': {
                model: count/total_decisions 
                for model, count in model_counts.items()
            },
            'exploration_ratio': exploration_counts / total_decisions,
            'performance_stats': performance_stats
        }
    
    def _calculate_trend(self, values: List[float], window: int = 20) -> str:
        """Calculate recent performance trend"""
        if len(values) < window:
            return "insufficient_data"
        
        recent = values[-window:]
        earlier = values[-2*window:-window] if len(values) >= 2*window else values[:-window]
        
        if np.mean(recent) > np.mean(earlier) + 0.05:
            return "improving"
        elif np.mean(recent) < np.mean(earlier) - 0.05:
            return "declining"
        else:
            return "stable"

    def update_rewards(self, model: str, actual_performance: float, query_data: Optional[pd.Series] = None):
        """Update beta distribution with observed results (main_experiment.py compatibility)"""
        self.update(model, actual_performance)
    
    def get_selection_stats(self) -> Dict[str, int]:
        """Return model selection statistics"""
        if not self.routing_history:
            return {model: 0 for model in self.models}
        
        model_counts = {}
        for entry in self.routing_history:
            model = entry['selected_model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Initialize stats for all models to 0
        stats = {model: 0 for model in self.models}
        stats.update(model_counts)
        
        return stats
    
    def get_model_preferences(self) -> Dict[str, float]:
        """Return model preferences (average performance)"""
        preferences = {}
        for model in self.models:
            confidence = self.get_model_confidence(model)
            preferences[model] = confidence['mean_performance']
        
        return preferences
    
    def get_exploration_rate(self) -> float:
        """Return current exploration rate"""
        return self.exploration_rate
    
    def check_convergence(self, window: int = 50, threshold: float = 0.05) -> bool:
        """Check convergence status"""
        if len(self.performance_history) < window:
            return False
        
        recent_performances = [h['performance'] for h in self.performance_history[-window:]]
        performance_variance = np.var(recent_performances)
        
        return performance_variance < threshold

    def add_new_model(self, model: str, alpha: float = 1.0, beta: float = 1.0):
        """Add new model"""
        if model not in self.models:
            self.models.append(model)
        
        self.alpha[model] = alpha
        self.beta[model] = beta
        
        print(f"➕ Add new model: {model} (α={alpha}, β={beta})")

class ColdStartExperiment:
    """Cold start scenario experiment"""
    
    def __init__(self, predictor, router_class):
        self.predictor = predictor
        self.router_class = router_class
        self.results = {}
    
    def simulate_cold_start(self, new_model: str, family: str, test_data: pd.DataFrame, 
                          n_iterations: int = 500) -> Dict:
        """Cold start simulation"""
        
        print(f"\n❄️ Cold start simulation: {new_model}")
        
        # List of models excluding the new one
        available_models = [m for m in self.predictor.get_model_info()['trained_models'] 
                           if m != new_model]
        available_models.append(new_model)  # Add the new model
        
        # Initialize router
        router = self.router_class(available_models, self.predictor)
        
        # Set family-based prior distribution
        family_priors = self.predictor.family_priors
        if family in family_priors:
            family_performance = (family_priors[family]['mean'] - 100) / 200  # Normalize to 0-1
            router.set_informative_prior(
                new_model, 
                family_performance, 
                confidence=0.3
            )
        
        # Save simulation results
        results = {
            'iterations': [],
            'selected_models': [],
            'regrets': [],
            'cumulative_regret': [],
            'convergence_metrics': [],
            'new_model_selections': 0,
            'performance_history': []
        }
        
        # Sample from test data
        test_sample = test_data.sample(min(n_iterations, len(test_data)), random_state=42)
        
        cumulative_regret = 0
        convergence_window = []
        
        for i, (idx, row) in enumerate(test_sample.iterrows()):
            # Extract query features
            query_features = self.predictor._extract_query_features(row)
            
            # Model selection
            try:
                # Perform token prediction
                predicted_tokens = {}
                model_uncertainties = {}
                
                for model in available_models:
                    if model == new_model:
                        # New model: use family average
                        if family in family_priors:
                            pred_result = {
                                'mean': [family_priors[family]['mean']],
                                'std': [family_priors[family]['std']]
                            }
                        else:
                            pred_result = {'mean': [150.0], 'std': [50.0]}
                    else:
                        pred_result = self.predictor.predict_with_uncertainty(query_features, model)
                    
                    predicted_tokens[model] = pred_result['mean'][0]
                    model_uncertainties[model] = pred_result['std'][0]
                
                selected_model = router.select_model(
                    query_features, predicted_tokens, model_uncertainties, available_models, risk_tolerance=0.25
                )
                utility = 0  # placeholder
                prediction = {}  # placeholder
            except Exception as e:
                print(f"⚠️ Routing error: {e}")
                selected_model = np.random.choice(available_models)
                utility = 0
                prediction = {}
            
            # Actual optimal model decision (ground truth)
            true_optimal = self._get_optimal_model(row, available_models)
            
            # Calculate regret
            regret = 1.0 if selected_model != true_optimal else 0.0
            cumulative_regret += regret
            
            # Actual performance simulation
            actual_performance = self._simulate_actual_performance(selected_model, row)
            
            # Router update
            router.update(selected_model, actual_performance)
            
            # Save results
            results['iterations'].append(i + 1)
            results['selected_models'].append(selected_model)
            results['regrets'].append(regret)
            results['cumulative_regret'].append(cumulative_regret)
            results['performance_history'].append(actual_performance)
            
            if selected_model == new_model:
                results['new_model_selections'] += 1
            
            # Convergence metric (50 query window)
            convergence_window.append(regret)
            if len(convergence_window) > 50:
                convergence_window.pop(0)
            
            if len(convergence_window) == 50:
                convergence_score = 1.0 - np.mean(convergence_window)
                results['convergence_metrics'].append(convergence_score)
        
        # Final statistics
        final_stats = {
            'total_iterations': len(results['iterations']),
            'final_cumulative_regret': results['cumulative_regret'][-1],
            'average_regret': results['cumulative_regret'][-1] / len(results['iterations']),
            'new_model_usage_rate': results['new_model_selections'] / len(results['iterations']),
            'convergence_time': self._find_convergence_time(results['regrets']),
            'final_convergence_score': results['convergence_metrics'][-1] if results['convergence_metrics'] else 0,
            'router_stats': router.get_routing_stats(),
            'performance_improvement': self._calculate_performance_improvement(results['performance_history'])
        }
        
        results['final_stats'] = final_stats
        return results
    
    def _get_optimal_model(self, row: pd.Series, available_models: List[str]) -> str:
        """Actual optimal model decision (quality and cost consideration)"""
        model_scores = {}
        
        # Use quality score if available
        if hasattr(row, 'quality_a') and hasattr(row, 'quality_b'):
            model_qualities = {
                row['model_a']: getattr(row, 'quality_a', 1000),
                row['model_b']: getattr(row, 'quality_b', 1000)
            }
        else:
            # Default quality score
            model_qualities = {
                'gpt-4-1106-preview': 1250,
                'gpt-4-0613': 1200,
                'claude-2.1': 1180,
                'claude-2.0': 1150,
                'gemini-pro': 1140,
                'mixtral-8x7b-instruct': 1120,
                'llama-2-70b-chat': 1080,
                'gpt-3.5-turbo-0613': 1050,
                'llama-2-13b-chat': 1000,
                'claude-instant-1': 970
            }
        
        for model in available_models:
            if model in model_qualities:
                base_score = model_qualities[model]
                
                # Complexity bonus
                complexity_bonus = getattr(row, 'query_complexity', 0.5) * 50
                
                # Cost penalty
                cost_penalty = self.predictor.model_costs.get(model, 0.01) * 1000
                
                final_score = base_score + complexity_bonus - cost_penalty
                model_scores[model] = final_score
        
        return max(model_scores, key=model_scores.get) if model_scores else available_models[0]
    
    def _simulate_actual_performance(self, model: str, row: pd.Series) -> float:
        """Actual performance simulation"""
        # Performance calculation based on quality score
        if hasattr(row, 'quality_a') and row['model_a'] == model:
            base_performance = (getattr(row, 'quality_a', 1000) - 950) / 300
        elif hasattr(row, 'quality_b') and row['model_b'] == model:
            base_performance = (getattr(row, 'quality_b', 1000) - 950) / 300
        else:
            # Default performance mapping
            performance_map = {
                'gpt-4-1106-preview': 0.85,
                'gpt-4-0613': 0.80,
                'claude-2.1': 0.78,
                'claude-2.0': 0.75,
                'gemini-pro': 0.73,
                'mixtral-8x7b-instruct': 0.70,
                'llama-2-70b-chat': 0.65,
                'gpt-3.5-turbo-0613': 0.60,
                'llama-2-13b-chat': 0.55,
                'claude-instant-1': 0.50
            }
            base_performance = performance_map.get(model, 0.5)
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        return max(0.1, min(0.9, base_performance + noise))
    
    def _find_convergence_time(self, regrets: List[float], threshold: float = 0.2, 
                              window: int = 50) -> int:
        """Find convergence time"""
        if len(regrets) < window:
            return len(regrets)
        
        for i in range(window, len(regrets)):
            window_regrets = regrets[i-window:i]
            if np.mean(window_regrets) <= threshold:
                return i - window
        
        return len(regrets)
    
    def _calculate_performance_improvement(self, performances: List[float], 
                                         window: int = 50) -> float:
        """Calculate performance improvement rate"""
        if len(performances) < window:
            return 0.0
        
        initial_perf = np.mean(performances[:window])
        final_perf = np.mean(performances[-window:])
        
        return (final_perf - initial_perf) / initial_perf if initial_perf > 0 else 0.0 