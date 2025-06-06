import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ThompsonSamplingRouter:
    """Thompson Sampling 기반 LLM 라우터"""
    
    def __init__(self, models: List[str], token_predictor, cost_weight: float = 0.01):
        self.models = models
        self.token_predictor = token_predictor
        self.cost_weight = cost_weight
        
        # 베타 분포 파라미터 (성공, 실패 카운트)
        self.alpha = {model: 1.0 for model in models}
        self.beta = {model: 1.0 for model in models}
        
        # 모델별 비용 정보
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
        
        # 라우팅 히스토리
        self.routing_history = []
        self.performance_history = []
        self.exploration_rate = 0.1  # 탐험 비율
        
    def set_informative_prior(self, model: str, family_performance: Optional[float] = None, 
                             confidence: float = 0.5):
        """정보적 사전 분포 설정 (콜드 스타트용)"""
        
        if family_performance is not None:
            # 베타 분포 파라미터 변환 (Method of Moments)
            mean_perf = max(0.01, min(0.99, family_performance))
            variance = confidence * (1 - confidence)
            
            if 0 < mean_perf < 1 and variance > 0:
                # 베타 분포의 평균과 분산으로부터 α, β 계산
                common_factor = mean_perf * (1 - mean_perf) / variance - 1
                alpha = mean_perf * common_factor
                beta = (1 - mean_perf) * common_factor
                
                self.alpha[model] = max(1.0, alpha)
                self.beta[model] = max(1.0, beta)
                
                print(f"🎯 {model} 사전 분포 설정: α={self.alpha[model]:.2f}, β={self.beta[model]:.2f}")
    
    def select_model(self, query_features: np.ndarray, model_predictions: Dict[str, float], 
                    model_uncertainties: Dict[str, float], available_models: Optional[List[str]] = None, 
                    risk_tolerance: float = 0.25) -> str:
        """쿼리에 대한 최적 모델 선택 (베이지안 예측 결과 활용)"""
        
        if available_models is None:
            available_models = self.models
            
        utilities = {}
        predictions = {}
        
        # 탐험 vs 활용 결정
        should_explore = np.random.random() < self.exploration_rate
        
        for model in available_models:
            if model not in model_predictions:
                continue
                
            # 성능 샘플링 (베타 분포에서)
            performance_sample = np.random.beta(self.alpha[model], self.beta[model])
            
            # 베이지안 예측 결과 사용
            expected_tokens = model_predictions[model]
            uncertainty = model_uncertainties[model]
            
            # 예상 비용 계산
            cost_per_1k = self.model_costs.get(model, 0.01)
            expected_cost = (expected_tokens / 1000) * cost_per_1k
            
            # 불확실성 페널티 적용
            uncertainty_penalty = risk_tolerance * uncertainty / 100
            
            if should_explore:
                # 탐험: 불확실성이 높은 모델 선호
                exploration_bonus = uncertainty / 100
                utility = performance_sample + exploration_bonus - self.cost_weight * expected_cost
            else:
                # 활용: 성능 우선, 불확실성 페널티
                utility = performance_sample - self.cost_weight * expected_cost - uncertainty_penalty
            
            utilities[model] = utility
            predictions[model] = {
                'performance_sample': performance_sample,
                'expected_cost': expected_cost,
                'uncertainty': uncertainty,
                'tokens': expected_tokens,
                'exploration_mode': should_explore
            }
        
        # 최고 효용의 모델 선택
        selected_model = max(utilities, key=utilities.get)
        
        # 히스토리 저장
        self.routing_history.append({
            'selected_model': selected_model,
            'utilities': utilities.copy(),
            'predictions': predictions.copy(),
            'timestamp': pd.Timestamp.now(),
            'exploration_mode': should_explore
        })
        
        return selected_model
    
    def update(self, model: str, actual_performance: float, actual_cost: Optional[float] = None):
        """관찰된 결과로 베타 분포 업데이트"""
        
        # 성과 기준: 임계값 이상이면 성공
        performance_threshold = 0.5
        
        if actual_performance > performance_threshold:
            self.alpha[model] += 1
        else:
            self.beta[model] += 1
        
        # 성능 히스토리 저장
        self.performance_history.append({
            'model': model,
            'performance': actual_performance,
            'cost': actual_cost,
            'timestamp': pd.Timestamp.now()
        })
        
        # 적응적 탐험률 조정 (성능이 안정되면 탐험 감소)
        if len(self.performance_history) > 100:
            recent_variance = np.var([h['performance'] for h in self.performance_history[-50:]])
            if recent_variance < 0.01:  # 성능이 안정됨
                self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
    
    def get_model_confidence(self, model: str) -> Dict:
        """모델별 신뢰도 계산"""
        alpha = self.alpha[model]
        beta = self.beta[model]
        
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = np.sqrt(variance)
        
        # 95% 신뢰구간 (베타 분포)
        ci_lower = stats.beta.ppf(0.025, alpha, beta)
        ci_upper = stats.beta.ppf(0.975, alpha, beta)
        
        return {
            'mean_performance': mean,
            'variance': variance,
            'std': std,
            'confidence_interval': [ci_lower, ci_upper],
            'samples': alpha + beta - 2,  # 초기값 2 제외
            'alpha': alpha,
            'beta': beta
        }
    
    def get_routing_stats(self) -> Dict:
        """라우팅 통계 반환"""
        if not self.routing_history:
            return {}
        
        # 모델 선택 분포
        model_counts = {}
        exploration_counts = 0
        
        for entry in self.routing_history:
            model = entry['selected_model']
            model_counts[model] = model_counts.get(model, 0) + 1
            if entry.get('exploration_mode', False):
                exploration_counts += 1
        
        total_decisions = len(self.routing_history)
        
        # 성능 통계
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
            'exploration_rate': exploration_counts / total_decisions if total_decisions > 0 else 0,
            'confidence_stats': {
                model: self.get_model_confidence(model) 
                for model in self.models
            },
            'performance_stats': performance_stats,
            'current_exploration_rate': self.exploration_rate
        }
    
    def _calculate_trend(self, values: List[float], window: int = 20) -> str:
        """최근 성능 트렌드 계산"""
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
        """관찰된 결과로 베타 분포 업데이트 (main_experiment.py 호환)"""
        self.update(model, actual_performance)
    
    def get_selection_stats(self) -> Dict[str, int]:
        """모델별 선택 통계 반환"""
        if not self.routing_history:
            return {model: 0 for model in self.models}
        
        model_counts = {}
        for entry in self.routing_history:
            model = entry['selected_model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # 모든 모델에 대해 0으로 초기화
        stats = {model: 0 for model in self.models}
        stats.update(model_counts)
        
        return stats
    
    def get_model_preferences(self) -> Dict[str, float]:
        """모델별 선호도 (평균 성능) 반환"""
        preferences = {}
        for model in self.models:
            confidence = self.get_model_confidence(model)
            preferences[model] = confidence['mean_performance']
        
        return preferences
    
    def get_exploration_rate(self) -> float:
        """현재 탐험률 반환"""
        return self.exploration_rate
    
    def check_convergence(self, window: int = 50, threshold: float = 0.05) -> bool:
        """수렴 여부 확인"""
        if len(self.performance_history) < window:
            return False
        
        recent_performances = [h['performance'] for h in self.performance_history[-window:]]
        performance_variance = np.var(recent_performances)
        
        return performance_variance < threshold

    def add_new_model(self, model: str, alpha: float = 1.0, beta: float = 1.0):
        """새로운 모델 추가"""
        if model not in self.models:
            self.models.append(model)
        
        self.alpha[model] = alpha
        self.beta[model] = beta
        
        print(f"➕ 새 모델 추가: {model} (α={alpha}, β={beta})")

class ColdStartExperiment:
    """콜드 스타트 시나리오 실험"""
    
    def __init__(self, predictor, router_class):
        self.predictor = predictor
        self.router_class = router_class
        self.results = {}
    
    def simulate_cold_start(self, new_model: str, family: str, test_data: pd.DataFrame, 
                          n_iterations: int = 500) -> Dict:
        """콜드 스타트 시뮬레이션"""
        
        print(f"\n❄️ 콜드 스타트 시뮬레이션: {new_model}")
        
        # 새 모델 제외한 모델 리스트
        available_models = [m for m in self.predictor.get_model_info()['trained_models'] 
                           if m != new_model]
        available_models.append(new_model)  # 새 모델 추가
        
        # 라우터 초기화
        router = self.router_class(available_models, self.predictor)
        
        # 가족 기반 사전 분포 설정
        family_priors = self.predictor.family_priors
        if family in family_priors:
            family_performance = (family_priors[family]['mean'] - 100) / 200  # 0-1 정규화
            router.set_informative_prior(
                new_model, 
                family_performance, 
                confidence=0.3
            )
        
        # 시뮬레이션 결과 저장
        results = {
            'iterations': [],
            'selected_models': [],
            'regrets': [],
            'cumulative_regret': [],
            'convergence_metrics': [],
            'new_model_selections': 0,
            'performance_history': []
        }
        
        # 테스트 데이터에서 샘플링
        test_sample = test_data.sample(min(n_iterations, len(test_data)), random_state=42)
        
        cumulative_regret = 0
        convergence_window = []
        
        for i, (idx, row) in enumerate(test_sample.iterrows()):
            # 쿼리 특성 추출
            query_features = self.predictor._extract_query_features(row)
            
            # 모델 선택
            try:
                # 토큰 예측 수행
                predicted_tokens = {}
                model_uncertainties = {}
                
                for model in available_models:
                    if model == new_model:
                        # 신규 모델: 가족 평균 사용
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
                print(f"⚠️ 라우팅 오류: {e}")
                selected_model = np.random.choice(available_models)
                utility = 0
                prediction = {}
            
            # 실제 최적 모델 결정 (ground truth)
            true_optimal = self._get_optimal_model(row, available_models)
            
            # 후회 계산
            regret = 1.0 if selected_model != true_optimal else 0.0
            cumulative_regret += regret
            
            # 실제 성능 시뮬레이션
            actual_performance = self._simulate_actual_performance(selected_model, row)
            
            # 라우터 업데이트
            router.update(selected_model, actual_performance)
            
            # 결과 저장
            results['iterations'].append(i + 1)
            results['selected_models'].append(selected_model)
            results['regrets'].append(regret)
            results['cumulative_regret'].append(cumulative_regret)
            results['performance_history'].append(actual_performance)
            
            if selected_model == new_model:
                results['new_model_selections'] += 1
            
            # 수렴 메트릭 (50 쿼리 윈도우)
            convergence_window.append(regret)
            if len(convergence_window) > 50:
                convergence_window.pop(0)
            
            if len(convergence_window) == 50:
                convergence_score = 1.0 - np.mean(convergence_window)
                results['convergence_metrics'].append(convergence_score)
        
        # 최종 통계
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
        """실제 최적 모델 결정 (품질과 비용 고려)"""
        model_scores = {}
        
        # 품질 점수가 있는 경우 활용
        if hasattr(row, 'quality_a') and hasattr(row, 'quality_b'):
            model_qualities = {
                row['model_a']: getattr(row, 'quality_a', 1000),
                row['model_b']: getattr(row, 'quality_b', 1000)
            }
        else:
            # 기본 품질 점수
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
                
                # 복잡도 보너스
                complexity_bonus = getattr(row, 'query_complexity', 0.5) * 50
                
                # 비용 페널티
                cost_penalty = self.predictor.model_costs.get(model, 0.01) * 1000
                
                final_score = base_score + complexity_bonus - cost_penalty
                model_scores[model] = final_score
        
        return max(model_scores, key=model_scores.get) if model_scores else available_models[0]
    
    def _simulate_actual_performance(self, model: str, row: pd.Series) -> float:
        """실제 성능 시뮬레이션"""
        # 품질 점수 기반 성능 계산
        if hasattr(row, 'quality_a') and row['model_a'] == model:
            base_performance = (getattr(row, 'quality_a', 1000) - 950) / 300
        elif hasattr(row, 'quality_b') and row['model_b'] == model:
            base_performance = (getattr(row, 'quality_b', 1000) - 950) / 300
        else:
            # 기본 성능 매핑
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
        
        # 노이즈 추가
        noise = np.random.normal(0, 0.1)
        return max(0.1, min(0.9, base_performance + noise))
    
    def _find_convergence_time(self, regrets: List[float], threshold: float = 0.2, 
                              window: int = 50) -> int:
        """수렴 시간 찾기"""
        if len(regrets) < window:
            return len(regrets)
        
        for i in range(window, len(regrets)):
            window_regrets = regrets[i-window:i]
            if np.mean(window_regrets) <= threshold:
                return i - window
        
        return len(regrets)
    
    def _calculate_performance_improvement(self, performances: List[float], 
                                         window: int = 50) -> float:
        """성능 개선율 계산"""
        if len(performances) < window:
            return 0.0
        
        initial_perf = np.mean(performances[:window])
        final_perf = np.mean(performances[-window:])
        
        return (final_perf - initial_perf) / initial_perf if initial_perf > 0 else 0.0 