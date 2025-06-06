import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LMArenaDataLoader:
    """LMArena 데이터 로더 및 전처리 클래스"""
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        self.raw_data = None
        self.processed_data = None
        
    def download_lmarena_data(self, dataset_name: str = "lmarena-ai/arena-human-preference-55k") -> pd.DataFrame:
        """
        Hugging Face에서 LMArena 데이터 다운로드
        
        Args:
            dataset_name: 다운로드할 데이터셋 이름
            
        Returns:
            pd.DataFrame: 원시 대화 데이터
        """
        print(f"🔄 LMArena 데이터 다운로드 중: {dataset_name}")
        
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            
            # train split을 DataFrame으로 변환
            if 'train' in dataset:
                self.raw_data = dataset['train'].to_pandas()
            else:
                # train split이 없으면 첫 번째 available split 사용
                first_split = list(dataset.keys())[0]
                self.raw_data = dataset[first_split].to_pandas()
                
            print(f"✅ 데이터 로드 완료: {len(self.raw_data):,} 개의 대화")
            print(f"📊 컬럼: {list(self.raw_data.columns)}")
            
            # 실제 데이터 구조에 맞게 변환
            processed_data = self._process_arena_data(self.raw_data)
            # 처리된 데이터를 self.raw_data에 할당하여 preprocess_data에서 사용할 수 있도록 함
            self.raw_data = processed_data
            return processed_data
            
        except Exception as e:
            print(f"❌ 데이터 다운로드 실패: {e}")
            print("💡 대안으로 시뮬레이션 데이터를 생성합니다.")
            return self._generate_simulated_data()
    
    def _process_arena_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """실제 LMArena 데이터를 실험용 형식으로 변환"""
        print("🔄 LMArena 데이터 형식 변환 중...")
        
        try:
            processed_conversations = []
            
            for idx, row in raw_data.iterrows():
                # 기본 정보 추출
                conversation_data = {
                    'conversation_id': row['id'],
                    'model_a': row['model_a'],
                    'model_b': row['model_b'], 
                    'prompt': row['prompt'] if pd.notna(row['prompt']) else '',
                    'response_a': row['response_a'] if pd.notna(row['response_a']) else '',
                    'response_b': row['response_b'] if pd.notna(row['response_b']) else '',
                }
                
                # 승자 정보 처리 (실제 데이터 구조에 맞게)
                if row['winner_model_a'] == 1:
                    conversation_data['winner'] = 'model_a'
                elif row['winner_model_b'] == 1:
                    conversation_data['winner'] = 'model_b'
                elif row['winner_tie'] == 1:
                    conversation_data['winner'] = 'tie'
                else:
                    conversation_data['winner'] = 'unknown'
                
                # 시간 정보 (9개월 범위에서 임의 생성)
                start_date = pd.Timestamp('2023-04-01')
                end_date = pd.Timestamp('2024-01-01')
                days_offset = int(np.random.exponential(120))
                days_offset = min(days_offset, 270)
                conversation_data['timestamp'] = start_date + pd.Timedelta(days=days_offset)
                
                # 쿼리 특성 추출
                prompt_text = str(conversation_data['prompt'])
                
                # 프롬프트가 리스트 형태의 문자열인 경우 처리
                if prompt_text.startswith('[') and prompt_text.endswith(']'):
                    try:
                        import ast
                        prompt_list = ast.literal_eval(prompt_text)
                        if isinstance(prompt_list, list):
                            prompt_text = ' '.join(str(p) for p in prompt_list)
                    except:
                        pass  # 파싱 실패시 원본 사용
                
                conversation_data['query_length'] = len(prompt_text)
                conversation_data['query_complexity'] = self._estimate_complexity(prompt_text)
                conversation_data['query_type'] = self._classify_query_type(prompt_text)
                
                # 응답 토큰 수 추정 (더 정확한 방식)
                response_a_text = str(conversation_data['response_a'])
                response_b_text = str(conversation_data['response_b'])
                
                # 단어 수 기반 토큰 추정 (영어: 1.3배, 다른 언어: 1.5배)
                def estimate_tokens(text):
                    if not text or text == 'nan':
                        return 50  # 기본값
                    words = len(text.split())
                    # 대략적 토큰 수 (GPT 토크나이저 기준)
                    return max(10, int(words * 1.3))
                
                conversation_data['response_tokens_a'] = estimate_tokens(response_a_text)
                conversation_data['response_tokens_b'] = estimate_tokens(response_b_text)
                
                processed_conversations.append(conversation_data)
                
                # 진행 상황 출력 (5000개마다)
                if idx % 5000 == 0:
                    print(f"  📈 처리 중... {idx:,}/{len(raw_data):,}")
            
            result_df = pd.DataFrame(processed_conversations)
            print(f"✅ 변환 완료: {len(result_df):,} 개의 대화")
            
            # 데이터 품질 체크
            print(f"📊 데이터 품질 체크:")
            print(f"  • 고유 모델 A: {result_df['model_a'].nunique()}개")
            print(f"  • 고유 모델 B: {result_df['model_b'].nunique()}개")
            print(f"  • 승자 분포: {result_df['winner'].value_counts().to_dict()}")
            print(f"  • 쿼리 타입 분포: {result_df['query_type'].value_counts().to_dict()}")
            print(f"  • 평균 쿼리 길이: {result_df['query_length'].mean():.1f}")
            print(f"  • 평균 응답 토큰 A: {result_df['response_tokens_a'].mean():.1f}")
            print(f"  • 평균 응답 토큰 B: {result_df['response_tokens_b'].mean():.1f}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ 데이터 변환 실패: {e}")
            print("💡 시뮬레이션 데이터로 대체합니다.")
            return self._generate_simulated_data()
    
    def _estimate_complexity(self, text: str) -> float:
        """텍스트 복잡도 추정 (개선된 버전)"""
        if not text or text == 'nan':
            return 0.5
        
        text = str(text).lower()
        complexity_score = 0.2  # 기본 점수
        
        # 길이 기반 복잡도
        if len(text) > 500:
            complexity_score += 0.3
        elif len(text) > 200:
            complexity_score += 0.2
        elif len(text) > 100:
            complexity_score += 0.1
        
        # 문장 수 기반
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 5:
            complexity_score += 0.15
        elif sentences > 2:
            complexity_score += 0.1
        
        # 복잡한 키워드 기반
        complex_keywords = [
            'algorithm', 'analysis', 'technical', 'code', 'math', 'explain', 'complex',
            'implementation', 'optimization', 'architecture', 'design', 'strategy',
            'reasoning', 'logic', 'computation', 'methodology', 'framework'
        ]
        keyword_count = sum(1 for keyword in complex_keywords if keyword in text)
        complexity_score += min(0.3, keyword_count * 0.05)
        
        # 코드 블록이나 수식 감지
        if '```' in text or '$$' in text or 'def ' in text or 'class ' in text:
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    def _classify_query_type(self, text: str) -> str:
        """쿼리 타입 분류 (개선된 버전)"""
        if not text or text == 'nan':
            return 'general'
        
        text = str(text).lower()
        
        # 코딩 관련
        coding_keywords = ['code', 'programming', 'function', 'python', 'javascript', 
                          'algorithm', 'debug', 'syntax', 'implementation', '```']
        if any(word in text for word in coding_keywords):
            return 'coding'
        
        # 수학 관련  
        math_keywords = ['math', 'calculate', 'equation', 'solve', 'formula', 
                        'theorem', 'proof', 'derivative', 'integral']
        if any(word in text for word in math_keywords):
            return 'math'
        
        # 창작 관련
        creative_keywords = ['write', 'story', 'creative', 'poem', 'fiction', 
                           'character', 'plot', 'narrative', 'novel']
        if any(word in text for word in creative_keywords):
            return 'creative'
        
        # 분석 관련
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'explain', 'discuss',
                           'argument', 'critique', 'assessment', 'review', 'examine']
        if any(word in text for word in analysis_keywords):
            return 'analysis'
        
        # 지식 질문
        knowledge_keywords = ['what', 'how', 'why', 'when', 'where', 'who', 
                            'definition', 'meaning', 'history', 'fact']
        if any(word in text for word in knowledge_keywords):
            return 'knowledge'
        
        return 'general'
    
    def _generate_simulated_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        실제 LMArena 패턴을 기반으로 한 시뮬레이션 데이터 생성
        (실제 데이터에 접근할 수 없는 경우 사용)
        """
        print(f"🎲 시뮬레이션 데이터 생성 중: {n_samples:,} 샘플")
        
        # 실제 LMArena에서 관찰된 모델들
        models = [
            'gpt-4-0613', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0613',
            'claude-2.1', 'claude-2.0', 'claude-instant-1',
            'gemini-pro', 'llama-2-70b-chat', 'llama-2-13b-chat',
            'mixtral-8x7b-instruct'
        ]
        
        # 모델 품질 점수 (Elo 기반)
        model_quality = {
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
        
        # 시뮬레이션 데이터 생성
        np.random.seed(42)
        data = []
        
        # 시간 범위 (9개월)
        start_date = pd.Timestamp('2023-04-01')
        end_date = pd.Timestamp('2024-01-01')
        
        for i in range(n_samples):
            # 시간 생성 (지수 분포로 최근 데이터 더 많이)
            days_offset = int(np.random.exponential(120))
            days_offset = min(days_offset, 270)
            timestamp = start_date + pd.Timedelta(days=days_offset)
            
            # 모델 쌍 선택
            model_a, model_b = np.random.choice(models, 2, replace=False)
            
            # 쿼리 유형
            query_types = ['coding', 'math', 'creative', 'analysis', 'knowledge']
            query_type = np.random.choice(query_types)
            
            # 쿼리 길이와 복잡도
            if query_type == 'coding':
                query_length = int(np.random.normal(80, 20))
                complexity = np.random.uniform(0.6, 0.9)
            elif query_type == 'math':
                query_length = int(np.random.normal(45, 15))
                complexity = np.random.uniform(0.7, 1.0)
            elif query_type == 'creative':
                query_length = int(np.random.normal(60, 25))
                complexity = np.random.uniform(0.3, 0.7)
            elif query_type == 'analysis':
                query_length = int(np.random.normal(90, 30))
                complexity = np.random.uniform(0.5, 0.8)
            else:  # knowledge
                query_length = int(np.random.normal(35, 12))
                complexity = np.random.uniform(0.2, 0.6)
            
            query_length = max(10, query_length)
            
            # 승자 결정 (품질 점수 + 복잡도 보정 + 노이즈)
            quality_a = model_quality[model_a]
            quality_b = model_quality[model_b]
            
            # 복잡한 쿼리일수록 고급 모델에게 유리
            complex_bonus_a = complexity * 50 if quality_a > 1100 else 0
            complex_bonus_b = complexity * 50 if quality_b > 1100 else 0
            
            # 최종 점수
            score_a = quality_a + complex_bonus_a + np.random.normal(0, 50)
            score_b = quality_b + complex_bonus_b + np.random.normal(0, 50)
            
            if abs(score_a - score_b) < 30:
                winner = 'tie'
            elif score_a > score_b:
                winner = 'model_a'
            else:
                winner = 'model_b'
            
            # 응답 토큰 수 시뮬레이션 - 더 현실적으로
            # 모델별 토큰 생성 특성
            model_token_patterns = {
                'gpt-4-1106-preview': {'base': 180, 'std': 65, 'complexity_factor': 1.3},
                'gpt-4-0613': {'base': 175, 'std': 60, 'complexity_factor': 1.2},
                'claude-2.1': {'base': 220, 'std': 80, 'complexity_factor': 1.5},
                'claude-2.0': {'base': 210, 'std': 75, 'complexity_factor': 1.4},
                'claude-instant-1': {'base': 120, 'std': 45, 'complexity_factor': 0.8},
                'gemini-pro': {'base': 160, 'std': 55, 'complexity_factor': 1.1},
                'llama-2-70b-chat': {'base': 140, 'std': 50, 'complexity_factor': 1.0},
                'llama-2-13b-chat': {'base': 100, 'std': 40, 'complexity_factor': 0.7},
                'mixtral-8x7b-instruct': {'base': 155, 'std': 52, 'complexity_factor': 1.0},
                'gpt-3.5-turbo-0613': {'base': 135, 'std': 48, 'complexity_factor': 0.9}
            }
            
            # 쿼리 유형별 토큰 승수
            type_multipliers = {
                'coding': 1.6,  # 코딩은 더 긴 응답
                'math': 1.2,    # 수학은 중간 길이
                'creative': 1.8, # 창작은 가장 긴 응답
                'analysis': 1.4, # 분석은 상당히 긴 응답
                'knowledge': 0.8 # 지식 질문은 짧은 응답
            }
            
            # 모델 A 토큰 생성
            pattern_a = model_token_patterns[model_a]
            base_tokens_a = pattern_a['base'] + np.random.normal(0, pattern_a['std'])
            complexity_bonus_a = complexity * pattern_a['complexity_factor'] * 30
            type_bonus_a = base_tokens_a * (type_multipliers[query_type] - 1)
            length_factor_a = min(2.0, np.log1p(query_length) / 3.5)  # 쿼리 길이 영향
            
            # 랜덤 노이즈와 이상치
            if np.random.random() < 0.05:  # 5% 확률로 이상치
                noise_a = np.random.uniform(-50, 150)
            else:
                noise_a = np.random.normal(0, 20)
            
            response_tokens_a = max(15, int(base_tokens_a + complexity_bonus_a + type_bonus_a * length_factor_a + noise_a))
            
            # 모델 B 토큰 생성 (독립적)
            pattern_b = model_token_patterns[model_b]
            base_tokens_b = pattern_b['base'] + np.random.normal(0, pattern_b['std'])
            complexity_bonus_b = complexity * pattern_b['complexity_factor'] * 30
            type_bonus_b = base_tokens_b * (type_multipliers[query_type] - 1)
            length_factor_b = min(2.0, np.log1p(query_length) / 3.5)
            
            if np.random.random() < 0.05:  # 5% 확률로 이상치
                noise_b = np.random.uniform(-50, 150)
            else:
                noise_b = np.random.normal(0, 20)
                
            response_tokens_b = max(15, int(base_tokens_b + complexity_bonus_b + type_bonus_b * length_factor_b + noise_b))
            
            data.append({
                'conversation_id': f"conv_{i:06d}",
                'timestamp': timestamp,
                'model_a': model_a,
                'model_b': model_b,
                'query_type': query_type,
                'query_length': query_length,
                'query_complexity': complexity,
                'response_tokens_a': response_tokens_a,
                'response_tokens_b': response_tokens_b,
                'winner': winner,
                'quality_a': quality_a,
                'quality_b': quality_b
            })
        
        self.raw_data = pd.DataFrame(data)
        print(f"✅ 시뮬레이션 데이터 생성 완료!")
        return self.raw_data
    
    def preprocess_data(self) -> pd.DataFrame:
        """데이터 전처리 및 특성 추출"""
        if self.raw_data is None:
            raise ValueError("데이터가 로드되지 않았습니다. download_lmarena_data()를 먼저 실행하세요.")
        
        print("🔄 데이터 전처리 중...")
        df = self.raw_data.copy()
        
        # 시간 특성 추출 (이미 변환된 데이터에서)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        else:
            # timestamp가 없으면 기본값으로 생성
            print("⚠️ timestamp 컬럼이 없습니다. 임의로 생성합니다.")
            start_date = pd.Timestamp('2023-04-01')
            end_date = pd.Timestamp('2024-01-01')
            timestamps = []
            for i in range(len(df)):
                days_offset = int(np.random.exponential(120))
                days_offset = min(days_offset, 270)
                timestamps.append(start_date + pd.Timedelta(days=days_offset))
            
            df['timestamp'] = pd.to_datetime(timestamps)
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        
        # 쿼리 특성 정규화 (이미 변환된 데이터에서)
        if 'query_length' in df.columns:
            df['query_tokens'] = df['query_length'] / 4  # 대략적 토큰 수
            df['length_normalized'] = np.log1p(df['query_length'])
        else:
            # 기본값 설정
            df['query_length'] = 50
            df['query_tokens'] = 12.5
            df['length_normalized'] = np.log1p(50)
            df['query_complexity'] = 0.5
            df['query_type'] = 'knowledge'
        
        # 모델 가족 분류
        def get_model_family(model_name):
            model_lower = model_name.lower()
            if 'gpt' in model_lower or 'chatgpt' in model_lower:
                return 'openai'
            elif 'claude' in model_lower:
                return 'anthropic'
            elif 'gemini' in model_lower or 'bard' in model_lower:
                return 'google'
            elif 'llama' in model_lower:
                return 'meta'
            elif 'mixtral' in model_lower or 'mistral' in model_lower:
                return 'mistral'
            elif 'vicuna' in model_lower:
                return 'lmsys'
            elif 'alpaca' in model_lower:
                return 'stanford'
            elif 'wizardlm' in model_lower:
                return 'microsoft'
            elif 'palm' in model_lower:
                return 'google'
            else:
                return 'other'
        
        df['family_a'] = df['model_a'].apply(get_model_family)
        df['family_b'] = df['model_b'].apply(get_model_family)
        
        # 비용 정보 추가 (토큰당 비용 - 실제 모델들 포함)
        cost_per_1k_tokens = {
            # OpenAI models
            'gpt-4-1106-preview': 0.03,
            'gpt-4-0613': 0.03,
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.03,
            'gpt-3.5-turbo-0613': 0.002,
            'gpt-3.5-turbo': 0.002,
            'chatgpt': 0.002,
            
            # Anthropic models
            'claude-2.1': 0.024,
            'claude-2.0': 0.024,
            'claude-2': 0.024,
            'claude-instant-1': 0.00163,
            'claude-instant': 0.00163,
            'claude-3-opus-20240229': 0.075,
            'claude-3-sonnet-20240229': 0.015,
            'claude-3-haiku-20240307': 0.00125,
            
            # Google models
            'gemini-pro': 0.0005,
            'palm-2': 0.001,
            'bard': 0.0005,
            
            # Meta models
            'llama-2-70b-chat': 0.00275,
            'llama-2-13b-chat': 0.0013,
            'llama-2-7b-chat': 0.001,
            
            # Mistral models
            'mixtral-8x7b-instruct': 0.00024,
            'mistral-7b-instruct': 0.0002,
            
            # Open source / research models (estimated costs)
            'vicuna-33b': 0.002,
            'vicuna-13b': 0.001,
            'vicuna-7b': 0.0008,
            'alpaca-7b': 0.0008,
            'wizardlm-70b': 0.003,
            'wizardlm-13b': 0.001,
        }
        
        # 토큰 수가 없으면 추정 (시뮬레이션 데이터에서는 이미 있으므로 건드리지 않음)
        if 'response_tokens_a' not in df.columns:
            print("⚠️ 토큰 수 컬럼이 없어서 기본값을 설정합니다.")
            df['response_tokens_a'] = 150
            df['response_tokens_b'] = 150
        else:
            print(f"✅ 기존 토큰 수 유지: A 평균={df['response_tokens_a'].mean():.1f}, B 평균={df['response_tokens_b'].mean():.1f}")
            print(f"   토큰 수 분포: A std={df['response_tokens_a'].std():.1f}, B std={df['response_tokens_b'].std():.1f}")
            print(f"   토큰 수 범위: A {df['response_tokens_a'].min()}-{df['response_tokens_a'].max()}, B {df['response_tokens_b'].min()}-{df['response_tokens_b'].max()}")
            print(f"   고유 토큰 값: A {df['response_tokens_a'].nunique()}개, B {df['response_tokens_b'].nunique()}개")
            # 기존 토큰 수가 있으면 그대로 유지 (덮어쓰지 않음)
        
        def get_cost(model, tokens):
            return (tokens / 1000) * cost_per_1k_tokens.get(model, 0.01)
        
        df['cost_a'] = df.apply(lambda row: get_cost(row['model_a'], row['response_tokens_a']), axis=1)
        df['cost_b'] = df.apply(lambda row: get_cost(row['model_b'], row['response_tokens_b']), axis=1)
        
        # 품질 정규화 (0-1 스케일) - 임의로 생성
        if 'quality_a' not in df.columns:
            # 모델별 대략적 품질 점수 (Elo 기반 추정)
            quality_map = {
                # Tier 1: Top performing models
                'gpt-4': 1250, 'gpt-4-turbo': 1250, 'gpt-4-1106-preview': 1250, 'gpt-4-0613': 1200,
                'claude-3-opus-20240229': 1240, 'claude-2.1': 1180, 'claude-2.0': 1150, 'claude-2': 1150,
                'claude-3-sonnet-20240229': 1170, 'claude-3-haiku-20240307': 1100,
                
                # Tier 2: Strong models
                'gemini-pro': 1140, 'mixtral-8x7b-instruct': 1120, 'llama-2-70b-chat': 1080,
                'gpt-3.5-turbo': 1050, 'gpt-3.5-turbo-0613': 1050, 'chatgpt': 1050,
                'claude-instant-1': 970, 'claude-instant': 970,
                
                # Tier 3: Mid-range models
                'vicuna-33b': 1020, 'llama-2-13b-chat': 1000, 'wizardlm-70b': 1010,
                'palm-2': 980, 'bard': 980,
                
                # Tier 4: Smaller models
                'vicuna-13b': 950, 'wizardlm-13b': 940, 'llama-2-7b-chat': 920,
                'vicuna-7b': 900, 'alpaca-7b': 880, 'mistral-7b-instruct': 930
            }
            
            # 기본값 설정 (매핑에 없는 모델들)
            def get_quality_score(model_name):
                if model_name in quality_map:
                    return quality_map[model_name]
                else:
                    # 모델 이름에서 유추
                    model_lower = model_name.lower()
                    if '70b' in model_lower or '65b' in model_lower:
                        return 1050  # 큰 모델
                    elif '30b' in model_lower or '33b' in model_lower:
                        return 1000  # 중간 모델
                    elif '13b' in model_lower or '15b' in model_lower:
                        return 950   # 작은 모델
                    elif '7b' in model_lower or '6b' in model_lower:
                        return 900   # 매우 작은 모델
                    else:
                        return 1000  # 기본값
            
            df['quality_a'] = df['model_a'].apply(get_quality_score)
            df['quality_b'] = df['model_b'].apply(get_quality_score)
            
            # 정규화
            min_quality = 850
            max_quality = 1300
            df['quality_normalized_a'] = (df['quality_a'] - min_quality) / (max_quality - min_quality)
            df['quality_normalized_b'] = (df['quality_b'] - min_quality) / (max_quality - min_quality)
        
        self.processed_data = df
        print(f"✅ 전처리 완료! 최종 데이터: {len(df):,} 개 대화")
        return df
    
    def get_temporal_splits(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """시계열 기반 데이터 분할"""
        if self.processed_data is None:
            raise ValueError("전처리된 데이터가 없습니다. preprocess_data()를 먼저 실행하세요.")
        
        # 시간순 정렬
        df_sorted = self.processed_data.sort_values('timestamp').reset_index(drop=True)
        
        n_total = len(df_sorted)
        train_end = int(train_ratio * n_total)
        val_end = int((train_ratio + val_ratio) * n_total)
        
        # 버퍼 구간 (일주일)
        buffer_samples = max(1, int(0.02 * n_total))
        
        train_data = df_sorted.iloc[:train_end]
        val_data = df_sorted.iloc[train_end + buffer_samples:val_end]
        test_data = df_sorted.iloc[val_end + buffer_samples:]
        
        print(f"📊 데이터 분할 완료:")
        print(f"  🔹 훈련: {len(train_data):,} 개 ({train_data['timestamp'].min()} ~ {train_data['timestamp'].max()})")
        print(f"  🔹 검증: {len(val_data):,} 개 ({val_data['timestamp'].min()} ~ {val_data['timestamp'].max()})")
        print(f"  🔹 테스트: {len(test_data):,} 개 ({test_data['timestamp'].min()} ~ {test_data['timestamp'].max()})")
        
        return train_data, val_data, test_data
    
    def get_summary_stats(self) -> Dict:
        """데이터 요약 통계"""
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        stats = {
            'total_conversations': len(df),
            'unique_models': df[['model_a', 'model_b']].stack().nunique(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'most_common_models': df[['model_a', 'model_b']].stack().value_counts().head(10).to_dict()
        }
        
        # winner 컬럼이 있으면 추가
        if 'winner' in df.columns:
            stats['winner_distribution'] = df['winner'].value_counts().to_dict()
        
        # query_type 컬럼이 있으면 추가
        if 'query_type' in df.columns:
            stats['query_types'] = df['query_type'].value_counts().to_dict()
        
        # query_length 컬럼이 있으면 추가
        if 'query_length' in df.columns:
            stats['avg_query_length'] = df['query_length'].mean()
        
        # query_complexity 컬럼이 있으면 추가
        if 'query_complexity' in df.columns:
            stats['avg_complexity'] = df['query_complexity'].mean()
        
        return stats 