"""
Multinomial Logit Model 테스트 코드

각 모듈의 기능을 검증하는 단위 테스트와 통합 테스트를 포함합니다.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# 테스트를 위한 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

from multinomial_logit import (
    DCEDataLoader, DCEDataPreprocessor, ModelConfig, ModelConfigManager,
    MultinomialLogitEstimator, ResultsAnalyzer,
    load_dce_data, preprocess_dce_data, create_default_config,
    estimate_multinomial_logit, analyze_results
)


class TestDCEDataLoader(unittest.TestCase):
    """DCE 데이터 로더 테스트"""
    
    def setUp(self):
        """테스트 데이터 설정"""
        self.test_data_dir = "processed_data/dce_data"
    
    def test_data_loader_initialization(self):
        """데이터 로더 초기화 테스트"""
        if Path(self.test_data_dir).exists():
            loader = DCEDataLoader(self.test_data_dir)
            self.assertIsInstance(loader, DCEDataLoader)
            self.assertEqual(str(loader.data_dir), self.test_data_dir)
    
    def test_load_choice_matrix(self):
        """선택 매트릭스 로딩 테스트"""
        if Path(self.test_data_dir).exists():
            loader = DCEDataLoader(self.test_data_dir)
            df = loader.load_choice_matrix()
            
            # 기본 검증
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # 필수 컬럼 확인
            required_columns = [
                'respondent_id', 'question_id', 'alternative', 'chosen',
                'sugar_type', 'health_label', 'price', 'sugar_free', 'has_health_label'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns)
    
    def test_data_summary(self):
        """데이터 요약 테스트"""
        if Path(self.test_data_dir).exists():
            loader = DCEDataLoader(self.test_data_dir)
            summary = loader.get_data_summary()
            
            self.assertIsInstance(summary, dict)
            self.assertIn('total_respondents', summary)
            self.assertIn('total_questions', summary)
            self.assertGreater(summary['total_respondents'], 0)


class TestDCEDataPreprocessor(unittest.TestCase):
    """DCE 데이터 전처리기 테스트"""
    
    def setUp(self):
        """테스트 데이터 생성"""
        # 간단한 테스트 데이터 생성
        self.test_data = pd.DataFrame({
            'respondent_id': [1, 1, 1, 1, 2, 2, 2, 2],
            'question_id': ['q1', 'q1', 'q2', 'q2', 'q1', 'q1', 'q2', 'q2'],
            'alternative': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'chosen': [1, 0, 0, 1, 1, 0, 0, 1],
            'sugar_type': ['무설탕', '일반당', '일반당', '무설탕', '무설탕', '일반당', '일반당', '무설탕'],
            'health_label': ['있음', '없음', '없음', '있음', '있음', '없음', '없음', '있음'],
            'price': [2500, 2000, 3000, 2500, 2500, 2000, 3000, 2500],
            'sugar_free': [1, 0, 0, 1, 1, 0, 0, 1],
            'has_health_label': [1, 0, 0, 1, 1, 0, 0, 1]
        })
    
    def test_preprocessor_initialization(self):
        """전처리기 초기화 테스트"""
        preprocessor = DCEDataPreprocessor()
        self.assertIsInstance(preprocessor, DCEDataPreprocessor)
        self.assertFalse(preprocessor.fitted)
    
    def test_prepare_choice_data(self):
        """선택 데이터 준비 테스트"""
        preprocessor = DCEDataPreprocessor()
        processed_data = preprocessor.prepare_choice_data(self.test_data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIn('price_scaled', processed_data.columns)
        self.assertIn('alternative_B', processed_data.columns)
        
        # 가격 스케일링 확인
        self.assertTrue((processed_data['price_scaled'] == processed_data['price'] / 1000).all())
    
    def test_create_choice_sets(self):
        """선택 세트 생성 테스트"""
        preprocessor = DCEDataPreprocessor()
        processed_data = preprocessor.prepare_choice_data(self.test_data)
        X, y, choice_sets = preprocessor.create_choice_sets(processed_data)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(choice_sets, np.ndarray)
        
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 4)  # 4개 특성
        self.assertGreater(len(choice_sets), 0)
    
    def test_feature_names(self):
        """특성 이름 테스트"""
        preprocessor = DCEDataPreprocessor()
        feature_names = preprocessor.get_feature_names()
        
        expected_features = ['sugar_free', 'has_health_label', 'price_scaled', 'alternative_B']
        self.assertEqual(feature_names, expected_features)


class TestModelConfig(unittest.TestCase):
    """모델 설정 테스트"""
    
    def test_model_config_creation(self):
        """모델 설정 생성 테스트"""
        config = ModelConfig()
        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.tolerance, 1e-6)
        self.assertEqual(config.method, 'bfgs')
    
    def test_config_manager(self):
        """설정 관리자 테스트"""
        feature_names = ['feature1', 'feature2']
        config = create_default_config(feature_names)
        manager = ModelConfigManager(config)
        
        self.assertIsInstance(manager, ModelConfigManager)
        self.assertEqual(manager.config.feature_names, feature_names)
    
    def test_optimization_params(self):
        """최적화 파라미터 테스트"""
        config = ModelConfig()
        manager = ModelConfigManager(config)
        opt_params = manager.get_optimization_params()
        
        self.assertIsInstance(opt_params, dict)
        self.assertIn('method', opt_params)
        self.assertIn('maxiter', opt_params)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        """실제 데이터를 사용한 테스트 설정"""
        self.data_dir = "processed_data/dce_data"
        self.data_available = Path(self.data_dir).exists()
    
    def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        if not self.data_available:
            self.skipTest("테스트 데이터가 없습니다")
        
        try:
            # 1. 데이터 로딩
            data = load_dce_data(self.data_dir)
            self.assertIn('choice_matrix', data)
            
            # 2. 데이터 전처리
            choice_matrix = data['choice_matrix']
            X, y, choice_sets, feature_names = preprocess_dce_data(choice_matrix)
            
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(choice_sets, np.ndarray)
            self.assertIsInstance(feature_names, list)
            
            # 3. 모델 설정
            config = create_default_config(feature_names)
            config.max_iterations = 100  # 테스트용으로 줄임
            config.verbose = False
            
            # 4. 모델 추정 (작은 샘플로)
            # 테스트를 위해 데이터 크기 줄이기
            n_test_sets = min(10, len(choice_sets))
            test_indices = []
            for i in range(n_test_sets):
                start_idx = choice_sets[i]
                if i < len(choice_sets) - 1:
                    end_idx = choice_sets[i + 1]
                else:
                    end_idx = len(y)
                test_indices.extend(range(start_idx, end_idx))
            
            X_test = X[test_indices]
            y_test = y[test_indices]
            choice_sets_test = np.arange(0, len(test_indices), 2)  # 2개 대안씩
            
            results = estimate_multinomial_logit(X_test, y_test, choice_sets_test, config)
            
            # 5. 결과 분석
            analyzer = analyze_results(results)
            coeffs_table = analyzer.create_coefficients_table()
            
            self.assertIsInstance(results, dict)
            self.assertIn('estimation_results', results)
            self.assertIsInstance(coeffs_table, pd.DataFrame)
            
        except Exception as e:
            self.fail(f"통합 테스트 실패: {e}")
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        # 잘못된 데이터로 테스트
        with self.assertRaises(FileNotFoundError):
            DCEDataLoader("nonexistent_directory")
        
        # 빈 데이터로 테스트
        empty_data = pd.DataFrame()
        preprocessor = DCEDataPreprocessor()
        
        with self.assertRaises(Exception):
            preprocessor.prepare_choice_data(empty_data)


def run_basic_functionality_test():
    """기본 기능 테스트 실행"""
    print("기본 기능 테스트 시작...")
    
    # 데이터 경로 확인
    data_dir = "processed_data/dce_data"
    if not Path(data_dir).exists():
        print(f"경고: 테스트 데이터 디렉토리가 없습니다: {data_dir}")
        return False
    
    try:
        # 1. 데이터 로딩 테스트
        print("1. 데이터 로딩 테스트...")
        data = load_dce_data(data_dir)
        print(f"   ✓ 데이터 로딩 성공: {len(data)} 개 데이터셋")
        
        # 2. 전처리 테스트
        print("2. 데이터 전처리 테스트...")
        choice_matrix = data['choice_matrix']
        X, y, choice_sets, feature_names = preprocess_dce_data(choice_matrix)
        print(f"   ✓ 전처리 성공: X={X.shape}, y={y.shape}, choice_sets={len(choice_sets)}")
        
        # 3. 모델 설정 테스트
        print("3. 모델 설정 테스트...")
        config = create_default_config(feature_names)
        config.max_iterations = 50  # 테스트용
        config.verbose = False
        print(f"   ✓ 설정 생성 성공: {len(feature_names)} 개 특성")
        
        # 4. 작은 샘플로 모델 추정 테스트
        print("4. 모델 추정 테스트 (작은 샘플)...")
        n_test = min(20, len(choice_sets))
        test_end = choice_sets[n_test] if n_test < len(choice_sets) else len(y)
        
        X_test = X[:test_end]
        y_test = y[:test_end]
        choice_sets_test = choice_sets[:n_test]
        
        results = estimate_multinomial_logit(X_test, y_test, choice_sets_test, config)
        print(f"   ✓ 모델 추정 성공: 수렴={results['convergence_info']['converged']}")
        
        # 5. 결과 분석 테스트
        print("5. 결과 분석 테스트...")
        analyzer = analyze_results(results)
        coeffs_table = analyzer.create_coefficients_table()
        print(f"   ✓ 결과 분석 성공: {len(coeffs_table)} 개 계수")
        
        print("모든 기본 기능 테스트 통과! ✓")
        return True
        
    except Exception as e:
        print(f"기본 기능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 기본 기능 테스트 먼저 실행
    print("=" * 50)
    print("기본 기능 테스트")
    print("=" * 50)
    
    basic_test_passed = run_basic_functionality_test()
    
    print("\n" + "=" * 50)
    print("단위 테스트")
    print("=" * 50)
    
    # 단위 테스트 실행
    if basic_test_passed:
        unittest.main(verbosity=2)
    else:
        print("기본 기능 테스트가 실패하여 단위 테스트를 건너뜁니다.")
        sys.exit(1)
