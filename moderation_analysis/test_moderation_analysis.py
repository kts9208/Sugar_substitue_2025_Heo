"""
Moderation Analysis Test Module

조절효과 분석 모듈의 기능을 검증하는 단위 테스트와 통합 테스트를 포함합니다.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 테스트를 위한 경로 설정
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 조절효과 분석 모듈 임포트
from moderation_analysis import (
    # 핵심 분석
    ModerationAnalyzer,
    analyze_moderation_effects,
    calculate_simple_slopes,
    calculate_conditional_effects,
    
    # 데이터 로딩
    ModerationDataLoader,
    load_moderation_data,
    combine_factor_data,
    
    # 상호작용 모델링
    InteractionBuilder,
    create_interaction_terms,
    build_moderation_model,
    
    # 결과 저장
    ModerationResultsExporter,
    export_moderation_results,
    
    # 시각화
    ModerationVisualizer,
    create_moderation_plot,
    
    # 설정
    ModerationAnalysisConfig,
    create_default_moderation_config,
    create_custom_moderation_config
)


class TestModerationAnalysisConfig(unittest.TestCase):
    """조절효과 분석 설정 테스트"""
    
    def test_default_config_creation(self):
        """기본 설정 생성 테스트"""
        config = create_default_moderation_config()
        
        self.assertIsInstance(config, ModerationAnalysisConfig)
        self.assertEqual(config.estimator, "ML")
        self.assertTrue(config.standardized)
        self.assertEqual(config.bootstrap_samples, 5000)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertTrue(config.center_variables)
    
    def test_custom_config_creation(self):
        """사용자 정의 설정 생성 테스트"""
        config = create_custom_moderation_config(
            estimator="GLS",
            bootstrap_samples=1000,
            confidence_level=0.99,
            center_variables=False
        )
        
        self.assertEqual(config.estimator, "GLS")
        self.assertEqual(config.bootstrap_samples, 1000)
        self.assertEqual(config.confidence_level, 0.99)
        self.assertFalse(config.center_variables)
    
    def test_config_validation(self):
        """설정 유효성 검증 테스트"""
        # 잘못된 신뢰수준
        with self.assertRaises(ValueError):
            ModerationAnalysisConfig(confidence_level=1.5)
        
        with self.assertRaises(ValueError):
            ModerationAnalysisConfig(confidence_level=-0.1)


class TestModerationDataLoader(unittest.TestCase):
    """데이터 로더 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config = create_custom_moderation_config(
            data_dir=str(self.temp_dir),
            results_dir=str(self.temp_dir / "results")
        )
        
        # 테스트 데이터 생성
        self._create_test_data()
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """테스트용 요인 데이터 생성"""
        np.random.seed(42)
        n_samples = 100
        
        factors = {
            'health_concern': pd.DataFrame({
                'q6': np.random.normal(4, 1, n_samples),
                'q7': np.random.normal(4, 1, n_samples),
                'q8': np.random.normal(4, 1, n_samples)
            }),
            'perceived_benefit': pd.DataFrame({
                'q16': np.random.normal(3.5, 1.2, n_samples),
                'q17': np.random.normal(3.5, 1.2, n_samples)
            }),
            'nutrition_knowledge': pd.DataFrame({
                'q30': np.random.normal(3, 1, n_samples),
                'q31': np.random.normal(3, 1, n_samples),
                'q32': np.random.normal(3, 1, n_samples)
            })
        }
        
        for factor_name, factor_data in factors.items():
            factor_data.to_csv(self.temp_dir / f"{factor_name}.csv", index=False)
    
    def test_data_loader_initialization(self):
        """데이터 로더 초기화 테스트"""
        loader = ModerationDataLoader(self.test_config)
        
        self.assertEqual(loader.data_dir, Path(self.test_config.data_dir))
        self.assertIsInstance(loader.factor_items, dict)
    
    def test_single_factor_loading(self):
        """단일 요인 로딩 테스트"""
        loader = ModerationDataLoader(self.test_config)
        
        data = loader.load_single_factor('health_concern')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('q6', data.columns)
    
    def test_multiple_factors_loading(self):
        """다중 요인 로딩 테스트"""
        loader = ModerationDataLoader(self.test_config)
        
        factor_data = loader.load_multiple_factors(['health_concern', 'perceived_benefit'])
        
        self.assertIsInstance(factor_data, dict)
        self.assertEqual(len(factor_data), 2)
        self.assertIn('health_concern', factor_data)
        self.assertIn('perceived_benefit', factor_data)
    
    def test_factor_data_combination(self):
        """요인 데이터 결합 테스트"""
        loader = ModerationDataLoader(self.test_config)
        
        combined_data = loader.combine_factor_data(
            ['health_concern', 'perceived_benefit', 'nutrition_knowledge']
        )
        
        self.assertIsInstance(combined_data, pd.DataFrame)
        self.assertEqual(len(combined_data.columns), 3)
        self.assertIn('health_concern', combined_data.columns)
        self.assertIn('perceived_benefit', combined_data.columns)
        self.assertIn('nutrition_knowledge', combined_data.columns)


class TestInteractionBuilder(unittest.TestCase):
    """상호작용 빌더 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'health_concern': np.random.normal(4, 1, 100),
            'perceived_benefit': np.random.normal(3.5, 1.2, 100),
            'nutrition_knowledge': np.random.normal(3, 1, 100)
        })
        
        self.builder = InteractionBuilder()
    
    def test_interaction_terms_creation(self):
        """상호작용항 생성 테스트"""
        interaction_data = self.builder.create_interaction_terms(
            self.test_data, 'health_concern', 'nutrition_knowledge'
        )
        
        interaction_name = 'health_concern_x_nutrition_knowledge'
        self.assertIn(interaction_name, interaction_data.columns)
        
        # 상호작용항이 올바르게 계산되었는지 확인
        expected_interaction = (self.test_data['health_concern'] * 
                              self.test_data['nutrition_knowledge'])
        np.testing.assert_array_almost_equal(
            interaction_data[interaction_name], expected_interaction
        )
    
    def test_model_spec_generation(self):
        """모델 스펙 생성 테스트"""
        model_spec = self.builder.build_moderation_model_spec(
            'health_concern', 'perceived_benefit', 'nutrition_knowledge'
        )
        
        self.assertIsInstance(model_spec, str)
        self.assertIn('perceived_benefit ~', model_spec)
        self.assertIn('health_concern', model_spec)
        self.assertIn('nutrition_knowledge', model_spec)
        self.assertIn('health_concern_x_nutrition_knowledge', model_spec)
    
    def test_interaction_data_validation(self):
        """상호작용 데이터 유효성 검증 테스트"""
        # 유효한 데이터
        valid = self.builder.validate_interaction_data(
            self.test_data, 'health_concern', 'nutrition_knowledge'
        )
        self.assertTrue(valid)
        
        # 누락된 변수가 있는 데이터
        invalid_data = self.test_data.drop('nutrition_knowledge', axis=1)
        invalid = self.builder.validate_interaction_data(
            invalid_data, 'health_concern', 'nutrition_knowledge'
        )
        self.assertFalse(invalid)


class TestModerationAnalyzer(unittest.TestCase):
    """조절효과 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 데이터 생성
        np.random.seed(42)
        n = 200
        
        # 상관관계가 있는 데이터 생성
        health_concern = np.random.normal(4, 1, n)
        nutrition_knowledge = 0.3 * health_concern + np.random.normal(0, 0.8, n)
        
        # 조절효과가 있는 종속변수 생성
        perceived_benefit = (0.5 * health_concern + 
                           0.3 * nutrition_knowledge + 
                           0.2 * health_concern * nutrition_knowledge + 
                           np.random.normal(0, 0.5, n))
        
        self.test_data = pd.DataFrame({
            'health_concern': health_concern,
            'nutrition_knowledge': nutrition_knowledge,
            'perceived_benefit': perceived_benefit
        })
        
        # 변수 중심화
        for col in ['health_concern', 'nutrition_knowledge']:
            self.test_data[col] = self.test_data[col] - self.test_data[col].mean()
    
    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        analyzer = ModerationAnalyzer()
        
        self.assertIsInstance(analyzer.config, ModerationAnalysisConfig)
        self.assertIsInstance(analyzer.data_loader, ModerationDataLoader)
        self.assertIsInstance(analyzer.interaction_builder, InteractionBuilder)
    
    def test_moderation_analysis_with_data(self):
        """데이터를 사용한 조절효과 분석 테스트"""
        try:
            analyzer = ModerationAnalyzer()
            
            results = analyzer.analyze_moderation_effects(
                independent_var='health_concern',
                dependent_var='perceived_benefit',
                moderator_var='nutrition_knowledge',
                data=self.test_data
            )
            
            # 결과 구조 검증
            self.assertIn('variables', results)
            self.assertIn('coefficients', results)
            self.assertIn('moderation_test', results)
            self.assertIn('simple_slopes', results)
            
            # 변수 정보 검증
            variables = results['variables']
            self.assertEqual(variables['independent'], 'health_concern')
            self.assertEqual(variables['dependent'], 'perceived_benefit')
            self.assertEqual(variables['moderator'], 'nutrition_knowledge')
            
            # 조절효과 검정 결과 검증
            moderation_test = results['moderation_test']
            self.assertIn('interaction_coefficient', moderation_test)
            self.assertIn('p_value', moderation_test)
            self.assertIn('significant', moderation_test)
            
        except ImportError:
            self.skipTest("semopy가 설치되지 않았습니다.")
        except Exception as e:
            self.fail(f"조절효과 분석 실패: {e}")


class TestModerationResultsExporter(unittest.TestCase):
    """결과 저장기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config = create_custom_moderation_config(
            results_dir=str(self.temp_dir)
        )
        
        # 테스트용 결과 데이터
        self.test_results = {
            'variables': {
                'independent': 'health_concern',
                'dependent': 'perceived_benefit',
                'moderator': 'nutrition_knowledge',
                'interaction': 'health_concern_x_nutrition_knowledge'
            },
            'model_info': {
                'n_observations': 200,
                'n_parameters': 4
            },
            'coefficients': {
                'health_concern': {
                    'estimate': 0.5,
                    'std_error': 0.1,
                    'z_value': 5.0,
                    'p_value': 0.001,
                    'significant': True
                },
                'health_concern_x_nutrition_knowledge': {
                    'estimate': 0.2,
                    'std_error': 0.05,
                    'z_value': 4.0,
                    'p_value': 0.001,
                    'significant': True
                }
            },
            'moderation_test': {
                'interaction_coefficient': 0.2,
                'p_value': 0.001,
                'significant': True,
                'interpretation': '조절변수가 증가할수록 독립변수의 효과가 강화됨'
            },
            'simple_slopes': {
                'low': {'simple_slope': 0.3, 'p_value': 0.01, 'significant': True},
                'mean': {'simple_slope': 0.5, 'p_value': 0.001, 'significant': True},
                'high': {'simple_slope': 0.7, 'p_value': 0.001, 'significant': True}
            },
            'fit_indices': {
                'CFI': 0.95,
                'RMSEA': 0.06,
                'SRMR': 0.05
            }
        }
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_exporter_initialization(self):
        """결과 저장기 초기화 테스트"""
        exporter = ModerationResultsExporter(self.test_config)
        
        self.assertEqual(exporter.results_dir, Path(self.test_config.results_dir))
        self.assertTrue(exporter.results_dir.exists())
    
    def test_comprehensive_results_export(self):
        """포괄적 결과 저장 테스트"""
        exporter = ModerationResultsExporter(self.test_config)
        
        saved_files = exporter.export_comprehensive_results(
            self.test_results, 
            analysis_name="test_analysis"
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 0)
        
        # 저장된 파일들이 실제로 존재하는지 확인
        for file_type, file_path in saved_files.items():
            self.assertTrue(file_path.exists(), f"{file_type} 파일이 존재하지 않습니다: {file_path}")


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 테스트용 요인 데이터 생성
        np.random.seed(42)
        n = 150
        
        # 실제와 유사한 데이터 생성
        health_concern_data = pd.DataFrame({
            'q6': np.random.normal(4, 1, n),
            'q7': np.random.normal(4, 1, n),
            'q8': np.random.normal(4, 1, n)
        })
        
        perceived_benefit_data = pd.DataFrame({
            'q16': np.random.normal(3.5, 1.2, n),
            'q17': np.random.normal(3.5, 1.2, n)
        })
        
        nutrition_knowledge_data = pd.DataFrame({
            'q30': np.random.normal(3, 1, n),
            'q31': np.random.normal(3, 1, n),
            'q32': np.random.normal(3, 1, n)
        })
        
        # CSV 파일로 저장
        health_concern_data.to_csv(self.temp_dir / "health_concern.csv", index=False)
        perceived_benefit_data.to_csv(self.temp_dir / "perceived_benefit.csv", index=False)
        nutrition_knowledge_data.to_csv(self.temp_dir / "nutrition_knowledge.csv", index=False)
        
        # 테스트 설정
        self.test_config = create_custom_moderation_config(
            data_dir=str(self.temp_dir),
            results_dir=str(self.temp_dir / "results"),
            bootstrap_samples=100  # 빠른 테스트를 위해 감소
        )
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_analysis(self):
        """전체 분석 프로세스 테스트"""
        try:
            # 1. 데이터 로드
            data = load_moderation_data(
                independent_var='health_concern',
                dependent_var='perceived_benefit',
                moderator_var='nutrition_knowledge',
                config=self.test_config
            )
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            
            # 2. 조절효과 분석
            results = analyze_moderation_effects(
                independent_var='health_concern',
                dependent_var='perceived_benefit',
                moderator_var='nutrition_knowledge',
                data=data
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn('moderation_test', results)
            
            # 3. 결과 저장
            saved_files = export_moderation_results(
                results, 
                analysis_name="integration_test",
                config=self.test_config
            )
            
            self.assertIsInstance(saved_files, dict)
            self.assertGreater(len(saved_files), 0)
            
        except ImportError:
            self.skipTest("semopy가 설치되지 않았습니다.")
        except Exception as e:
            self.fail(f"통합 테스트 실패: {e}")


def run_tests():
    """테스트 실행"""
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_classes = [
        TestModerationAnalysisConfig,
        TestModerationDataLoader,
        TestInteractionBuilder,
        TestModerationAnalyzer,
        TestModerationResultsExporter,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
