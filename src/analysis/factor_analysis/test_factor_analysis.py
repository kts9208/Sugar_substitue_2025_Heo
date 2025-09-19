"""
Factor Analysis 모듈 테스트

이 모듈은 factor analysis 패키지의 모든 기능을 테스트합니다.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# 테스트를 위한 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from factor_analysis import (
    FactorDataLoader, FactorAnalyzer, FactorResultsExporter,
    FactorAnalysisConfig, create_factor_model_spec,
    load_factor_data, get_available_factors, analyze_factor_loading,
    export_factor_results
)


class TestFactorDataLoader(unittest.TestCase):
    """FactorDataLoader 테스트"""
    
    def setUp(self):
        """테스트 데이터 준비"""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # 테스트용 CSV 파일 생성
        self.create_test_data()
        
        self.loader = FactorDataLoader(self.test_dir)
    
    def tearDown(self):
        """테스트 데이터 정리"""
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """테스트용 데이터 생성"""
        # health_concern 테스트 데이터
        health_data = pd.DataFrame({
            'no': range(1, 101),
            'q6': np.random.randint(1, 8, 100),
            'q7': np.random.randint(1, 8, 100),
            'q8': np.random.randint(1, 8, 100),
            'q9': np.random.randint(1, 8, 100),
            'q10': np.random.randint(1, 8, 100),
            'q11': np.random.randint(1, 8, 100)
        })
        health_data.to_csv(self.test_dir / 'health_concern.csv', index=False)
        
        # perceived_benefit 테스트 데이터
        benefit_data = pd.DataFrame({
            'no': range(1, 101),
            'q12': np.random.randint(1, 8, 100),
            'q13': np.random.randint(1, 8, 100),
            'q14': np.random.randint(1, 8, 100),
            'q15': np.random.randint(1, 8, 100),
            'q16': np.random.randint(1, 8, 100),
            'q17': np.random.randint(1, 8, 100)
        })
        benefit_data.to_csv(self.test_dir / 'perceived_benefit.csv', index=False)
    
    def test_load_single_factor(self):
        """단일 요인 로딩 테스트"""
        data = self.loader.load_single_factor('health_concern')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('no', data.columns)
        self.assertIn('q6', data.columns)
    
    def test_load_multiple_factors(self):
        """다중 요인 로딩 테스트"""
        factor_data = self.loader.load_multiple_factors(['health_concern', 'perceived_benefit'])
        self.assertEqual(len(factor_data), 2)
        self.assertIn('health_concern', factor_data)
        self.assertIn('perceived_benefit', factor_data)
    
    def test_merge_factors(self):
        """요인 병합 테스트"""
        factor_data = self.loader.load_multiple_factors(['health_concern', 'perceived_benefit'])
        merged = self.loader.merge_factors_for_analysis(factor_data)
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertEqual(len(merged), 100)
        # 모든 문항이 포함되어야 함
        expected_columns = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17']
        for col in expected_columns:
            self.assertIn(col, merged.columns)


class TestFactorAnalysisConfig(unittest.TestCase):
    """FactorAnalysisConfig 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = FactorAnalysisConfig()
        self.assertEqual(config.estimator, 'MLW')
        self.assertEqual(config.optimizer, 'SLSQP')
        self.assertTrue(config.standardized)
    
    def test_invalid_estimator(self):
        """잘못된 추정방법 테스트"""
        with self.assertRaises(ValueError):
            FactorAnalysisConfig(estimator='INVALID')
    
    def test_model_spec_creation(self):
        """모델 스펙 생성 테스트"""
        spec = create_factor_model_spec(single_factor='health_concern')
        self.assertIsInstance(spec, str)
        self.assertIn('health_concern =~', spec)
        self.assertIn('q6', spec)


class TestFactorResultsExporter(unittest.TestCase):
    """FactorResultsExporter 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.exporter = FactorResultsExporter(self.test_dir)
        
        # 테스트용 결과 데이터
        self.test_results = {
            'analysis_type': 'single_factor',
            'factor_name': 'health_concern',
            'model_info': {
                'n_observations': 100,
                'n_variables': 6,
                'estimator': 'MLW'
            },
            'factor_loadings': pd.DataFrame({
                'Factor': ['health_concern'] * 6,
                'Item': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
                'Loading': [0.75, 0.68, 0.72, 0.69, 0.71, 0.66],
                'SE': [0.05, 0.06, 0.05, 0.06, 0.05, 0.06],
                'Z_value': [15.0, 11.3, 14.4, 11.5, 14.2, 11.0],
                'P_value': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                'Significant': [True] * 6
            }),
            'fit_indices': {
                'CFI': 0.95,
                'TLI': 0.93,
                'RMSEA': 0.06,
                'SRMR': 0.05
            },
            'standardized_results': pd.DataFrame({
                'Factor': ['health_concern'] * 6,
                'Item': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
                'Std_Loading': [0.75, 0.68, 0.72, 0.69, 0.71, 0.66]
            })
        }
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.test_dir)
    
    def test_export_factor_loadings(self):
        """Factor loadings 내보내기 테스트"""
        file_path = self.exporter.export_factor_loadings(self.test_results)
        self.assertTrue(file_path.exists())
        
        # 저장된 파일 확인
        saved_data = pd.read_csv(file_path)
        self.assertEqual(len(saved_data), 6)
        self.assertIn('Factor', saved_data.columns)
        self.assertIn('Loading', saved_data.columns)
    
    def test_export_fit_indices(self):
        """적합도 지수 내보내기 테스트"""
        file_path = self.exporter.export_fit_indices(self.test_results)
        self.assertTrue(file_path.exists())
        
        # 저장된 파일 확인
        saved_data = pd.read_csv(file_path)
        self.assertEqual(len(saved_data), 4)  # CFI, TLI, RMSEA, SRMR
        self.assertIn('Fit_Index', saved_data.columns)
        self.assertIn('Value', saved_data.columns)
        self.assertIn('Interpretation', saved_data.columns)
    
    def test_comprehensive_export(self):
        """종합 내보내기 테스트"""
        saved_files = self.exporter.export_comprehensive_results(self.test_results)
        
        # 여러 파일이 저장되어야 함
        self.assertGreater(len(saved_files), 1)
        
        # 각 파일이 존재하는지 확인
        for file_type, file_path in saved_files.items():
            self.assertTrue(file_path.exists(), f"{file_type} 파일이 존재하지 않음")


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        """통합 테스트 준비"""
        # 실제 데이터 디렉토리 확인
        self.data_dir = project_root / "processed_data" / "survey_data"
        self.skip_if_no_data()
    
    def skip_if_no_data(self):
        """실제 데이터가 없으면 테스트 건너뛰기"""
        if not self.data_dir.exists():
            self.skipTest("실제 데이터 디렉토리가 없습니다")
        
        # health_concern.csv 파일 확인
        health_file = self.data_dir / "health_concern.csv"
        if not health_file.exists():
            self.skipTest("health_concern.csv 파일이 없습니다")
    
    def test_end_to_end_analysis(self):
        """전체 분석 프로세스 테스트"""
        try:
            # 편의 함수를 사용한 분석
            results = analyze_factor_loading('health_concern', data_dir=self.data_dir)
            
            # 결과 검증
            self.assertIn('factor_loadings', results)
            self.assertIn('model_info', results)
            
            # 결과 내보내기
            output_dir = Path(tempfile.mkdtemp())
            try:
                saved_files = export_factor_results(results, output_dir)
                self.assertGreater(len(saved_files), 0)
            finally:
                shutil.rmtree(output_dir)
                
        except Exception as e:
            # semopy가 설치되지 않은 경우 등을 위한 예외 처리
            if "semopy" in str(e).lower():
                self.skipTest(f"semopy 관련 오류: {e}")
            else:
                raise


def run_tests():
    """테스트 실행"""
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_classes = [
        TestFactorDataLoader,
        TestFactorAnalysisConfig, 
        TestFactorResultsExporter,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
