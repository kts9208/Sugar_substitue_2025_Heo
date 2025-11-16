"""
순차추정 단계별 실행 테스트

1단계와 2단계를 분리하여 실행하는 기능을 테스트합니다.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.analysis.hybrid_choice_model.iclv_models.sequential_estimator import SequentialEstimator
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_model import MultinomialLogitChoice
from src.analysis.hybrid_choice_model.config import MultiLatentConfig


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)
    n_individuals = 50
    n_choice_sets = 3
    n_alternatives = 3
    
    data_list = []
    
    for i in range(n_individuals):
        # 잠재변수 지표 (개인별 동일)
        indicators = {
            'HC1': np.random.randint(1, 6),
            'HC2': np.random.randint(1, 6),
            'HC3': np.random.randint(1, 6),
            'PB1': np.random.randint(1, 6),
            'PB2': np.random.randint(1, 6),
            'PB3': np.random.randint(1, 6),
            'PI1': np.random.randint(1, 6),
            'PI2': np.random.randint(1, 6),
            'PI3': np.random.randint(1, 6),
        }
        
        for cs in range(n_choice_sets):
            for alt in range(n_alternatives):
                row = {
                    'respondent_id': i,
                    'choice_set': cs,
                    'alternative': alt,
                    'price': np.random.uniform(1000, 5000),
                    'sugar_content': np.random.uniform(0, 10),
                    'brand': np.random.randint(0, 3),
                    'choice': 1 if alt == 0 else 0,  # 첫 번째 대안 선택
                    **indicators
                }
                data_list.append(row)
    
    return pd.DataFrame(data_list)


@pytest.fixture
def config():
    """테스트용 설정"""
    return MultiLatentConfig(
        latent_variables={
            'health_concern': ['HC1', 'HC2', 'HC3'],
            'perceived_benefit': ['PB1', 'PB2', 'PB3'],
            'purchase_intention': ['PI1', 'PI2', 'PI3'],
        },
        structural_paths={
            'health_concern': [],
            'perceived_benefit': ['health_concern'],
            'purchase_intention': ['perceived_benefit'],
        },
        choice_attributes=['price', 'sugar_content'],
        choice_column='choice',
        individual_id_column='respondent_id'
    )


def test_stage1_only(sample_data, config):
    """1단계만 실행 테스트"""
    # 모델 생성
    measurement_model = MultiLatentMeasurement(config)
    structural_model = MultiLatentStructural(config)
    estimator = SequentialEstimator(config)
    
    # 1단계 실행
    results = estimator.estimate_stage1_only(
        data=sample_data,
        measurement_model=measurement_model,
        structural_model=structural_model
    )
    
    # 결과 검증
    assert 'factor_scores' in results
    assert 'paths' in results
    assert 'loadings' in results
    assert 'fit_indices' in results
    assert 'log_likelihood' in results
    
    # 요인점수 검증
    assert len(results['factor_scores']) == 3  # 3개 잠재변수
    for lv_name, scores in results['factor_scores'].items():
        assert len(scores) == 50  # 50명
        # 표준화 확인 (평균 ≈ 0, 표준편차 ≈ 1)
        assert abs(np.mean(scores)) < 0.1
        assert abs(np.std(scores) - 1.0) < 0.1


def test_stage2_only_with_dict(sample_data, config):
    """2단계만 실행 테스트 (딕셔너리 전달)"""
    # 1단계 실행
    measurement_model = MultiLatentMeasurement(config)
    structural_model = MultiLatentStructural(config)
    estimator = SequentialEstimator(config)
    
    stage1_results = estimator.estimate_stage1_only(
        data=sample_data,
        measurement_model=measurement_model,
        structural_model=structural_model
    )
    
    # 2단계 실행 (딕셔너리 전달)
    choice_model = MultinomialLogitChoice(
        choice_attributes=config.choice_attributes,
        latent_variable='purchase_intention',
        choice_column=config.choice_column,
        individual_id_column=config.individual_id_column
    )
    
    stage2_results = estimator.estimate_stage2_only(
        data=sample_data,
        choice_model=choice_model,
        factor_scores=stage1_results['factor_scores']
    )
    
    # 결과 검증
    assert 'params' in stage2_results
    assert 'log_likelihood' in stage2_results
    assert stage2_results['success']


def test_stage2_only_with_file(sample_data, config):
    """2단계만 실행 테스트 (파일 경로 전달)"""
    # 1단계 실행 및 저장
    measurement_model = MultiLatentMeasurement(config)
    structural_model = MultiLatentStructural(config)
    estimator = SequentialEstimator(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "stage1_results.pkl"
        
        stage1_results = estimator.estimate_stage1_only(
            data=sample_data,
            measurement_model=measurement_model,
            structural_model=structural_model,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        
        # 2단계 실행 (파일 경로 전달)
        choice_model = MultinomialLogitChoice(
            choice_attributes=config.choice_attributes,
            latent_variable='purchase_intention',
            choice_column=config.choice_column,
            individual_id_column=config.individual_id_column
        )
        
        stage2_results = estimator.estimate_stage2_only(
            data=sample_data,
            choice_model=choice_model,
            factor_scores=str(save_path)
        )
        
        # 결과 검증
        assert 'params' in stage2_results
        assert 'log_likelihood' in stage2_results


def test_save_load_stage1_results(sample_data, config):
    """1단계 결과 저장/로드 테스트"""
    measurement_model = MultiLatentMeasurement(config)
    structural_model = MultiLatentStructural(config)
    estimator = SequentialEstimator(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "stage1_results.pkl"
        
        # 1단계 실행 및 저장
        original_results = estimator.estimate_stage1_only(
            data=sample_data,
            measurement_model=measurement_model,
            structural_model=structural_model,
            save_path=str(save_path)
        )
        
        # 로드
        loaded_results = SequentialEstimator.load_stage1_results(str(save_path))
        
        # 검증
        assert set(loaded_results['factor_scores'].keys()) == set(original_results['factor_scores'].keys())
        for lv_name in loaded_results['factor_scores'].keys():
            np.testing.assert_array_almost_equal(
                loaded_results['factor_scores'][lv_name],
                original_results['factor_scores'][lv_name]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

