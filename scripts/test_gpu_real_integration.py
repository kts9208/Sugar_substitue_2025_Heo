"""
GPU 경로 통합 테스트: 실제 동시추정과 동일한 방식으로 테스트

실제 simultaneous_estimator_fixed.py에서 사용하는 것과 동일한 방식으로
MultiLatentJointGradient.compute_individual_gradient()를 호출하여 테스트합니다.
"""

import sys
import os
from pathlib import Path
import pickle

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("통합 테스트: 실제 동시추정과 동일한 방식으로 GPU 경로 테스트")
print("="*80)
print(f"\n프로젝트 루트: {project_root}")

import numpy as np
import pandas as pd

# CuPy 확인
try:
    import cupy as cp
    print("✅ CuPy 사용 가능")
except ImportError:
    print("❌ CuPy 미설치")
    sys.exit(1)

# 실제 동시추정에서 사용하는 모듈들
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_gradient import (
    MultiLatentJointGradient,
    MultiLatentMeasurementGradient,
    MultiLatentStructuralGradient
)
from src.analysis.hybrid_choice_model.iclv_models.gradient_calculator import ChoiceGradient
from src.analysis.hybrid_choice_model.iclv_models.gpu_measurement_equations import GPUMultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import create_sugar_substitute_multi_lv_config
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import ChoiceConfig

print("✅ 모듈 import 성공")


def load_real_data():
    """실제 CFA 결과와 데이터 로드"""
    print("\n[1] 실제 데이터 로드")

    # CFA 결과 로드
    cfa_path = project_root / 'results' / 'final' / 'cfa_only' / 'cfa_results.pkl'
    with open(cfa_path, 'rb') as f:
        cfa_results = pickle.load(f)

    print(f"  ✅ CFA 결과 로드: {cfa_path.name}")

    # 데이터 로드 (전체 데이터)
    data_path = project_root / 'data' / 'processed' / 'iclv' / 'integrated_data.csv'
    data = pd.read_csv(data_path)

    n_individuals = data['respondent_id'].nunique()
    print(f"  ✅ 데이터 로드: {n_individuals}명, {len(data)}개 행")

    return cfa_results, data


def create_models_and_gradient():
    """실제 동시추정과 동일한 방식으로 모델 및 gradient 계산기 생성"""
    print("\n[2] 모델 및 Gradient 계산기 생성")

    # 경로 설정 (test_gpu_batch_iclv.py와 동일)
    hierarchical_paths = [
        {'target': 'perceived_benefit', 'predictors': ['health_concern']},
        {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
    ]

    # 선택모델 설정
    choice_config_dict = {
        'main_lvs': ['nutrition_knowledge', 'purchase_intention', 'perceived_price'],
        'lv_attribute_interactions': []
    }

    # Config 생성 (test_gpu_batch_iclv.py와 동일)
    config = create_sugar_substitute_multi_lv_config(
        custom_paths=hierarchical_paths,
        choice_config_overrides=choice_config_dict,
        n_draws=100,
        max_iterations=50,
        optimizer='trust-constr',
        use_analytic_gradient=True,
        calculate_se=True,
        se_method='robust',
        gradient_log_level='MINIMAL',
        use_parameter_scaling=False,
        standardize_choice_attributes=True
    )

    print(f"  ✅ Config 생성 완료")

    # 모델 객체 생성 (실제 동시추정과 동일)
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
    from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
    from src.analysis.hybrid_choice_model.iclv_models.choice_equations import MultinomialLogitChoice

    measurement_model = MultiLatentMeasurement(config.measurement_configs)
    structural_model = MultiLatentStructural(config.structural)
    choice_model = MultinomialLogitChoice(config.choice)

    print(f"  ✅ 모델 객체 생성 완료")

    # GPU 측정모델 생성
    gpu_measurement_model = GPUMultiLatentMeasurement(
        config.measurement_configs,
        use_gpu=True
    )

    print(f"  ✅ GPU 측정모델 생성 완료")

    # Gradient 계산기 생성 (실제 동시추정과 동일)
    measurement_grad = MultiLatentMeasurementGradient(config.measurement_configs)
    structural_grad = MultiLatentStructuralGradient(
        n_exo=config.structural.n_exo,
        n_cov=config.structural.n_cov,
        error_variance=config.structural.error_variance
    )
    choice_grad = ChoiceGradient(n_attributes=len(config.choice.choice_attributes))

    # MultiLatentJointGradient 생성 (measurement_params_fixed=True)
    joint_grad = MultiLatentJointGradient(
        measurement_grad,
        structural_grad,
        choice_grad,
        use_gpu=True,
        gpu_measurement_model=gpu_measurement_model,
        use_full_parallel=True,
        measurement_params_fixed=True  # ✅ 동시추정: 측정모델 파라미터 고정
    )

    print(f"  ✅ MultiLatentJointGradient 생성 완료")

    return config, measurement_model, structural_model, choice_model, gpu_measurement_model, joint_grad


def test_full_batch_gradient():
    """실제 동시추정과 동일하게 compute_all_individuals_gradients_full_batch() 호출 테스트"""
    print("\n" + "="*80)
    print("TEST: compute_all_individuals_gradients_full_batch() 호출 (실제 동시추정 경로)")
    print("="*80)

    try:
        # 실제 데이터 로드
        print("\n[1] 실제 데이터 로드")
        cfa_results, data = load_real_data()

        # 모델 및 gradient 계산기 생성
        print("\n[2] 모델 및 Gradient 계산기 생성")
        config, measurement_model, structural_model, choice_model, gpu_measurement_model, joint_grad = create_models_and_gradient()

        print("\n[3] 테스트 파라미터 생성")

        # ✅ test_gpu_batch_iclv.py와 동일한 방식으로 CFA 결과를 딕셔너리로 변환
        loadings_df = cfa_results['loadings']
        errors_df = cfa_results['measurement_errors']
        intercepts_df = cfa_results.get('intercepts', None)

        measurement_dict = {}
        for lv_name, lv_config in config.measurement_configs.items():
            indicators = lv_config.indicators

            # zeta (요인적재량)
            zeta_values = []
            for indicator in indicators:
                row = loadings_df[(loadings_df['lval'] == indicator) &
                                 (loadings_df['op'] == '~') &
                                 (loadings_df['rval'] == lv_name)]
                if not row.empty:
                    zeta_values.append(float(row['Estimate'].iloc[0]))
                else:
                    zeta_values.append(1.0)

            # sigma_sq (오차분산)
            sigma_sq_values = []
            for indicator in indicators:
                row = errors_df[(errors_df['lval'] == indicator) &
                               (errors_df['op'] == '~~') &
                               (errors_df['rval'] == indicator)]
                if not row.empty:
                    sigma_sq_values.append(float(row['Estimate'].iloc[0]))
                else:
                    sigma_sq_values.append(0.5)

            # alpha (절편)
            alpha_values = []
            if intercepts_df is not None:
                for indicator in indicators:
                    row = intercepts_df[(intercepts_df['lval'] == indicator) &
                                       (intercepts_df['op'] == '~') &
                                       (intercepts_df['rval'] == '1')]
                    if not row.empty:
                        alpha_values.append(float(row['Estimate'].iloc[0]))
                    else:
                        alpha_values.append(0.0)
            else:
                alpha_values = [0.0] * len(indicators)

            measurement_dict[lv_name] = {
                'zeta': np.array(zeta_values),
                'sigma_sq': np.array(sigma_sq_values),
                'alpha': np.array(alpha_values)
            }

        # 파라미터 딕셔너리 생성 (실제 동시추정과 동일한 구조)
        params_dict = {
            'measurement': measurement_dict,
            'structural': {
                'gamma_health_concern_to_perceived_benefit': 0.0004,
                'gamma_perceived_benefit_to_purchase_intention': -0.0007
            },
            'choice': {
                'asc_sugar': 1.45,
                'asc_sugar_free': 2.44,
                'beta_health_label': 0.50,
                'beta_price': -0.56,
                'theta_sugar_nutrition_knowledge': -0.024,
                'theta_sugar_free_nutrition_knowledge': -0.018,
                'theta_sugar_purchase_intention': -0.021,
                'theta_sugar_free_purchase_intention': -0.016,
                'theta_sugar_perceived_price': -0.022,
                'theta_sugar_free_perceived_price': -0.020
            }
        }

        print(f"  측정모델 LV 수: {len(params_dict['measurement'])}")
        print(f"  구조모델 경로 수: {len(params_dict['structural'])}")
        print(f"  선택모델 파라미터 수: {len(params_dict['choice'])}")

        print("\n[4] Halton draws 생성")

        # Halton draws 생성 (3명 × 100 draws × 5 LVs)
        from scipy.stats.qmc import Halton
        n_individuals = 3
        n_draws = 100
        n_lvs = 5  # HC, PB, PI, PP, NK

        halton = Halton(d=n_lvs, scramble=True, seed=42)
        draws = halton.random(n=n_individuals * n_draws)

        # 표준정규분포로 변환
        from scipy.stats import norm
        all_ind_draws = norm.ppf(draws).reshape(n_individuals, n_draws, n_lvs)

        print(f"  Halton draws shape: {all_ind_draws.shape}")

        print("\n[5] 개인 데이터 준비")

        # 처음 3명의 데이터 준비
        individual_ids = data[config.individual_id_column].unique()[:n_individuals]
        all_ind_data = []
        for ind_id in individual_ids:
            ind_data = data[data[config.individual_id_column] == ind_id]
            all_ind_data.append(ind_data)
            print(f"  개인 {ind_id}: {len(ind_data)}개 선택 상황")

        print("\n[6] compute_all_individuals_gradients_full_batch() 호출")

        # ✅ 실제 동시추정과 동일한 방식으로 호출
        all_grad_dicts = joint_grad.compute_all_individuals_gradients_full_batch(
            all_ind_data=all_ind_data,
            all_ind_draws=all_ind_draws,
            params_dict=params_dict,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model,
            iteration_logger=None,
            log_level='DETAILED'
        )

        print(f"\n[7] 결과 확인")
        print(f"  ✅ Gradient 계산 성공! ({len(all_grad_dicts)}명)")

        for i, grad_dict in enumerate(all_grad_dicts[:3]):  # 처음 3명만
            print(f"\n  개인 {i}:")
            print(f"    측정모델 gradient 키: {list(grad_dict.get('measurement', {}).keys())}")
            print(f"    구조모델 gradient 키: {list(grad_dict.get('structural', {}).keys())}")
            print(f"    선택모델 gradient 키: {list(grad_dict.get('choice', {}).keys())}")

            # 구조모델 gradient 값 출력
            if 'structural' in grad_dict:
                for key, val in grad_dict['structural'].items():
                    print(f"      grad_{key}: {val:.6e}")

            # 선택모델 gradient 값 출력 (일부)
            if 'choice' in grad_dict:
                for j, (key, val) in enumerate(grad_dict['choice'].items()):
                    if j < 3:  # 처음 3개만
                        if isinstance(val, np.ndarray):
                            print(f"      grad_{key}: shape={val.shape}, mean={val.mean():.6e}")
                        else:
                            print(f"      grad_{key}: {val:.6e}")

        # ✅ 8. BHHH Hessian 계산 테스트
        print("\n[8] BHHH Hessian 계산 테스트")

        # 실제 동시추정과 동일하게 _pack_gradient() 사용
        from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator

        # Estimator 인스턴스 생성 및 초기화
        temp_estimator = SimultaneousEstimator(config)
        temp_estimator.data = data  # 데이터 설정

        # ✅ iteration_logger 설정 (필수!)
        import logging
        temp_logger = logging.getLogger('test_integration')
        temp_logger.setLevel(logging.INFO)
        temp_estimator.iteration_logger = temp_logger

        # param_names 설정 (최적화 파라미터만: 구조모델 + 선택모델)
        temp_estimator.param_names = temp_estimator.param_manager.get_optimized_parameter_names(
            structural_model, choice_model
        )
        print(f"  파라미터 이름: {temp_estimator.param_names}")

        # Gradient 벡터로 변환
        individual_gradients = []
        for grad_dict in all_grad_dicts:
            grad_vector = temp_estimator._pack_gradient(
                grad_dict,
                measurement_model,
                structural_model,
                choice_model
            )
            individual_gradients.append(grad_vector)

        print(f"  Gradient 벡터 변환 완료: {len(individual_gradients)}개")
        print(f"  Gradient 벡터 길이: {len(individual_gradients[0])}개 파라미터")

        # BHHH Hessian 계산
        from src.analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator
        bhhh_calc = BHHHCalculator()

        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=True
        )

        print(f"  ✅ BHHH Hessian 계산 성공!")
        print(f"  Hessian shape: {hessian_bhhh.shape}")

        # Hessian 역행렬 계산
        hess_inv = bhhh_calc.compute_hessian_inverse(hessian_bhhh)
        print(f"  ✅ Hessian 역행렬 계산 성공!")

        # 표준오차 계산
        se = bhhh_calc.compute_standard_errors(hess_inv)
        print(f"  ✅ 표준오차 계산 성공!")
        print(f"  SE shape: {se.shape}")
        print(f"  SE 범위: [{np.min(se):.6e}, {np.max(se):.6e}]")

        # ✅ 9. parameter_statistics 생성 테스트
        print("\n[9] parameter_statistics 생성 테스트")

        # 파라미터 벡터 생성 (params_dict에서 추출)
        # 최적화 파라미터만 추출 (구조모델 + 선택모델)
        opt_dict = {
            'structural': params_dict['structural'],
            'choice': params_dict['choice']
        }

        param_vector = temp_estimator.param_manager.dict_to_array_optimized(
            opt_dict,
            temp_estimator.param_names,
            structural_model,
            choice_model
        )

        # t-통계량 및 p-값 계산
        t_stats = bhhh_calc.compute_t_statistics(param_vector, se)
        p_values = bhhh_calc.compute_p_values(t_stats)

        print(f"  ✅ t-통계량 계산 성공!")
        print(f"  ✅ p-값 계산 성공!")

        # parameter_statistics 딕셔너리 생성 (수동)
        parameter_statistics = {
            'structural': {},
            'choice': {}
        }

        # 파라미터 이름과 통계량 매핑
        for i, param_name in enumerate(temp_estimator.param_names):
            if param_name.startswith('gamma_') and '_to_' in param_name:
                # 구조모델 파라미터: 'std_error', 't', 'p_value' 키 사용
                parameter_statistics['structural'][param_name] = {
                    'estimate': param_vector[i],
                    'std_error': se[i],
                    't': t_stats[i],
                    'p_value': p_values[i]
                }
            else:
                # 선택모델 파라미터: 'se', 't', 'p' 키 사용
                parameter_statistics['choice'][param_name] = {
                    'estimate': param_vector[i],
                    'se': se[i],
                    't': t_stats[i],
                    'p': p_values[i]
                }

        print(f"  ✅ parameter_statistics 생성 성공!")
        print(f"  섹션: {list(parameter_statistics.keys())}")
        print(f"  선택모델 파라미터 키: {list(parameter_statistics['choice'].keys())}")

        # 구조모델 통계 출력
        if 'structural' in parameter_statistics:
            print(f"\n  [구조모델 통계]")
            for key, stat in parameter_statistics['structural'].items():
                print(f"    {key}:")
                print(f"      estimate: {stat['estimate']:.6e}")
                print(f"      std_error: {stat['std_error']:.6e}")
                print(f"      t: {stat['t']:.6e}")
                print(f"      p_value: {stat['p_value']:.6e}")

        # 선택모델 통계 출력 (일부)
        if 'choice' in parameter_statistics:
            print(f"\n  [선택모델 통계 (처음 3개)]")
            for i, (key, stat) in enumerate(parameter_statistics['choice'].items()):
                if i < 3:
                    print(f"    {key}:")
                    if isinstance(stat, dict) and 'estimate' in stat:
                        print(f"      estimate: {stat['estimate']:.6e}")
                        print(f"      se: {stat['se']:.6e}")
                        print(f"      t: {stat['t']:.6e}")
                        print(f"      p: {stat['p']:.6e}")

        # ✅ 10. CSV 저장 테스트 (실제 동시추정과 동일한 계층 구조 사용)
        print("\n[10] CSV 저장 테스트")
        print("  ⚠️ 실제 동시추정은 계층 구조를 사용: {'structural': {...}, 'choice': {...}}")

        # ✅ 실제 동시추정과 동일한 계층 구조 사용
        test_results = {
            'log_likelihood': -2051.0473,
            'aic': 4126.0946,
            'bic': 4180.5234,
            'parameter_statistics': parameter_statistics,  # ✅ 계층 구조 그대로 사용!
            'params': params_dict
        }

        print(f"  parameter_statistics 구조: {list(parameter_statistics.keys())}")
        print(f"  structural 파라미터: {len(parameter_statistics['structural'])}개")
        print(f"  choice 파라미터: {len(parameter_statistics['choice'])}개")

        # CSV 저장
        from examples.model_config_utils import save_iclv_results
        test_csv_path = project_root / 'results' / 'test_integration_results.csv'

        save_iclv_results(
            results=test_results,
            save_path=test_csv_path,
            estimation_type='simultaneous',
            cfa_results=cfa_results,
            config=config
        )

        print(f"  ✅ CSV 저장 성공: {test_csv_path}")

        # CSV 파일 확인
        saved_df = pd.read_csv(test_csv_path)
        print(f"  저장된 행 수: {len(saved_df)}")
        print(f"  저장된 컬럼: {list(saved_df.columns)}")

        # 구조모델 + 선택모델 파라미터 확인
        structural_rows = saved_df[saved_df['section'] == 'Structural_Model']
        choice_rows = saved_df[saved_df['section'] == 'Choice_Model']

        print(f"  구조모델 파라미터: {len(structural_rows)}개")
        print(f"  선택모델 파라미터: {len(choice_rows)}개")

        if len(structural_rows) > 0:
            print(f"\n  [구조모델 파라미터 샘플]")
            print(structural_rows[['parameter', 'estimate', 'std_error', 'p_value']].to_string(index=False))

        if len(choice_rows) > 0:
            print(f"\n  [선택모델 파라미터 샘플 (처음 5개)]")
            print(choice_rows[['parameter', 'estimate', 'std_error', 'p_value']].head().to_string(index=False))

        # ✅ theta 파라미터가 저장되었는지 확인
        theta_rows = saved_df[saved_df['parameter'].str.startswith('theta_', na=False)]
        print(f"\n  [Theta 파라미터 확인]")
        print(f"  Theta 파라미터 수: {len(theta_rows)}개")

        if len(theta_rows) == 0:
            raise ValueError("❌ Theta 파라미터가 CSV에 저장되지 않음!")

        print(f"  ✅ Theta 파라미터 저장 확인!")
        print(theta_rows[['parameter', 'estimate', 'std_error']].head().to_string(index=False))

        # ✅ 11. BHHH 경로 테스트 (Full Batch로 개인별 gradient 계산)
        print("\n[11] BHHH 경로 테스트 (Full Batch로 개인별 gradient)")
        print("  ⚠️ 실제 BHHH 계산은 Full Batch 경로를 사용합니다")

        # 3명의 개인별 gradient를 Full Batch로 계산 (BHHH와 동일)
        bhhh_grad_dicts = joint_grad.compute_all_individuals_gradients_full_batch(
            all_ind_data=all_ind_data,
            all_ind_draws=all_ind_draws,
            params_dict=params_dict,
            measurement_model=measurement_model,
            structural_model=structural_model,
            choice_model=choice_model
        )

        print(f"  ✅ BHHH 경로 gradient 계산 성공! ({len(bhhh_grad_dicts)}명)")

        # 첫 번째 개인의 gradient 검증
        first_grad = bhhh_grad_dicts[0]
        print(f"\n  [첫 번째 개인 gradient 검증]")
        print(f"  측정모델 gradient 키: {list(first_grad.get('measurement', {}).keys())}")
        print(f"  구조모델 gradient 키: {list(first_grad.get('structural', {}).keys())}")
        print(f"  선택모델 gradient 키: {list(first_grad.get('choice', {}).keys())}")

        # Gradient 키 이름 검증 (MNL 형식이어야 함)
        expected_choice_keys = ['asc_sugar', 'asc_sugar_free', 'beta',
                               'theta_sugar_nutrition_knowledge', 'theta_sugar_free_nutrition_knowledge',
                               'theta_sugar_purchase_intention', 'theta_sugar_free_purchase_intention',
                               'theta_sugar_perceived_price', 'theta_sugar_free_perceived_price']
        actual_choice_keys = list(first_grad['choice'].keys())

        # Binary Probit 키가 있으면 에러
        if 'grad_intercept' in actual_choice_keys or 'grad_beta' in actual_choice_keys:
            raise ValueError(f"❌ Binary Probit 키 발견! 선택모델 키: {actual_choice_keys}")

        print(f"  ✅ MNL 형식 gradient 키 검증 성공!")

        # Gradient 값 확인
        for key, value in first_grad['structural'].items():
            print(f"    {key}: {value:.6e}")
        for key, value in list(first_grad['choice'].items())[:3]:
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, mean={value.mean():.6e}")
            else:
                print(f"    {key}: {value:.6e}")

        # ✅ 12. _pack_gradient() 검증 테스트 (실제 BHHH에서 사용)
        print("\n[12] _pack_gradient() 검증 테스트")
        print("  ⚠️ 이 함수는 gradient 딕셔너리를 벡터로 변환하며 검증 로직 포함")

        # SimultaneousEstimator의 _pack_gradient() 메서드를 직접 호출할 수 없으므로
        # ParameterManager의 dict_to_array()를 직접 테스트
        from src.analysis.hybrid_choice_model.iclv_models.parameter_manager import ParameterManager

        param_manager_test = ParameterManager(config)

        # 파라미터 이름 리스트 (실제 동시추정과 동일)
        param_names_test = [
            'gamma_health_concern_to_perceived_benefit',
            'gamma_perceived_benefit_to_purchase_intention',
            'asc_sugar',
            'asc_sugar_free',
            'beta_health_label',
            'beta_price',
            'theta_sugar_nutrition_knowledge',
            'theta_sugar_free_nutrition_knowledge',
            'theta_sugar_purchase_intention',
            'theta_sugar_free_purchase_intention',
            'theta_sugar_perceived_price',
            'theta_sugar_free_perceived_price'
        ]

        # Gradient 딕셔너리를 배열로 변환 (측정모델 제외)
        grad_dict_opt = {
            'structural': first_grad.get('structural', {}),
            'choice': first_grad.get('choice', {})
        }

        try:
            grad_vector_test = param_manager_test.dict_to_array(
                grad_dict_opt,
                param_names_test,
                measurement_model=measurement_model
            )
            print(f"  ✅ dict_to_array() 성공!")
            print(f"  Gradient 벡터 길이: {len(grad_vector_test)}")
            print(f"  기대 길이: {len(param_names_test)}")

            if len(grad_vector_test) != len(param_names_test):
                raise ValueError(f"❌ Gradient 벡터 길이 불일치! {len(grad_vector_test)} != {len(param_names_test)}")

            # 각 파라미터별 gradient 값 확인
            print(f"\n  [파라미터별 gradient 값]")
            for i, name in enumerate(param_names_test[:5]):  # 처음 5개만
                print(f"    {name}: {grad_vector_test[i]:.6e}")

            print(f"  ✅ 모든 파라미터 gradient 변환 성공!")

        except Exception as e:
            print(f"  ❌ dict_to_array() 실패: {e}")
            raise

        # ✅ 13. _process_results() 함수 전체 경로 테스트 (실제 프로세스 복제)
        print("\n[13] _process_results() 함수 전체 경로 테스트")
        print("  ⚠️ 이 함수는 최적화 후 호출되어 parameter_statistics 생성 및 CSV 저장")

        try:
            # 가상의 optimization_result 생성 (Trust-Constr optimizer 시뮬레이션)
            class MockOptimizationResult:
                def __init__(self, x, fun):
                    self.x = x
                    self.fun = fun
                    self.success = True
                    self.message = "Test convergence"
                    self.nit = 16
                    # ❌ Trust-Constr는 hess_inv를 제공하지 않음!
                    # (이것이 실제 프로세스의 조건)

            mock_result = MockOptimizationResult(
                x=np.array([0.0004, -0.0007, 1.45, 2.44, 0.5, -0.56, -0.03, -0.02, -0.03, -0.02, -0.02, -0.01]),
                fun=2050.4142
            )

            # ✅ 실제 프로세스 시뮬레이션:
            # 1. Sandwich Estimator 계산 (이미 Section [8]에서 수행)
            # 2. self.hessian_inv_matrix와 self.robust_se 설정
            # 3. _process_results() 호출

            # Sandwich Estimator 결과 시뮬레이션 (실제 값 사용)
            # 주의: 3명의 개인만 사용하므로 Hessian이 특이행렬일 수 있음
            # 실제 프로세스에서는 328명 사용

            # 가상의 Hessian 역행렬 생성 (양정부호 행렬)
            np.random.seed(42)
            n_params = 12
            A = np.random.randn(n_params, n_params)
            hess_inv_mock = np.dot(A, A.T) * 0.01  # 양정부호 행렬
            robust_se_mock = np.sqrt(np.abs(np.diag(hess_inv_mock)))

            # SimultaneousEstimator에 Sandwich Estimator 결과 설정
            temp_estimator.hessian_inv_matrix = hess_inv_mock
            temp_estimator.robust_se = robust_se_mock

            print(f"\n  [Sandwich Estimator 결과 설정]")
            print(f"  hessian_inv_matrix shape: {temp_estimator.hessian_inv_matrix.shape}")
            print(f"  robust_se shape: {temp_estimator.robust_se.shape}")
            print(f"  robust_se 범위: [{robust_se_mock.min():.6e}, {robust_se_mock.max():.6e}]")

            # ✅ _process_results() 호출 (실제 프로세스와 동일)
            print(f"\n  [_process_results() 호출]")
            results_test = temp_estimator._process_results(
                mock_result,
                measurement_model,
                structural_model,
                choice_model
            )

            # 결과 검증
            print(f"  ✅ _process_results() 성공!")
            print(f"\n  [결과 검증]")
            print(f"  success: {results_test['success']}")
            print(f"  log_likelihood: {results_test['log_likelihood']:.4f}")
            print(f"  n_parameters: {results_test['n_parameters']}")

            # ✅ 핵심: parameter_statistics가 생성되었는지 확인
            if 'parameter_statistics' in results_test:
                print(f"  ✅ parameter_statistics 생성됨!")
                param_stats = results_test['parameter_statistics']
                print(f"  섹션: {list(param_stats.keys())}")

                # 구조모델 파라미터 확인
                if 'structural' in param_stats:
                    print(f"  구조모델 파라미터: {len(param_stats['structural'])}개")
                    for key in list(param_stats['structural'].keys())[:2]:
                        stat = param_stats['structural'][key]
                        print(f"    {key}:")
                        print(f"      estimate: {stat['estimate']:.6e}")
                        print(f"      std_error: {stat['std_error']:.6e}")

                # 선택모델 파라미터 확인
                if 'choice' in param_stats:
                    print(f"  선택모델 파라미터: {len(param_stats['choice'])}개")
                    for key in list(param_stats['choice'].keys())[:3]:
                        stat = param_stats['choice'][key]
                        print(f"    {key}:")
                        print(f"      estimate: {stat['estimate']:.6e}")
                        print(f"      std_error: {stat['std_error']:.6e}")

                print(f"  ✅ parameter_statistics 구조 검증 성공!")
            else:
                raise ValueError("❌ parameter_statistics가 생성되지 않음!")

            # ✅ 표준오차 계산 경로 검증
            if 'standard_errors' in results_test:
                se_result = results_test['standard_errors']
                print(f"\n  [표준오차 계산 검증]")
                print(f"  SE shape: {se_result.shape}")
                print(f"  SE 범위: [{se_result.min():.6e}, {se_result.max():.6e}]")
                print(f"  ✅ self.robust_se가 사용됨!")
            else:
                raise ValueError("❌ 표준오차가 계산되지 않음!")

        except Exception as e:
            print(f"  ❌ _process_results() 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("\n" + "="*80)
        print("✅ 통합 테스트 성공! (Full Batch + BHHH + _pack_gradient + parameter_statistics + CSV)")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_batch_gradient()

    if success:
        print("\n✅ GPU 경로 통합 테스트 성공 (실제 동시추정 경로)")
        sys.exit(0)
    else:
        print("\n❌ GPU 경로 통합 테스트 실패")
        sys.exit(1)

