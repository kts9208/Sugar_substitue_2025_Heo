"""
단순 Binary Probit 추정으로 선택모델 초기값 획득

목적:
1. 측정모델 추정 (CFA) → 요인점수 획득
2. 요인점수를 사용한 Binary Probit 추정
3. 추정값을 동시추정 초기값으로 사용

작성일: 2025-11-13
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_measurement_model_cfa(data: pd.DataFrame) -> pd.DataFrame:
    """
    측정모델 추정 (간단한 CFA)
    
    실제로는 factor_analyzer 또는 semopy 사용 권장
    여기서는 단순 평균으로 요인점수 계산
    
    Returns:
        요인점수 DataFrame (개인별)
    """
    logger.info("=" * 80)
    logger.info("Step 1: 측정모델 추정 (CFA)")
    logger.info("=" * 80)
    
    # 개인별 첫 번째 행만 추출 (측정모델 지표는 개인 수준)
    individual_data = data.groupby('respondent_id').first().reset_index()
    
    # 요인점수 계산 (단순 평균)
    factor_scores = pd.DataFrame()
    factor_scores['respondent_id'] = individual_data['respondent_id']
    
    # Health Concern (q6-q11)
    hc_items = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    factor_scores['health_concern'] = individual_data[hc_items].mean(axis=1)
    
    # Perceived Benefit (q12-q17)
    pb_items = ['q12', 'q13', 'q14', 'q15', 'q16', 'q17']
    factor_scores['perceived_benefit'] = individual_data[pb_items].mean(axis=1)
    
    # Perceived Price (q27-q29)
    pp_items = ['q27', 'q28', 'q29']
    factor_scores['perceived_price'] = individual_data[pp_items].mean(axis=1)
    
    # Nutrition Knowledge (q30-q49)
    nk_items = [f'q{i}' for i in range(30, 50)]
    factor_scores['nutrition_knowledge'] = individual_data[nk_items].mean(axis=1)
    
    # Purchase Intention (q18-q20)
    pi_items = ['q18', 'q19', 'q20']
    factor_scores['purchase_intention'] = individual_data[pi_items].mean(axis=1)
    
    # 표준화 (평균 0, 분산 1)
    for col in ['health_concern', 'perceived_benefit', 'perceived_price', 
                'nutrition_knowledge', 'purchase_intention']:
        factor_scores[col] = (factor_scores[col] - factor_scores[col].mean()) / factor_scores[col].std()
    
    logger.info(f"요인점수 계산 완료: {len(factor_scores)}명")
    logger.info(f"요인점수 기술통계:")
    logger.info(f"\n{factor_scores.describe()}")
    
    return factor_scores


def prepare_probit_data(data: pd.DataFrame, factor_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Binary Probit 추정용 데이터 준비
    
    DCE 데이터 + 요인점수 병합
    """
    logger.info("=" * 80)
    logger.info("Step 2: Probit 추정용 데이터 준비")
    logger.info("=" * 80)
    
    # DCE 데이터에 요인점수 병합
    probit_data = data.merge(factor_scores, on='respondent_id', how='left')
    
    # 선택 변수 생성 (chosen == 1)
    probit_data['choice'] = (probit_data['chosen'] == 1).astype(int)
    
    # 속성 변수
    probit_data['sugar_free'] = probit_data['sugar_free'].astype(float)
    probit_data['health_label'] = probit_data['has_health_label'].astype(float)
    probit_data['price'] = probit_data['price_scaled'].astype(float)
    
    # 조절효과 변수 생성
    probit_data['PI_x_PP'] = probit_data['purchase_intention'] * probit_data['perceived_price']
    probit_data['PI_x_NK'] = probit_data['purchase_intention'] * probit_data['nutrition_knowledge']
    
    logger.info(f"Probit 데이터 준비 완료: {len(probit_data)}행")
    logger.info(f"선택 비율: {probit_data['choice'].mean():.3f}")
    
    return probit_data


def binary_probit_log_likelihood(params: np.ndarray, data: pd.DataFrame) -> float:
    """
    Binary Probit 음의 로그우도
    
    모델:
        V = intercept + β_sf*sugar_free + β_hl*health_label + β_p*price 
            + λ_main*PI + λ_PP*(PI×PP) + λ_NK*(PI×NK)
        P(Yes) = Φ(V)
    
    파라미터 순서:
        [0] intercept
        [1] beta_sugar_free
        [2] beta_health_label
        [3] beta_price
        [4] lambda_main
        [5] lambda_mod_perceived_price
        [6] lambda_mod_nutrition_knowledge
    """
    intercept = params[0]
    beta_sugar_free = params[1]
    beta_health_label = params[2]
    beta_price = params[3]
    lambda_main = params[4]
    lambda_mod_pp = params[5]
    lambda_mod_nk = params[6]
    
    # 효용 계산
    V = (intercept 
         + beta_sugar_free * data['sugar_free']
         + beta_health_label * data['health_label']
         + beta_price * data['price']
         + lambda_main * data['purchase_intention']
         + lambda_mod_pp * data['PI_x_PP']
         + lambda_mod_nk * data['PI_x_NK'])
    
    # 선택 확률
    prob_yes = norm.cdf(V)
    prob_yes = np.clip(prob_yes, 1e-10, 1 - 1e-10)
    
    # 로그우도
    choice = data['choice'].values
    ll = np.sum(choice * np.log(prob_yes) + (1 - choice) * np.log(1 - prob_yes))
    
    return -ll  # 음의 로그우도 (최소화)


def estimate_binary_probit(data: pd.DataFrame) -> dict:
    """
    Binary Probit 모델 추정
    
    Returns:
        추정 결과 딕셔너리
    """
    logger.info("=" * 80)
    logger.info("Step 3: Binary Probit 추정")
    logger.info("=" * 80)
    
    # 초기값
    initial_params = np.array([
        0.0,   # intercept
        0.0,   # beta_sugar_free
        0.0,   # beta_health_label
        0.0,   # beta_price
        0.0,   # lambda_main
        0.0,   # lambda_mod_perceived_price
        0.0,   # lambda_mod_nutrition_knowledge
    ])
    
    logger.info("초기값:")
    logger.info(f"  intercept: {initial_params[0]:.6f}")
    logger.info(f"  beta_sugar_free: {initial_params[1]:.6f}")
    logger.info(f"  beta_health_label: {initial_params[2]:.6f}")
    logger.info(f"  beta_price: {initial_params[3]:.6f}")
    logger.info(f"  lambda_main: {initial_params[4]:.6f}")
    logger.info(f"  lambda_mod_perceived_price: {initial_params[5]:.6f}")
    logger.info(f"  lambda_mod_nutrition_knowledge: {initial_params[6]:.6f}")
    
    # 최적화
    logger.info("\n최적화 시작 (BFGS)...")
    result = minimize(
        fun=binary_probit_log_likelihood,
        x0=initial_params,
        args=(data,),
        method='BFGS',
        options={'disp': True, 'maxiter': 1000}
    )
    
    logger.info("\n최적화 완료!")
    logger.info(f"성공 여부: {result.success}")
    logger.info(f"메시지: {result.message}")
    logger.info(f"반복 횟수: {result.nit}")
    logger.info(f"함수 호출: {result.nfev}")
    logger.info(f"Log-Likelihood: {-result.fun:.4f}")
    
    # 추정값
    params = result.x
    logger.info("\n추정값:")
    logger.info(f"  intercept: {params[0]:.6f}")
    logger.info(f"  beta_sugar_free: {params[1]:.6f}")
    logger.info(f"  beta_health_label: {params[2]:.6f}")
    logger.info(f"  beta_price: {params[3]:.6f}")
    logger.info(f"  lambda_main: {params[4]:.6f}")
    logger.info(f"  lambda_mod_perceived_price: {params[5]:.6f}")
    logger.info(f"  lambda_mod_nutrition_knowledge: {params[6]:.6f}")
    
    return {
        'success': result.success,
        'log_likelihood': -result.fun,
        'n_iterations': result.nit,
        'intercept': params[0],
        'beta_sugar_free': params[1],
        'beta_health_label': params[2],
        'beta_price': params[3],
        'lambda_main': params[4],
        'lambda_mod_perceived_price': params[5],
        'lambda_mod_nutrition_knowledge': params[6],
    }


def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("단순 Binary Probit 추정으로 초기값 획득")
    logger.info("=" * 80)
    
    # 데이터 로드
    logger.info("\n데이터 로드...")
    data = pd.read_csv('data/processed/iclv/integrated_data.csv')
    logger.info(f"데이터 크기: {data.shape}")
    logger.info(f"개인 수: {data['respondent_id'].nunique()}")
    logger.info(f"선택 상황 수: {len(data) // data['respondent_id'].nunique()}")
    
    # Step 1: 측정모델 추정 (요인점수)
    factor_scores = estimate_measurement_model_cfa(data)
    
    # Step 2: Probit 데이터 준비
    probit_data = prepare_probit_data(data, factor_scores)
    
    # Step 3: Binary Probit 추정
    results = estimate_binary_probit(probit_data)
    
    # 결과 저장
    logger.info("\n" + "=" * 80)
    logger.info("결과 저장")
    logger.info("=" * 80)
    
    # 초기값 파일 생성
    output_code = f"""\"\"\"
단순 Binary Probit 추정 결과 기반 초기값

추정 방법:
1. 측정모델 (CFA) → 요인점수
2. Binary Probit with 요인점수 및 조절효과

추정 결과:
- Log-Likelihood: {results['log_likelihood']:.4f}
- 반복 횟수: {results['n_iterations']}

생성일: 2025-11-13
\"\"\"

# 선택모델 파라미터 (Binary Probit 추정값)
CHOICE_INITIAL_VALUES = {{
    'intercept': {results['intercept']:.6f},
    'beta_sugar_free': {results['beta_sugar_free']:.6f},
    'beta_health_label': {results['beta_health_label']:.6f},
    'beta_price': {results['beta_price']:.6f},
    'lambda_main': {results['lambda_main']:.6f},
    'lambda_mod_perceived_price': {results['lambda_mod_perceived_price']:.6f},
    'lambda_mod_nutrition_knowledge': {results['lambda_mod_nutrition_knowledge']:.6f},
}}
"""
    
    with open('src/analysis/hybrid_choice_model/iclv_models/initial_values_from_probit.py', 'w', encoding='utf-8') as f:
        f.write(output_code)
    
    logger.info("초기값 파일 저장 완료:")
    logger.info("  → src/analysis/hybrid_choice_model/iclv_models/initial_values_from_probit.py")
    
    logger.info("\n" + "=" * 80)
    logger.info("완료!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

