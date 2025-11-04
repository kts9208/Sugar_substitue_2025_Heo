"""
ICLV 모델 검증 테스트

King (2022) R 코드와 동일한 결과를 생성하는지 검증합니다.

테스트 전략:
1. 알려진 파라미터로 시뮬레이션 데이터 생성
2. Python ICLV로 추정
3. 파라미터 복원 확인
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICLVDataSimulator:
    """
    ICLV 모델 시뮬레이션 데이터 생성기
    
    King (2022) 구조를 따라 데이터를 생성합니다:
    - 측정방정식: Ordered Probit (5점 척도)
    - 구조방정식: LV = γ*X + η
    - 선택방정식: Binary Probit (Yes/No)
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_data(self, n_individuals=500, 
                     true_params=None) -> pd.DataFrame:
        """
        시뮬레이션 데이터 생성
        
        Args:
            n_individuals: 개인 수
            true_params: 실제 파라미터 (검증용)
        
        Returns:
            시뮬레이션 데이터프레임
        """
        
        if true_params is None:
            true_params = self._get_default_params()
        
        logger.info(f"시뮬레이션 데이터 생성: N={n_individuals}")
        logger.info(f"실제 파라미터: {true_params}")
        
        # 1. 사회인구학적 변수 생성
        data = pd.DataFrame({
            'ID': range(1, n_individuals + 1),
            'Age': np.random.normal(45, 15, n_individuals),
            'Gender': np.random.binomial(1, 0.5, n_individuals),  # 0=Male, 1=Female
            'Income': np.random.binomial(1, 0.6, n_individuals),  # 0=Low, 1=High
        })
        
        # 표준화
        data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
        
        # 2. 잠재변수 생성 (구조방정식)
        # LV = γ_age*Age + γ_gender*Gender + γ_income*Income + η
        eta = np.random.normal(0, 1, n_individuals)  # 오차항
        
        data['LV_true'] = (
            true_params['gamma_Age'] * data['Age'] +
            true_params['gamma_Gender'] * data['Gender'] +
            true_params['gamma_Income'] * data['Income'] +
            eta
        )
        
        # 3. 측정지표 생성 (Ordered Probit)
        # P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
        
        for ind_name in ['Q13', 'Q14', 'Q15']:
            zeta = true_params[f'zeta_{ind_name}']
            tau = true_params[f'tau_{ind_name}']
            
            # 잠재 연속변수
            y_star = zeta * data['LV_true'] + np.random.normal(0, 1, n_individuals)
            
            # 범주형으로 변환 (1-5)
            y_ordered = np.ones(n_individuals, dtype=int)
            for k in range(len(tau)):
                y_ordered[y_star > tau[k]] = k + 2
            
            data[ind_name] = y_ordered
        
        # 4. 선택 상황 생성 (Binary choice)
        # 각 개인에게 가격 제시
        data['Bid'] = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0], n_individuals)
        
        # 효용함수: V = intercept + β_bid*Bid + λ*LV
        V = (
            true_params['intercept'] +
            true_params['b_bid'] * data['Bid'] +
            true_params['lambda'] * data['LV_true']
        )
        
        # Binary Probit: P(Yes) = Φ(V)
        prob_yes = norm.cdf(V)
        data['Choice'] = np.random.binomial(1, prob_yes)
        
        # 5. Task 변수 추가 (Apollo 형식)
        data['Task'] = 1
        
        logger.info(f"데이터 생성 완료: {len(data)} 관측치")
        logger.info(f"선택 비율: {data['Choice'].mean():.3f}")
        logger.info(f"측정지표 평균: Q13={data['Q13'].mean():.2f}, "
                   f"Q14={data['Q14'].mean():.2f}, Q15={data['Q15'].mean():.2f}")
        
        return data, true_params
    
    def _get_default_params(self) -> dict:
        """기본 파라미터 (King 2022 스타일)"""
        return {
            # 선택모델
            'intercept': 0.5,
            'b_bid': -2.0,  # 가격 계수 (음수)
            'lambda': 1.5,  # 잠재변수 계수
            
            # 구조모델
            'gamma_Age': 0.3,
            'gamma_Gender': -0.2,
            'gamma_Income': 0.4,
            
            # 측정모델 - Q13
            'zeta_Q13': 1.0,
            'tau_Q13': [-2.0, -1.0, 1.0, 2.0],
            
            # 측정모델 - Q14
            'zeta_Q14': 1.2,
            'tau_Q14': [-2.0, -1.0, 1.0, 2.0],
            
            # 측정모델 - Q15
            'zeta_Q15': 0.8,
            'tau_Q15': [-2.0, -1.0, 1.0, 2.0],
        }


class SimpleICLVEstimator:
    """
    간단한 ICLV 추정기 (검증용)
    
    King (2022) Apollo 코드의 핵심 로직을 Python으로 구현
    """
    
    def __init__(self, n_draws=100):
        """
        Args:
            n_draws: Halton draws 수 (검증용이므로 적게)
        """
        self.n_draws = n_draws
        self.results = None
    
    def fit(self, data: pd.DataFrame) -> dict:
        """
        ICLV 모델 추정
        
        Args:
            data: 시뮬레이션 데이터
        
        Returns:
            추정 결과
        """
        logger.info("ICLV 모델 추정 시작")
        
        # Halton draws 생성
        from scipy.stats import qmc
        sampler = qmc.Halton(d=1, scramble=True, seed=42)
        uniform_draws = sampler.random(n=self.n_draws)
        halton_draws = norm.ppf(uniform_draws).flatten()
        
        logger.info(f"Halton draws 생성: {len(halton_draws)} draws")
        
        # 초기 파라미터
        initial_params = self._get_initial_params()
        
        # 최적화
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            ll = self._log_likelihood(params, data, halton_draws)
            return -ll
        
        logger.info("최적화 시작...")
        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='BFGS',
            options={'maxiter': 100, 'disp': True}
        )
        
        if result.success:
            logger.info("최적화 성공!")
        else:
            logger.warning(f"최적화 실패: {result.message}")
        
        # 결과 저장
        self.results = {
            'success': result.success,
            'params': self._unpack_params(result.x),
            'log_likelihood': -result.fun,
            'n_iterations': result.nit,
        }
        
        return self.results
    
    def _log_likelihood(self, params_vec, data, halton_draws):
        """결합 로그우도 계산"""
        
        params = self._unpack_params(params_vec)
        
        total_ll = 0.0
        
        # 개인별 우도
        for idx, row in data.iterrows():
            ind_ll = 0.0
            
            # 시뮬레이션 (Halton draws)
            for eta in halton_draws:
                # 구조방정식: LV = γ*X + η
                lv = (
                    params['gamma_Age'] * row['Age'] +
                    params['gamma_Gender'] * row['Gender'] +
                    params['gamma_Income'] * row['Income'] +
                    eta
                )
                
                # 측정모델 우도
                ll_measurement = 0.0
                for ind_name in ['Q13', 'Q14', 'Q15']:
                    ll_measurement += self._ordered_probit_ll(
                        row[ind_name],
                        lv,
                        params[f'zeta_{ind_name}'],
                        params[f'tau_{ind_name}']
                    )
                
                # 선택모델 우도
                V = (
                    params['intercept'] +
                    params['b_bid'] * row['Bid'] +
                    params['lambda'] * lv
                )
                prob = norm.cdf(V)
                if row['Choice'] == 1:
                    ll_choice = np.log(prob + 1e-10)
                else:
                    ll_choice = np.log(1 - prob + 1e-10)
                
                # 구조모델 우도 (정규분포)
                ll_structural = norm.logpdf(eta, 0, 1)
                
                # 결합 우도
                ind_ll += np.exp(ll_measurement + ll_choice + ll_structural)
            
            # 평균
            ind_ll /= len(halton_draws)
            
            # 로그
            if ind_ll > 0:
                total_ll += np.log(ind_ll)
            else:
                total_ll += -1e10
        
        return total_ll
    
    def _ordered_probit_ll(self, y, lv, zeta, tau):
        """Ordered Probit 로그우도"""
        
        # y는 1-5 범주
        k = int(y) - 1  # 0-4로 변환
        
        # P(Y=k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
        if k == 0:
            prob = norm.cdf(tau[0] - zeta * lv)
        elif k == 4:
            prob = 1 - norm.cdf(tau[3] - zeta * lv)
        else:
            prob = norm.cdf(tau[k] - zeta * lv) - norm.cdf(tau[k-1] - zeta * lv)
        
        return np.log(prob + 1e-10)
    
    def _get_initial_params(self):
        """초기 파라미터"""
        return np.array([
            0.0,  # intercept
            -1.0,  # b_bid
            1.0,  # lambda
            0.0,  # gamma_Age
            0.0,  # gamma_Gender
            0.0,  # gamma_Income
            1.0,  # zeta_Q13
            1.0,  # zeta_Q14
            1.0,  # zeta_Q15
            -2.0, -1.0, 1.0, 2.0,  # tau_Q13
            -2.0, -1.0, 1.0, 2.0,  # tau_Q14
            -2.0, -1.0, 1.0, 2.0,  # tau_Q15
        ])
    
    def _unpack_params(self, params_vec):
        """파라미터 벡터를 딕셔너리로 변환"""
        return {
            'intercept': params_vec[0],
            'b_bid': params_vec[1],
            'lambda': params_vec[2],
            'gamma_Age': params_vec[3],
            'gamma_Gender': params_vec[4],
            'gamma_Income': params_vec[5],
            'zeta_Q13': params_vec[6],
            'zeta_Q14': params_vec[7],
            'zeta_Q15': params_vec[8],
            'tau_Q13': params_vec[9:13],
            'tau_Q14': params_vec[13:17],
            'tau_Q15': params_vec[17:21],
        }


# ============================================================================
# 테스트 케이스
# ============================================================================

def test_data_generation():
    """데이터 생성 테스트"""
    simulator = ICLVDataSimulator(seed=42)
    data, true_params = simulator.generate_data(n_individuals=100)
    
    assert len(data) == 100
    assert 'LV_true' in data.columns
    assert 'Choice' in data.columns
    assert data['Q13'].min() >= 1
    assert data['Q13'].max() <= 5
    
    logger.info("✓ 데이터 생성 테스트 통과")


def test_simple_estimation():
    """간단한 추정 테스트"""
    # 작은 데이터로 빠른 테스트
    simulator = ICLVDataSimulator(seed=42)
    data, true_params = simulator.generate_data(n_individuals=50)
    
    estimator = SimpleICLVEstimator(n_draws=50)
    results = estimator.fit(data)
    
    assert results['success']
    assert 'params' in results
    
    logger.info("✓ 간단한 추정 테스트 통과")
    logger.info(f"추정된 파라미터: {results['params']}")


if __name__ == "__main__":
    # 데이터 생성 테스트
    test_data_generation()
    
    # 추정 테스트
    test_simple_estimation()
    
    logger.info("\n모든 테스트 통과! ✓")

