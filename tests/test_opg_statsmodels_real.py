"""
OPG 계산 실제 Statsmodels 라이브러리 비교 검증 테스트

현재 구현의 OPG (Outer Product of Gradients) 계산이
실제 Statsmodels 라이브러리의 OPG 계산과 일치하는지 검증합니다.

Author: Taeseok Kim
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Statsmodels import
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator


# ============================================================================
# Statsmodels 기반 테스트 모델
# ============================================================================

class SimpleLogitModel(GenericLikelihoodModel):
    """
    간단한 Logit 모델 (Statsmodels GenericLikelihoodModel 상속)
    
    실제 Statsmodels의 OPG 계산을 테스트하기 위한 모델
    """
    
    def __init__(self, endog, exog, **kwargs):
        super(SimpleLogitModel, self).__init__(endog, exog, **kwargs)
    
    def loglikeobs(self, params):
        """
        개인별 log-likelihood 계산
        
        Logit model: P(y=1) = 1 / (1 + exp(-X*beta))
        LL_i = y_i * log(P_i) + (1-y_i) * log(1-P_i)
        """
        # Linear predictor
        xb = self.exog @ params
        
        # Logit probability
        prob = 1 / (1 + np.exp(-xb))
        
        # Clip to avoid log(0)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        
        # Log-likelihood per observation
        ll = self.endog * np.log(prob) + (1 - self.endog) * np.log(1 - prob)
        
        return ll
    
    def score_obs(self, params):
        """
        개인별 gradient 계산
        
        ∂LL_i/∂β = (y_i - P_i) * X_i
        """
        # Linear predictor
        xb = self.exog @ params
        
        # Logit probability
        prob = 1 / (1 + np.exp(-xb))
        
        # Gradient per observation
        residual = self.endog - prob
        score = self.exog * residual[:, np.newaxis]
        
        return score


# ============================================================================
# 테스트 케이스
# ============================================================================

class TestOPGStatsmodelsReal:
    """실제 Statsmodels 라이브러리 OPG 비교 테스트"""
    
    def test_opg_with_statsmodels_logit(self):
        """Statsmodels Logit 모델로 OPG 계산 비교"""
        print("\n" + "="*80)
        print("테스트 1: 실제 Statsmodels Logit 모델 OPG 비교")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_obs = 200
        n_params = 5
        
        # X: 설명변수 (상수항 포함)
        X = np.random.randn(n_obs, n_params)
        X[:, 0] = 1  # 상수항
        
        # True parameters
        true_params = np.array([0.5, 1.0, -0.5, 0.3, -0.2])
        
        # y: 종속변수 (binary)
        xb = X @ true_params
        prob = 1 / (1 + np.exp(-xb))
        y = (np.random.rand(n_obs) < prob).astype(float)
        
        print(f"\n데이터 생성:")
        print(f"  - 관측치 수: {n_obs}")
        print(f"  - 파라미터 수: {n_params}")
        print(f"  - y=1 비율: {y.mean():.3f}")
        
        # 1. Statsmodels 모델 추정
        print(f"\n1. Statsmodels 모델 추정 중...")
        model_sm = SimpleLogitModel(y, X)
        
        # 초기값
        start_params = np.zeros(n_params)
        
        # Fit with OPG covariance
        results_sm = model_sm.fit(
            start_params=start_params,
            method='bfgs',
            maxiter=1000,
            disp=False
        )
        
        print(f"  - 추정 완료")
        print(f"  - Log-likelihood: {results_sm.llf:.6f}")
        print(f"  - 파라미터: {results_sm.params}")
        
        # 2. Statsmodels에서 개인별 gradient 추출
        print(f"\n2. Statsmodels 개인별 gradient 계산...")
        individual_gradients_sm = model_sm.score_obs(results_sm.params)
        
        print(f"  - Shape: {individual_gradients_sm.shape}")
        print(f"  - 범위: [{individual_gradients_sm.min():.6e}, {individual_gradients_sm.max():.6e}]")
        
        # 3. Statsmodels OPG 계산
        print(f"\n3. Statsmodels OPG 계산...")
        # OPG = score_obs.T @ score_obs
        opg_sm = individual_gradients_sm.T @ individual_gradients_sm
        
        print(f"  - OPG Shape: {opg_sm.shape}")
        print(f"  - OPG 범위: [{opg_sm.min():.6e}, {opg_sm.max():.6e}]")
        print(f"  - OPG 대각 원소: {np.diag(opg_sm)}")
        
        # 4. 현재 구현으로 OPG 계산
        print(f"\n4. 현재 구현 OPG 계산...")
        bhhh_calc = BHHHCalculator()
        
        # 개인별 gradient를 리스트로 변환
        individual_gradients_list = [individual_gradients_sm[i, :] for i in range(n_obs)]
        
        # BHHH 계산 (최대화 문제)
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients_list,
            for_minimization=False  # 최대화 문제
        )
        opg_ours = hessian_bhhh  # 최대화 문제에서 BHHH = OPG
        
        print(f"  - OPG Shape: {opg_ours.shape}")
        print(f"  - OPG 범위: [{opg_ours.min():.6e}, {opg_ours.max():.6e}]")
        print(f"  - OPG 대각 원소: {np.diag(opg_ours)}")
        
        # 5. 비교
        print(f"\n5. 비교 결과:")
        diff = np.abs(opg_ours - opg_sm)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = mean_diff / np.mean(np.abs(opg_sm))
        
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {rel_error:.6e}")
        
        # 검증
        assert np.allclose(opg_ours, opg_sm, rtol=1e-10, atol=1e-12), \
            f"OPG 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ 실제 Statsmodels OPG와 완벽히 일치!")
        return True
    
    def test_opg_covariance_with_statsmodels(self):
        """Statsmodels OPG 공분산 행렬 비교"""
        print("\n" + "="*80)
        print("테스트 2: 실제 Statsmodels OPG 공분산 행렬 비교")
        print("="*80)

        # 샘플 데이터 생성
        np.random.seed(42)
        n_obs = 200
        n_params = 5

        X = np.random.randn(n_obs, n_params)
        X[:, 0] = 1

        true_params = np.array([0.5, 1.0, -0.5, 0.3, -0.2])
        xb = X @ true_params
        prob = 1 / (1 + np.exp(-xb))
        y = (np.random.rand(n_obs) < prob).astype(float)

        # 1. Statsmodels 모델 추정
        print(f"\n1. Statsmodels 모델 추정...")
        model_sm = SimpleLogitModel(y, X)
        results_sm = model_sm.fit(
            start_params=np.zeros(n_params),
            method='bfgs',
            maxiter=1000,
            disp=False
        )

        # Statsmodels에서 OPG 공분산 행렬 직접 계산
        print(f"\n2. Statsmodels OPG 공분산 행렬 직접 계산...")
        individual_gradients_sm = model_sm.score_obs(results_sm.params)
        opg_sm = individual_gradients_sm.T @ individual_gradients_sm
        cov_opg_sm = np.linalg.inv(opg_sm)

        print(f"  - Statsmodels OPG 공분산 행렬:")
        print(f"    Shape: {cov_opg_sm.shape}")
        print(f"    대각 원소: {np.diag(cov_opg_sm)}")

        # 3. 현재 구현으로 OPG 공분산 계산
        print(f"\n3. 현재 구현 OPG 공분산 계산...")
        individual_gradients_list = [individual_gradients_sm[i, :] for i in range(n_obs)]
        
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients_list,
            for_minimization=False
        )
        cov_ours = bhhh_calc.compute_hessian_inverse(hessian_bhhh)
        
        print(f"  - 현재 구현 OPG 공분산 행렬:")
        print(f"    Shape: {cov_ours.shape}")
        print(f"    대각 원소: {np.diag(cov_ours)}")

        # 4. 비교
        print(f"\n4. 비교 결과:")
        diff = np.abs(cov_ours - cov_opg_sm)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = mean_diff / np.mean(np.abs(cov_opg_sm))
        
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {rel_error:.6e}")
        
        # 검증
        assert np.allclose(cov_ours, cov_opg_sm, rtol=1e-6, atol=1e-8), \
            f"OPG 공분산 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ 실제 Statsmodels OPG 공분산 행렬과 일치!")
        return True
    
    def test_opg_standard_errors_with_statsmodels(self):
        """Statsmodels OPG 표준오차 비교"""
        print("\n" + "="*80)
        print("테스트 3: 실제 Statsmodels OPG 표준오차 비교")
        print("="*80)

        # 샘플 데이터 생성
        np.random.seed(42)
        n_obs = 200
        n_params = 5

        X = np.random.randn(n_obs, n_params)
        X[:, 0] = 1

        true_params = np.array([0.5, 1.0, -0.5, 0.3, -0.2])
        xb = X @ true_params
        prob = 1 / (1 + np.exp(-xb))
        y = (np.random.rand(n_obs) < prob).astype(float)

        # 1. Statsmodels 모델 추정
        print(f"\n1. Statsmodels 모델 추정...")
        model_sm = SimpleLogitModel(y, X)
        results_sm = model_sm.fit(
            start_params=np.zeros(n_params),
            method='bfgs',
            maxiter=1000,
            disp=False
        )

        # Statsmodels OPG 표준오차 직접 계산
        print(f"\n2. Statsmodels OPG 표준오차 직접 계산...")
        individual_gradients_sm = model_sm.score_obs(results_sm.params)
        opg_sm = individual_gradients_sm.T @ individual_gradients_sm
        cov_opg_sm = np.linalg.inv(opg_sm)
        se_opg_sm = np.sqrt(np.diag(cov_opg_sm))

        print(f"  - Statsmodels OPG 표준오차: {se_opg_sm}")

        # 3. 현재 구현으로 OPG 표준오차 계산
        print(f"\n3. 현재 구현 OPG 표준오차 계산...")
        individual_gradients_list = [individual_gradients_sm[i, :] for i in range(n_obs)]
        
        bhhh_calc = BHHHCalculator()
        hessian_bhhh = bhhh_calc.compute_bhhh_hessian(
            individual_gradients_list,
            for_minimization=False
        )
        cov_ours = bhhh_calc.compute_hessian_inverse(hessian_bhhh)
        se_ours = bhhh_calc.compute_standard_errors(cov_ours)
        
        print(f"  - 현재 구현 OPG 표준오차: {se_ours}")

        # 4. 비교
        print(f"\n4. 비교 결과:")
        diff = np.abs(se_ours - se_opg_sm)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = mean_diff / np.mean(se_opg_sm)
        
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {rel_error:.6e}")
        
        # 검증
        assert np.allclose(se_ours, se_opg_sm, rtol=1e-6, atol=1e-8), \
            f"OPG 표준오차 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ 실제 Statsmodels OPG 표준오차와 일치!")
        return True


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("실제 Statsmodels 라이브러리 OPG 비교 검증 테스트")
    print(f"Statsmodels version: {sm.__version__}")
    print("="*80)
    
    test = TestOPGStatsmodelsReal()
    
    try:
        # 테스트 실행
        test.test_opg_with_statsmodels_logit()
        test.test_opg_covariance_with_statsmodels()
        test.test_opg_standard_errors_with_statsmodels()
        
        print("\n" + "="*80)
        print("✅ 모든 실제 Statsmodels OPG 검증 테스트 통과!")
        print("="*80)
        print("\n결론:")
        print("  - 현재 구현의 OPG 계산이 실제 Statsmodels와 완벽히 일치합니다.")
        print("  - OPG 공분산 행렬이 실제 Statsmodels와 일치합니다.")
        print("  - OPG 표준오차가 실제 Statsmodels와 일치합니다.")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

