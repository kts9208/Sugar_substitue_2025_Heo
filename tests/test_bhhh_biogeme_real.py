"""
BHHH 모듈 실제 Biogeme 라이브러리 비교 검증 테스트

현재 구현의 BHHH 계산이 실제 Biogeme 라이브러리의
BHHH 계산과 일치하는지 검증합니다.

Author: Taeseok Kim
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analysis.hybrid_choice_model.iclv_models.bhhh_calculator import BHHHCalculator

# Biogeme import (설치되어 있어야 함)
try:
    import biogeme.database as db
    import biogeme.biogeme as bio
    from biogeme import models
    from biogeme.expressions import Beta, Variable
    BIOGEME_AVAILABLE = True
    print("Biogeme 라이브러리 로드 성공")
except ImportError:
    BIOGEME_AVAILABLE = False
    print("⚠️  Biogeme가 설치되어 있지 않습니다.")
    print("   설치 방법: pip install biogeme")


# ============================================================================
# 테스트 케이스
# ============================================================================

class TestBHHHBiogemeReal:
    """실제 Biogeme 라이브러리 BHHH 비교 테스트"""
    
    def test_bhhh_with_biogeme_logit(self):
        """Biogeme Binary Logit 모델로 BHHH 계산 비교"""
        
        if not BIOGEME_AVAILABLE:
            print("\n⚠️  Biogeme가 설치되어 있지 않아 테스트를 건너뜁니다.")
            print("   설치 방법: pip install biogeme")
            return False
        
        print("\n" + "="*80)
        print("테스트 1: 실제 Biogeme Binary Logit 모델 BHHH 비교")
        print("="*80)
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_obs = 200
        
        # 설명변수
        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)
        
        # True parameters
        true_asc = 0.5
        true_b1 = 1.0
        true_b2 = -0.5
        
        # 효용 및 선택 확률
        utility = true_asc + true_b1 * x1 + true_b2 * x2
        prob = 1 / (1 + np.exp(-utility))
        choice = (np.random.rand(n_obs) < prob).astype(int)
        
        print(f"\n데이터 생성:")
        print(f"  - 관측치 수: {n_obs}")
        print(f"  - choice=1 비율: {choice.mean():.3f}")
        
        # Pandas DataFrame 생성
        data_df = pd.DataFrame({
            'choice': choice,
            'x1': x1,
            'x2': x2
        })
        
        # 1. Biogeme 모델 정의
        print(f"\n1. Biogeme 모델 정의...")
        database = db.Database('test_data', data_df)
        
        # Variables
        CHOICE = Variable('choice')
        X1 = Variable('x1')
        X2 = Variable('x2')
        
        # Parameters
        ASC = Beta('ASC', 0, None, None, 0)
        B1 = Beta('B1', 0, None, None, 0)
        B2 = Beta('B2', 0, None, None, 0)
        
        # Utility
        V = ASC + B1 * X1 + B2 * X2
        
        # Binary logit probability
        P = models.logit({1: V, 0: 0}, None, 1)
        
        # Log-likelihood
        logprob = models.loglogit({1: V, 0: 0}, None, CHOICE)
        
        # 2. Biogeme 추정
        print(f"\n2. Biogeme 모델 추정 중...")
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = 'test_logit'

        # 추정 (BHHH 알고리즘 사용)
        results = biogeme.estimate()

        print(f"  - 추정 완료")
        print(f"  - Log-likelihood: {results.final_loglikelihood:.6f}")
        print(f"  - 파라미터:")
        print(results.get_estimated_parameters())

        # 3. Biogeme BHHH 행렬 추출
        print(f"\n3. Biogeme BHHH 행렬 추출...")

        # Biogeme는 BHHH를 자동으로 계산
        # bhhh_variance_covariance_matrix는 BHHH 기반 공분산 행렬
        bhhh_cov_biogeme = results.bhhh_variance_covariance_matrix

        print(f"  - BHHH 공분산 행렬 Shape: {bhhh_cov_biogeme.shape}")
        print(f"  - BHHH 공분산 대각 원소: {np.diag(bhhh_cov_biogeme)}")

        # BHHH 행렬 = inv(공분산)
        bhhh_biogeme = np.linalg.inv(bhhh_cov_biogeme)

        print(f"  - BHHH 행렬 Shape: {bhhh_biogeme.shape}")
        print(f"  - BHHH 행렬 범위: [{bhhh_biogeme.min():.6e}, {bhhh_biogeme.max():.6e}]")

        # 4. 현재 구현으로 BHHH 계산
        print(f"\n4. 현재 구현 BHHH 계산...")

        # 개인별 gradient 계산 (수치 미분)
        params = results.get_beta_values()
        param_names = list(params.keys())
        param_values = np.array([params[name] for name in param_names])
        
        print(f"  - 추정된 파라미터: {param_values}")
        
        # 개인별 gradient 계산 (간단한 수치 미분)
        epsilon = 1e-6
        individual_gradients = []
        
        for i in range(n_obs):
            # 개인 i의 데이터
            x1_i = data_df.loc[i, 'x1']
            x2_i = data_df.loc[i, 'x2']
            choice_i = data_df.loc[i, 'choice']
            
            # 개인 i의 log-likelihood
            def ll_i(p):
                asc, b1, b2 = p
                v = asc + b1 * x1_i + b2 * x2_i
                prob = 1 / (1 + np.exp(-v))
                prob = np.clip(prob, 1e-10, 1 - 1e-10)
                return choice_i * np.log(prob) + (1 - choice_i) * np.log(1 - prob)
            
            # 수치 미분으로 gradient 계산
            grad_i = np.zeros(len(param_values))
            for j in range(len(param_values)):
                p_plus = param_values.copy()
                p_plus[j] += epsilon
                p_minus = param_values.copy()
                p_minus[j] -= epsilon
                grad_i[j] = (ll_i(p_plus) - ll_i(p_minus)) / (2 * epsilon)
            
            individual_gradients.append(grad_i)
        
        print(f"  - 개인별 gradient 계산 완료: {len(individual_gradients)}개")
        
        # BHHH 계산
        bhhh_calc = BHHHCalculator()
        hessian_bhhh_ours = bhhh_calc.compute_bhhh_hessian(
            individual_gradients,
            for_minimization=False  # Biogeme는 최대화
        )
        
        print(f"  - BHHH 행렬 Shape: {hessian_bhhh_ours.shape}")
        print(f"  - BHHH 행렬 범위: [{hessian_bhhh_ours.min():.6e}, {hessian_bhhh_ours.max():.6e}]")
        
        # 5. 비교
        print(f"\n5. 비교 결과:")
        diff = np.abs(hessian_bhhh_ours - bhhh_biogeme)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = mean_diff / np.mean(np.abs(bhhh_biogeme))
        
        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {rel_error:.6e}")
        
        # 검증 (수치 미분 오차 고려하여 관대한 tolerance)
        assert np.allclose(hessian_bhhh_ours, bhhh_biogeme, rtol=1e-2, atol=1e-3), \
            f"BHHH 행렬 불일치: 최대 차이 {max_diff}"
        
        print("\n✅ 실제 Biogeme BHHH와 일치! (수치 미분 오차 범위 내)")
        return True
    
    def test_bhhh_standard_errors_with_biogeme(self):
        """Biogeme BHHH 표준오차 비교"""

        if not BIOGEME_AVAILABLE:
            print("\n⚠️  Biogeme가 설치되어 있지 않아 테스트를 건너뜁니다.")
            return False

        print("\n" + "="*80)
        print("테스트 2: 실제 Biogeme BHHH 공분산 행렬 비교")
        print("="*80)

        # 샘플 데이터 생성 (테스트 1과 동일)
        np.random.seed(42)
        n_obs = 200

        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        utility = 0.5 + 1.0 * x1 - 0.5 * x2
        prob = 1 / (1 + np.exp(-utility))
        choice = (np.random.rand(n_obs) < prob).astype(int)

        data_df = pd.DataFrame({
            'choice': choice,
            'x1': x1,
            'x2': x2
        })

        # 1. Biogeme 모델 추정
        print(f"\n1. Biogeme 모델 추정...")
        database = db.Database('test_data', data_df)

        CHOICE = Variable('choice')
        X1 = Variable('x1')
        X2 = Variable('x2')

        ASC = Beta('ASC', 0, None, None, 0)
        B1 = Beta('B1', 0, None, None, 0)
        B2 = Beta('B2', 0, None, None, 0)

        V = ASC + B1 * X1 + B2 * X2
        logprob = models.loglogit({1: V, 0: 0}, None, CHOICE)

        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = 'test_logit'
        results = biogeme.estimate()

        # Biogeme BHHH 공분산 행렬
        bhhh_cov_biogeme = results.bhhh_variance_covariance_matrix

        print(f"  - Biogeme BHHH 공분산 행렬:")
        print(f"    Shape: {bhhh_cov_biogeme.shape}")
        print(f"    대각 원소: {np.diag(bhhh_cov_biogeme)}")
        print(f"    sqrt(대각): {np.sqrt(np.diag(bhhh_cov_biogeme))}")

        # 2. 현재 구현으로 표준오차 계산
        print(f"\n2. 현재 구현 BHHH 표준오차 계산...")

        bhhh_calc = BHHHCalculator()
        se_ours = bhhh_calc.compute_standard_errors(bhhh_cov_biogeme)

        print(f"  - 현재 구현 BHHH 표준오차: {se_ours}")

        # 3. 비교
        print(f"\n3. 비교 결과:")

        # Biogeme 공분산 행렬에서 표준오차 계산
        se_biogeme = np.sqrt(np.diag(bhhh_cov_biogeme))

        print(f"  - Biogeme sqrt(diag(cov)): {se_biogeme}")
        print(f"  - 현재 구현 SE: {se_ours}")

        diff = np.abs(se_ours - se_biogeme)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = mean_diff / np.mean(se_biogeme)

        print(f"  - 최대 차이: {max_diff:.6e}")
        print(f"  - 평균 차이: {mean_diff:.6e}")
        print(f"  - 상대 오차: {rel_error:.6e}")

        # 검증
        assert np.allclose(se_ours, se_biogeme, rtol=1e-10, atol=1e-12), \
            f"BHHH 표준오차 불일치: 최대 차이 {max_diff}"

        print("\n✅ 실제 Biogeme BHHH 공분산 행렬과 완벽히 일치!")
        print("   (주의: Biogeme의 'Robust std err'는 Sandwich estimator 사용)")
        return True


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("실제 Biogeme 라이브러리 BHHH 비교 검증 테스트")
    print("="*80)
    
    if not BIOGEME_AVAILABLE:
        print("\n⚠️  Biogeme가 설치되어 있지 않습니다.")
        print("   설치 방법: pip install biogeme")
        print("\n테스트를 건너뜁니다.")
        sys.exit(0)
    
    test = TestBHHHBiogemeReal()
    
    try:
        # 테스트 실행
        result1 = test.test_bhhh_with_biogeme_logit()
        result2 = test.test_bhhh_standard_errors_with_biogeme()
        
        if result1 and result2:
            print("\n" + "="*80)
            print("✅ 모든 실제 Biogeme BHHH 검증 테스트 통과!")
            print("="*80)
            print("\n결론:")
            print("  - 현재 구현의 BHHH 계산이 실제 Biogeme와 일치합니다.")
            print("  - BHHH 표준오차가 실제 Biogeme와 일치합니다.")
            print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

