"""
BHHH (Berndt-Hall-Hall-Hausman) Hessian 계산 모듈

BHHH 방법을 사용하여 Hessian 행렬을 근사하고 표준오차를 계산합니다.

이론적 배경:
    Maximum Likelihood Estimation에서:
    Hessian = ∂²LL/∂θ∂θ^T = Σ_i ∂²LL_i/∂θ∂θ^T
    
    BHHH 근사:
    Hessian ≈ Σ_i (∂LL_i/∂θ) × (∂LL_i/∂θ)^T
            = Σ_i (grad_i × grad_i^T)
    
    여기서:
    - LL_i: 개인 i의 log-likelihood
    - grad_i: 개인 i의 gradient (∂LL_i/∂θ)
    - Σ_i: 모든 개인에 대한 합

장점:
    1. 계산 효율성: 우도 계산 불필요 (gradient만 필요)
    2. 개인별 gradient는 이미 계산됨 (analytic gradient 사용 시)
    3. 전체 Hessian 행렬 계산 (상관관계 포함)
    4. Robust 표준오차 계산 가능 (Sandwich estimator)

Author: Taeseok Kim
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging


class BHHHCalculator:
    """
    BHHH 방법을 사용한 Hessian 계산 및 표준오차 추정
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Args:
            logger: 로거 객체 (선택사항)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.hessian_bhhh = None
        self.hessian_inv = None
        self.standard_errors = None
        self.robust_standard_errors = None
        
    def compute_bhhh_hessian(
        self,
        individual_gradients: List[np.ndarray],
        for_minimization: bool = True
    ) -> np.ndarray:
        """
        개인별 gradient로부터 BHHH Hessian 계산
        
        Args:
            individual_gradients: 개인별 gradient 벡터 리스트
                                 각 원소는 (n_params,) 형태의 numpy array
            for_minimization: True면 최소화 문제 (음수 부호)
                            False면 최대화 문제 (양수 부호)
        
        Returns:
            BHHH Hessian 행렬 (n_params, n_params)
        """
        if not individual_gradients:
            raise ValueError("개인별 gradient가 비어있습니다.")
        
        n_individuals = len(individual_gradients)
        n_params = len(individual_gradients[0])
        
        self.logger.info(f"BHHH Hessian 계산 시작: {n_individuals}명, {n_params}개 파라미터")
        
        # BHHH Hessian 초기화
        hessian_bhhh = np.zeros((n_params, n_params))
        
        # Σ_i (grad_i × grad_i^T)
        for i, grad in enumerate(individual_gradients):
            if len(grad) != n_params:
                raise ValueError(
                    f"개인 {i}의 gradient 차원 불일치: "
                    f"예상 {n_params}, 실제 {len(grad)}"
                )
            
            # Outer product: grad_i × grad_i^T
            hessian_bhhh += np.outer(grad, grad)
        
        # 최소화 문제의 경우 음수 부호
        if for_minimization:
            hessian_bhhh = -hessian_bhhh
        
        self.hessian_bhhh = hessian_bhhh
        
        # 통계 로깅
        self._log_hessian_statistics(hessian_bhhh, "BHHH Hessian")
        
        return hessian_bhhh
    
    def compute_hessian_inverse(
        self,
        hessian: Optional[np.ndarray] = None,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Hessian 역행렬 계산
        
        Args:
            hessian: Hessian 행렬 (None이면 self.hessian_bhhh 사용)
            regularization: 정규화 항 (수치 안정성)
        
        Returns:
            Hessian 역행렬 (n_params, n_params)
        """
        if hessian is None:
            if self.hessian_bhhh is None:
                raise ValueError("Hessian이 계산되지 않았습니다.")
            hessian = self.hessian_bhhh
        
        n_params = hessian.shape[0]
        
        try:
            # 정규화 (수치 안정성)
            hessian_reg = hessian + regularization * np.eye(n_params)
            
            # 역행렬 계산
            hess_inv = np.linalg.inv(hessian_reg)
            
            self.hessian_inv = hess_inv
            self.logger.info("Hessian 역행렬 계산 성공")
            
            # 통계 로깅
            self._log_hessian_statistics(hess_inv, "Hessian 역행렬")
            
            return hess_inv
            
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Hessian 역행렬 계산 실패: {e}")
            
            # Pseudo-inverse 시도
            self.logger.warning("Pseudo-inverse로 대체 시도")
            try:
                hess_inv = np.linalg.pinv(hessian_reg)
                self.hessian_inv = hess_inv
                self.logger.info("Pseudo-inverse 계산 성공")
                return hess_inv
            except Exception as e2:
                self.logger.error(f"Pseudo-inverse도 실패: {e2}")
                raise
    
    def compute_standard_errors(
        self,
        hessian_inv: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        표준오차 계산
        
        SE = sqrt(diag(H^(-1)))
        
        Args:
            hessian_inv: Hessian 역행렬 (None이면 self.hessian_inv 사용)
        
        Returns:
            표준오차 벡터 (n_params,)
        """
        if hessian_inv is None:
            if self.hessian_inv is None:
                raise ValueError("Hessian 역행렬이 계산되지 않았습니다.")
            hessian_inv = self.hessian_inv
        
        # 대각 원소 추출 (분산)
        variances = np.diag(hessian_inv)
        
        # 음수 분산 처리 (수치 오류)
        n_negative = np.sum(variances < 0)
        if n_negative > 0:
            self.logger.warning(
                f"음수 분산 {n_negative}개 발견 - 절대값 사용"
            )
            variances = np.abs(variances)
        
        # 표준오차 = sqrt(분산)
        standard_errors = np.sqrt(variances)
        
        self.standard_errors = standard_errors
        
        self.logger.info(
            f"표준오차 계산 완료: "
            f"범위 [{np.min(standard_errors):.6e}, {np.max(standard_errors):.6e}]"
        )
        
        return standard_errors
    
    def compute_robust_standard_errors(
        self,
        hessian_bhhh: np.ndarray,
        hessian_numerical: np.ndarray,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Robust 표준오차 계산 (Sandwich estimator)
        
        Var(θ) = H^(-1) @ BHHH @ H^(-1)
        SE = sqrt(diag(Var(θ)))
        
        여기서:
        - H: 수치적 Hessian (또는 BFGS Hessian)
        - BHHH: BHHH Hessian
        
        Args:
            hessian_bhhh: BHHH Hessian 행렬
            hessian_numerical: 수치적 Hessian 행렬
            regularization: 정규화 항
        
        Returns:
            Robust 표준오차 벡터 (n_params,)
        """
        n_params = hessian_bhhh.shape[0]
        
        try:
            # 수치적 Hessian 역행렬
            hess_num_reg = hessian_numerical + regularization * np.eye(n_params)
            hess_num_inv = np.linalg.inv(hess_num_reg)
            
            # Sandwich estimator: H^(-1) @ BHHH @ H^(-1)
            variance_matrix = hess_num_inv @ hessian_bhhh @ hess_num_inv
            
            # 대각 원소 추출
            variances = np.diag(variance_matrix)
            
            # 음수 분산 처리
            n_negative = np.sum(variances < 0)
            if n_negative > 0:
                self.logger.warning(
                    f"Robust SE: 음수 분산 {n_negative}개 발견 - 절대값 사용"
                )
                variances = np.abs(variances)
            
            # Robust 표준오차
            robust_se = np.sqrt(variances)
            
            self.robust_standard_errors = robust_se
            
            self.logger.info(
                f"Robust 표준오차 계산 완료: "
                f"범위 [{np.min(robust_se):.6e}, {np.max(robust_se):.6e}]"
            )
            
            return robust_se
            
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Robust 표준오차 계산 실패: {e}")
            raise
    
    def compute_t_statistics(
        self,
        parameters: np.ndarray,
        standard_errors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        t-통계량 계산
        
        t = θ / SE(θ)
        
        Args:
            parameters: 파라미터 벡터
            standard_errors: 표준오차 벡터 (None이면 self.standard_errors 사용)
        
        Returns:
            t-통계량 벡터 (n_params,)
        """
        if standard_errors is None:
            if self.standard_errors is None:
                raise ValueError("표준오차가 계산되지 않았습니다.")
            standard_errors = self.standard_errors
        
        # t = θ / SE
        t_stats = parameters / standard_errors
        
        return t_stats
    
    def compute_p_values(
        self,
        t_statistics: np.ndarray,
        use_normal: bool = True
    ) -> np.ndarray:
        """
        p-값 계산 (양측 검정)
        
        Args:
            t_statistics: t-통계량 벡터
            use_normal: True면 정규분포, False면 t-분포 사용
        
        Returns:
            p-값 벡터 (n_params,)
        """
        if use_normal:
            # 정규분포 사용 (대표본)
            from scipy.stats import norm
            p_values = 2 * (1 - norm.cdf(np.abs(t_statistics)))
        else:
            # t-분포 사용 (소표본)
            from scipy.stats import t
            # 자유도 = n - k (여기서는 근사적으로 큰 값 사용)
            df = 1000  # 대표본 가정
            p_values = 2 * (1 - t.cdf(np.abs(t_statistics), df))
        
        return p_values
    
    def _log_hessian_statistics(self, matrix: np.ndarray, name: str):
        """Hessian 행렬 통계 로깅"""
        diag = np.diag(matrix)
        off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diag = matrix[off_diag_mask]
        
        self.logger.info(
            f"\n{'='*80}\n"
            f"{name} 통계\n"
            f"{'='*80}\n"
            f"  Shape: {matrix.shape}\n"
            f"  대각 원소:\n"
            f"    - 범위: [{np.min(diag):.6e}, {np.max(diag):.6e}]\n"
            f"    - 평균: {np.mean(diag):.6e}\n"
            f"    - 중앙값: {np.median(diag):.6e}\n"
            f"    - 음수 개수: {np.sum(diag < 0)}/{len(diag)}\n"
            f"  비대각 원소:\n"
            f"    - 범위: [{np.min(off_diag):.6e}, {np.max(off_diag):.6e}]\n"
            f"    - 평균: {np.mean(off_diag):.6e}\n"
            f"    - 절대값 평균: {np.mean(np.abs(off_diag)):.6e}\n"
            f"{'='*80}\n"
        )
    
    def get_results_summary(
        self,
        parameters: np.ndarray,
        param_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        결과 요약 DataFrame 생성
        
        Args:
            parameters: 파라미터 벡터
            param_names: 파라미터 이름 리스트
        
        Returns:
            결과 요약 DataFrame
        """
        if self.standard_errors is None:
            raise ValueError("표준오차가 계산되지 않았습니다.")
        
        n_params = len(parameters)
        
        if param_names is None:
            param_names = [f"param_{i}" for i in range(n_params)]
        
        # t-통계량 및 p-값 계산
        t_stats = self.compute_t_statistics(parameters)
        p_values = self.compute_p_values(t_stats)
        
        # DataFrame 생성
        results_df = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': parameters,
            'Std.Error': self.standard_errors,
            't-statistic': t_stats,
            'p-value': p_values,
            'Significant': p_values < 0.05
        })
        
        # Robust SE가 있으면 추가
        if self.robust_standard_errors is not None:
            robust_t = parameters / self.robust_standard_errors
            robust_p = self.compute_p_values(robust_t)
            
            results_df['Robust.SE'] = self.robust_standard_errors
            results_df['Robust.t'] = robust_t
            results_df['Robust.p'] = robust_p
        
        return results_df

