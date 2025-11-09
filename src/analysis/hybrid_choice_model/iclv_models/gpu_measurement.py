"""
GPU-accelerated Ordered Probit Measurement Model

CuPy를 사용하여 측정모델의 핵심 연산을 GPU에서 수행합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부 확인
try:
    import cupy as cp
    from cupyx.scipy.special import ndtr  # 표준정규분포 CDF
    GPU_AVAILABLE = True
    logger.info("✅ CuPy 로드 성공 - GPU 가속 사용 가능")
except ImportError as e:
    GPU_AVAILABLE = False
    logger.warning(f"⚠️ CuPy 로드 실패 - CPU 모드로 작동: {e}")
    cp = None
    ndtr = None


class GPUOrderedProbitMeasurement:
    """
    GPU 가속 Ordered Probit 측정모델
    
    전략: 핵심 연산(정규분포 CDF)만 GPU 사용
    - 간단한 구현
    - 최소한의 코드 변경
    - CPU-GPU 전송 최소화
    """
    
    def __init__(self, config, use_gpu: bool = True):
        """
        초기화
        
        Args:
            config: MeasurementConfig 객체
            use_gpu: GPU 사용 여부 (기본값: True)
        """
        self.config = config
        self.n_indicators = len(config.indicators)
        self.n_categories = config.n_categories
        self.n_thresholds = config.n_categories - 1
        
        # GPU 사용 설정
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            self.xp = cp  # CuPy
            self.norm_cdf = ndtr  # GPU CDF
            logger.info(f"🚀 GPU 모드 활성화: {self.n_indicators}개 지표")
        else:
            self.xp = np  # NumPy
            from scipy.stats import norm
            self.norm_cdf = norm.cdf  # CPU CDF
            logger.info(f"💻 CPU 모드: {self.n_indicators}개 지표")
        
        self.fitted = False
    
    def _compute_ordered_probit_prob(self, y_value: int, linear_pred: float, 
                                     tau_indicator: np.ndarray) -> float:
        """
        Ordered Probit 확률 계산 (GPU 가속)
        
        P(Y = k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
        
        Args:
            y_value: 관측값 (1, 2, 3, 4, 5)
            linear_pred: 선형 예측값 (ζ * LV)
            tau_indicator: 해당 지표의 임계값 (n_thresholds,)
        
        Returns:
            확률값
        """
        k = int(y_value) - 1  # 0-based index
        
        # 상한/하한 계산
        if k == 0:
            # 첫 번째 범주: P(Y=1) = Φ(τ_1 - V)
            upper = tau_indicator[0] - linear_pred
            if self.use_gpu:
                upper_gpu = self.xp.array(upper)
                prob = float(self.norm_cdf(upper_gpu).get())
            else:
                prob = self.norm_cdf(upper)
        elif k == self.n_categories - 1:
            # 마지막 범주: P(Y=5) = 1 - Φ(τ_4 - V)
            lower = tau_indicator[-1] - linear_pred
            if self.use_gpu:
                lower_gpu = self.xp.array(lower)
                prob = float(1.0 - self.norm_cdf(lower_gpu).get())
            else:
                prob = 1.0 - self.norm_cdf(lower)
        else:
            # 중간 범주: P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
            upper = tau_indicator[k] - linear_pred
            lower = tau_indicator[k-1] - linear_pred
            if self.use_gpu:
                bounds_gpu = self.xp.array([upper, lower])
                cdf_vals = self.norm_cdf(bounds_gpu)
                prob = float((cdf_vals[0] - cdf_vals[1]).get())
            else:
                prob = self.norm_cdf(upper) - self.norm_cdf(lower)
        
        return max(prob, 1e-10)
    
    def log_likelihood(self, data: pd.DataFrame, latent_var: float,
                      params: Dict[str, np.ndarray]) -> float:
        """
        로그우도 계산 (GPU 가속)
        
        Args:
            data: 관측지표 데이터
            latent_var: 잠재변수 값
            params: 파라미터 딕셔너리
                - 'zeta': 요인적재량 (n_indicators,)
                - 'tau': 임계값 (n_indicators, n_thresholds)
        
        Returns:
            로그우도 값
        """
        zeta = params['zeta']
        tau = params['tau']
        
        total_ll = 0.0
        first_row = data.iloc[0]
        
        # 각 지표에 대해
        for i, indicator in enumerate(self.config.indicators):
            if indicator not in first_row.index:
                continue
            
            y_value = first_row[indicator]
            
            if pd.isna(y_value):
                continue
            
            # 선형 예측: V = ζ * LV
            linear_pred = zeta[i] * latent_var
            
            # Ordered Probit 확률 (GPU 가속)
            prob = self._compute_ordered_probit_prob(
                y_value, linear_pred, tau[i]
            )
            
            # 로그우도 누적
            total_ll += np.log(prob)
        
        return total_ll
    
    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """파라미터 초기화"""
        params = {
            'zeta': np.ones(self.n_indicators),
            'tau': np.zeros((self.n_indicators, self.n_thresholds))
        }
        
        # 임계값 초기화 (균등 간격)
        for i in range(self.n_indicators):
            params['tau'][i] = np.linspace(-2, 2, self.n_thresholds)
        
        return params


class GPUBatchOrderedProbitMeasurement:
    """
    GPU 배치 처리 Ordered Probit 측정모델
    
    전략: 여러 개인을 한번에 GPU로 처리
    - 최대 GPU 활용
    - CPU-GPU 전송 최소화
    - 높은 속도 향상
    """
    
    def __init__(self, config, use_gpu: bool = True):
        """
        초기화
        
        Args:
            config: MeasurementConfig 객체
            use_gpu: GPU 사용 여부
        """
        self.config = config
        self.n_indicators = len(config.indicators)
        self.n_categories = config.n_categories
        self.n_thresholds = config.n_categories - 1
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            self.xp = cp
            logger.info(f"🚀 GPU 배치 모드: {self.n_indicators}개 지표")
        else:
            self.xp = np
            logger.info(f"💻 CPU 배치 모드: {self.n_indicators}개 지표")
        
        self.fitted = False
    
    def log_likelihood_batch(self, data_batch: np.ndarray, 
                            latent_vars_batch: np.ndarray,
                            params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        배치 로그우도 계산 (GPU 가속)
        
        Args:
            data_batch: (n_persons, n_indicators) - 관측값
            latent_vars_batch: (n_persons,) - 잠재변수
            params: 파라미터
        
        Returns:
            (n_persons,) - 각 개인의 로그우도
        """
        if not self.use_gpu:
            # CPU 모드: 순차 처리
            return self._log_likelihood_batch_cpu(
                data_batch, latent_vars_batch, params
            )
        
        # GPU로 전송
        data_gpu = self.xp.array(data_batch)  # (n_persons, n_indicators)
        lv_gpu = self.xp.array(latent_vars_batch)  # (n_persons,)
        zeta_gpu = self.xp.array(params['zeta'])  # (n_indicators,)
        tau_gpu = self.xp.array(params['tau'])  # (n_indicators, n_thresholds)
        
        n_persons = data_gpu.shape[0]
        
        # 선형 예측: (n_persons, n_indicators)
        linear_pred = self.xp.outer(lv_gpu, zeta_gpu)
        
        # 로그우도 초기화
        ll_batch = self.xp.zeros(n_persons)
        
        # 각 지표에 대해
        for i in range(self.n_indicators):
            y_values = data_gpu[:, i]  # (n_persons,)
            linear_pred_i = linear_pred[:, i]  # (n_persons,)
            tau_i = tau_gpu[i]  # (n_thresholds,)
            
            # 각 범주에 대해 확률 계산
            for k in range(self.n_categories):
                mask = (y_values == (k + 1))  # 해당 범주인 개인들
                
                if self.xp.sum(mask) == 0:
                    continue
                
                # 확률 계산
                if k == 0:
                    upper = tau_i[0] - linear_pred_i[mask]
                    prob = ndtr(upper)
                elif k == self.n_categories - 1:
                    lower = tau_i[-1] - linear_pred_i[mask]
                    prob = 1.0 - ndtr(lower)
                else:
                    upper = tau_i[k] - linear_pred_i[mask]
                    lower = tau_i[k-1] - linear_pred_i[mask]
                    prob = ndtr(upper) - ndtr(lower)
                
                # 로그우도 누적
                ll_batch[mask] += self.xp.log(self.xp.maximum(prob, 1e-10))
        
        # CPU로 반환
        return self.xp.asnumpy(ll_batch)
    
    def _log_likelihood_batch_cpu(self, data_batch: np.ndarray,
                                   latent_vars_batch: np.ndarray,
                                   params: Dict[str, np.ndarray]) -> np.ndarray:
        """CPU 배치 처리"""
        from scipy.stats import norm
        
        n_persons = data_batch.shape[0]
        ll_batch = np.zeros(n_persons)
        
        zeta = params['zeta']
        tau = params['tau']
        
        for person_idx in range(n_persons):
            lv = latent_vars_batch[person_idx]
            
            for i in range(self.n_indicators):
                y_value = data_batch[person_idx, i]
                
                if np.isnan(y_value):
                    continue
                
                linear_pred = zeta[i] * lv
                k = int(y_value) - 1
                
                if k == 0:
                    prob = norm.cdf(tau[i, 0] - linear_pred)
                elif k == self.n_categories - 1:
                    prob = 1.0 - norm.cdf(tau[i, -1] - linear_pred)
                else:
                    prob = norm.cdf(tau[i, k] - linear_pred) - \
                           norm.cdf(tau[i, k-1] - linear_pred)
                
                ll_batch[person_idx] += np.log(max(prob, 1e-10))
        
        return ll_batch
    
    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """파라미터 초기화"""
        params = {
            'zeta': np.ones(self.n_indicators),
            'tau': np.zeros((self.n_indicators, self.n_thresholds))
        }
        
        for i in range(self.n_indicators):
            params['tau'][i] = np.linspace(-2, 2, self.n_thresholds)
        
        return params

