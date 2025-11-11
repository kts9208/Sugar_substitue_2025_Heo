"""
GPU-Accelerated Ordered Probit Measurement Model for ICLV

CuPy를 사용하여 GPU에서 측정모델 우도를 계산합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부 확인
try:
    import os
    # CUDA 경로 설정 (Windows)
    cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0')
    if os.path.exists(cuda_path):
        cuda_bin = os.path.join(cuda_path, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')

    import cupy as cp
    from cupyx.scipy.special import ndtr  # 표준정규 CDF

    # GPU 실제 사용 가능 여부 테스트
    try:
        cp.cuda.Device(0).use()
        _ = cp.array([1, 2, 3])  # 간단한 테스트
        GPU_AVAILABLE = True
        logger.info("✅ CuPy 로드 성공 - GPU 가속 사용 가능")
    except Exception as e:
        GPU_AVAILABLE = False
        logger.warning(f"⚠️ GPU 초기화 실패 - CPU 모드로 작동: {e}")
        cp = None
        ndtr = None

except ImportError as e:
    GPU_AVAILABLE = False
    logger.warning(f"⚠️ CuPy 미설치 - CPU 모드로 작동: {e}")
    cp = None
    ndtr = None
except Exception as e:
    GPU_AVAILABLE = False
    logger.warning(f"⚠️ GPU 로드 실패 - CPU 모드로 작동: {e}")
    cp = None
    ndtr = None


class GPUOrderedProbitMeasurement:
    """
    GPU 가속 Ordered Probit 측정모델
    
    CuPy를 사용하여 정규분포 CDF 계산을 GPU에서 수행합니다.
    
    Model:
        P(Y_i = k) = Φ(τ_k - ζ*LV) - Φ(τ_{k-1} - ζ*LV)
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
        self.measurement_method = 'ordered_probit'  # ✅ 측정 방법 명시

        # GPU 사용 설정
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            self.xp = cp
            logger.info(f"🚀 GPU 모드 활성화: {self.n_indicators}개 지표")
        else:
            self.xp = np
            if use_gpu and not GPU_AVAILABLE:
                logger.warning("⚠️ GPU 요청되었으나 CuPy 미설치 - CPU 모드 사용")
            else:
                logger.info(f"💻 CPU 모드: {self.n_indicators}개 지표")
        
        self.zeta = None
        self.tau = None
        self.fitted = False
    
    def _norm_cdf(self, x):
        """표준정규 누적분포함수 (GPU/CPU 자동 선택)"""
        if self.use_gpu:
            return ndtr(x)
        else:
            from scipy.stats import norm
            return norm.cdf(x)
    
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
        
        # GPU로 전송 (필요 시)
        if self.use_gpu:
            zeta = cp.asarray(zeta)
            tau = cp.asarray(tau)
        
        total_ll = 0.0
        first_row = data.iloc[0]

        # 각 지표에 대해
        for i, indicator in enumerate(self.config.indicators):
            if indicator not in first_row.index:
                continue

            # NaN 값 처리
            if pd.isna(first_row[indicator]):
                continue

            y_obs = int(first_row[indicator])

            if y_obs < 1 or y_obs > self.n_categories:
                continue
            
            # 선형 예측: V = ζ * LV
            linear_pred = zeta[i] * latent_var
            
            # 임계값
            tau_i = tau[i]
            
            # 확률 계산: P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
            if y_obs == 1:
                # P(Y=1) = Φ(τ_1 - V)
                upper = tau_i[0] - linear_pred
                prob = self._norm_cdf(upper)
            elif y_obs == self.n_categories:
                # P(Y=K) = 1 - Φ(τ_{K-1} - V)
                lower = tau_i[-1] - linear_pred
                prob = 1.0 - self._norm_cdf(lower)
            else:
                # P(Y=k) = Φ(τ_k - V) - Φ(τ_{k-1} - V)
                upper = tau_i[y_obs - 1] - linear_pred
                lower = tau_i[y_obs - 2] - linear_pred
                prob = self._norm_cdf(upper) - self._norm_cdf(lower)
            
            # GPU에서 CPU로 변환 (필요 시)
            if self.use_gpu:
                prob = float(cp.asnumpy(prob))
            
            # 로그우도 누적
            prob = max(prob, 1e-10)
            total_ll += np.log(prob)
        
        return total_ll
    
    def log_likelihood_batch(self, data_batch: np.ndarray, latent_vars: np.ndarray,
                            params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        배치 로그우도 계산 (GPU 최적화)
        
        여러 개인/draws를 한번에 처리하여 GPU 효율 극대화
        
        Args:
            data_batch: (n_batch, n_indicators) 관측 데이터
            latent_vars: (n_batch,) 잠재변수 값들
            params: 파라미터 딕셔너리
        
        Returns:
            (n_batch,) 로그우도 배열
        """
        if not self.use_gpu:
            # CPU 모드: 순차 처리
            lls = []
            for i in range(len(latent_vars)):
                data_dict = {ind: data_batch[i, j] 
                           for j, ind in enumerate(self.config.indicators)}
                data_df = pd.DataFrame([data_dict])
                ll = self.log_likelihood(data_df, latent_vars[i], params)
                lls.append(ll)
            return np.array(lls)
        
        # GPU 모드: 배치 처리
        zeta = cp.asarray(params['zeta'])  # (n_indicators,)
        tau = cp.asarray(params['tau'])    # (n_indicators, n_thresholds)
        data_gpu = cp.asarray(data_batch)  # (n_batch, n_indicators)
        lv_gpu = cp.asarray(latent_vars)   # (n_batch,)
        
        n_batch = len(latent_vars)
        ll_batch = cp.zeros(n_batch)
        
        # 각 지표에 대해
        for i in range(self.n_indicators):
            y_obs = data_gpu[:, i].astype(int)  # (n_batch,)
            
            # 선형 예측: V = ζ * LV
            linear_pred = zeta[i] * lv_gpu  # (n_batch,)
            
            # 임계값
            tau_i = tau[i]  # (n_thresholds,)
            
            # 각 범주에 대해 확률 계산
            probs = cp.zeros(n_batch)
            
            for k in range(1, self.n_categories + 1):
                mask = (y_obs == k)
                if not cp.any(mask):
                    continue
                
                if k == 1:
                    upper = tau_i[0] - linear_pred[mask]
                    probs[mask] = ndtr(upper)
                elif k == self.n_categories:
                    lower = tau_i[-1] - linear_pred[mask]
                    probs[mask] = 1.0 - ndtr(lower)
                else:
                    upper = tau_i[k - 1] - linear_pred[mask]
                    lower = tau_i[k - 2] - linear_pred[mask]
                    probs[mask] = ndtr(upper) - ndtr(lower)
            
            # 로그우도 누적
            probs = cp.maximum(probs, 1e-10)
            ll_batch += cp.log(probs)
        
        # CPU로 반환
        return cp.asnumpy(ll_batch)
    
    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """파라미터 초기화"""
        zeta = np.ones(self.n_indicators)
        
        tau = np.zeros((self.n_indicators, self.n_thresholds))
        for i in range(self.n_indicators):
            tau[i] = np.linspace(-2, 2, self.n_thresholds)
        
        return {'zeta': zeta, 'tau': tau}
    
    def get_n_parameters(self) -> int:
        """총 파라미터 수"""
        return self.n_indicators + (self.n_indicators * self.n_thresholds)


class GPUMultiLatentMeasurement:
    """
    다중 잠재변수 측정모델 (GPU 가속)

    여러 잠재변수의 측정모델을 GPU에서 처리합니다.
    """

    def __init__(self, measurement_configs: Dict, use_gpu: bool = True):
        """
        초기화

        Args:
            measurement_configs: {lv_name: MeasurementConfig} 딕셔너리
            use_gpu: GPU 사용 여부
        """
        self.configs = measurement_configs
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # 각 잠재변수별 측정모델 생성
        self.models = {}
        for lv_name, config in measurement_configs.items():
            # ✅ measurement_method에 따라 적절한 모델 선택
            method = getattr(config, 'measurement_method', 'ordered_probit')

            if method == 'continuous_linear':
                self.models[lv_name] = GPUContinuousLinearMeasurement(config, use_gpu)
            elif method == 'ordered_probit':
                self.models[lv_name] = GPUOrderedProbitMeasurement(config, use_gpu)
            else:
                raise ValueError(f"지원하지 않는 측정 방법: {method}")

        if self.use_gpu:
            logger.info(f"🚀 GPU 다중 측정모델: {len(self.models)}개 잠재변수")
        else:
            logger.info(f"💻 CPU 다중 측정모델: {len(self.models)}개 잠재변수")

    def log_likelihood(self, data: pd.DataFrame, latent_vars: Dict[str, float],
                      params: Dict[str, Dict]) -> float:
        """
        전체 로그우도 계산

        Args:
            data: 관측 데이터
            latent_vars: {lv_name: lv_value} 잠재변수 값들
            params: {lv_name: {'zeta': ..., 'tau': ...}} 파라미터

        Returns:
            전체 로그우도
        """
        total_ll = 0.0

        for lv_name, model in self.models.items():
            if lv_name not in latent_vars or lv_name not in params:
                continue

            ll = model.log_likelihood(data, latent_vars[lv_name], params[lv_name])
            total_ll += ll

        return total_ll

    def log_likelihood_batch(self, data_batch: Dict[str, np.ndarray],
                            latent_vars_batch: Dict[str, np.ndarray],
                            params: Dict[str, Dict]) -> np.ndarray:
        """
        배치 로그우도 계산 (GPU 최적화)

        모든 개인 × draws를 한 번에 처리하여 GPU 효율 극대화

        Args:
            data_batch: {lv_name: (n_batch, n_indicators)} 관측 데이터
            latent_vars_batch: {lv_name: (n_batch,)} 잠재변수 값들
            params: {lv_name: {'zeta': ..., 'tau': ...}} 파라미터

        Returns:
            (n_batch,) 로그우도 배열
        """
        # 첫 번째 LV로 배치 크기 확인
        first_lv = list(latent_vars_batch.keys())[0]
        n_batch = len(latent_vars_batch[first_lv])

        if self.use_gpu:
            total_ll = cp.zeros(n_batch)
        else:
            total_ll = np.zeros(n_batch)

        # 각 잠재변수별 측정모델 우도 계산
        for lv_idx, (lv_name, model) in enumerate(self.models.items()):
            if lv_name not in latent_vars_batch or lv_name not in params:
                continue

            if lv_name not in data_batch:
                continue

            # 첫 번째 LV에 대해서만 파라미터 로깅 (디버깅용)
            # if lv_idx == 0:
            #     print(f"  [GPU 측정모델 내부] {lv_name} zeta (처음 3개): {params[lv_name]['zeta'][:3]}")
            #     print(f"  [GPU 측정모델 내부] {lv_name} tau[0] (처음 3개): {params[lv_name]['tau'][0][:3]}")

            # 배치 우도 계산
            ll_batch = model.log_likelihood_batch(
                data_batch[lv_name],
                latent_vars_batch[lv_name],
                params[lv_name]
            )

            # GPU 모드일 때 NumPy 배열을 CuPy로 변환
            if self.use_gpu and isinstance(ll_batch, np.ndarray):
                ll_batch = cp.asarray(ll_batch)

            total_ll += ll_batch

        # GPU에서 CPU로 변환
        if self.use_gpu:
            total_ll = cp.asnumpy(total_ll)

        return total_ll

    def initialize_parameters(self) -> Dict[str, Dict]:
        """모든 측정모델 파라미터 초기화"""
        params = {}
        for lv_name, model in self.models.items():
            params[lv_name] = model.initialize_parameters()
        return params

    def log_likelihood_batch_draws(self, ind_data: pd.DataFrame,
                                    lvs_list: list,
                                    params: Dict[str, Dict]) -> list:
        """
        개인의 여러 draws에 대한 측정모델 우도 계산 (GPU 배치)

        Args:
            ind_data: 개인 데이터 (1행)
            lvs_list: 각 draw의 잠재변수 값 리스트 [{lv_name: value}, ...]
            params: {lv_name: {'zeta': ..., 'tau': ...}} 파라미터

        Returns:
            각 draw의 로그우도 리스트
        """
        n_draws = len(lvs_list)

        # 배치 데이터 구성
        data_batch = {}
        latent_vars_batch = {}

        for lv_name, model in self.models.items():
            # 지표 데이터 (모든 draws에 동일)
            indicators = model.config.indicators
            ind_values = ind_data[indicators].iloc[0].values
            data_batch[lv_name] = np.tile(ind_values, (n_draws, 1))  # (n_draws, n_indicators)

            # 잠재변수 값 (각 draw마다 다름)
            lv_values = np.array([lvs[lv_name] for lvs in lvs_list])
            latent_vars_batch[lv_name] = lv_values  # (n_draws,)

        # 배치 우도 계산
        ll_batch = self.log_likelihood_batch(data_batch, latent_vars_batch, params)

        return ll_batch.tolist()

    def get_n_parameters(self) -> int:
        """총 파라미터 수"""
        total = 0
        for model in self.models.values():
            n_indicators = model.n_indicators
            n_thresholds = model.n_thresholds
            total += n_indicators + (n_indicators * n_thresholds)
        return total


class GPUContinuousLinearMeasurement:
    """
    GPU 가속 연속형 선형 측정모델

    CuPy를 사용하여 GPU에서 로그우도를 계산합니다.

    Model:
        Y_i = ζ_i * LV + ε_i
        ε_i ~ N(0, σ²_i)
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
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.measurement_method = 'continuous_linear'  # ✅ 측정 방법 명시

        self.zeta = None
        self.sigma_sq = None
        self.fitted = False

        if self.use_gpu:
            self.xp = cp
            logger.info(f"🚀 GPU ContinuousLinear: {self.n_indicators}개 지표")
        else:
            self.xp = np
            logger.info(f"💻 CPU ContinuousLinear: {self.n_indicators}개 지표")

    def log_likelihood(self, data: pd.DataFrame, latent_var: float,
                      params: Dict[str, np.ndarray]) -> float:
        """
        로그우도 계산 (GPU 가속)

        Args:
            data: 관측지표 데이터
            latent_var: 잠재변수 값
            params: {'zeta': ..., 'sigma_sq': ...}

        Returns:
            로그우도 값
        """
        zeta = params['zeta']
        sigma_sq = params['sigma_sq']

        if self.use_gpu:
            # GPU 계산
            zeta_gpu = cp.asarray(zeta)
            sigma_sq_gpu = cp.asarray(sigma_sq)
            latent_var_gpu = cp.asarray(latent_var)
        else:
            zeta_gpu = zeta
            sigma_sq_gpu = sigma_sq
            latent_var_gpu = latent_var

        total_ll = 0.0
        first_row = data.iloc[0]

        for i, indicator in enumerate(self.config.indicators):
            if indicator not in first_row.index:
                continue

            y_obs = first_row[indicator]

            if pd.isna(y_obs):
                continue

            # 예측값
            if self.use_gpu:
                y_pred = float(zeta_gpu[i] * latent_var_gpu)
            else:
                y_pred = zeta_gpu[i] * latent_var_gpu

            # 잔차
            residual = y_obs - y_pred

            # 정규분포 로그우도
            if self.use_gpu:
                ll_i = -0.5 * cp.log(2 * cp.pi * sigma_sq_gpu[i])
                ll_i += -0.5 * (residual ** 2) / sigma_sq_gpu[i]
                ll_i = float(cp.asnumpy(ll_i))
            else:
                ll_i = -0.5 * np.log(2 * np.pi * sigma_sq_gpu[i])
                ll_i += -0.5 * (residual ** 2) / sigma_sq_gpu[i]

            total_ll += ll_i

        return total_ll

    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        """파라미터 초기화"""
        params = {}

        # 요인적재량
        zeta = np.ones(self.n_indicators)
        if self.config.fix_first_loading:
            zeta[0] = 1.0

        params['zeta'] = zeta

        # 오차분산
        sigma_sq = np.ones(self.n_indicators) * self.config.initial_error_variance
        params['sigma_sq'] = sigma_sq

        return params

    def log_likelihood_batch(self, data_batch: np.ndarray, latent_vars: np.ndarray,
                            params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        배치 로그우도 계산 (GPU 최적화)

        여러 개인/draws를 한번에 처리하여 GPU 효율 극대화

        Args:
            data_batch: (n_batch, n_indicators) 관측 데이터
            latent_vars: (n_batch,) 잠재변수 값들
            params: {'zeta': ..., 'sigma_sq': ...}

        Returns:
            (n_batch,) 로그우도 배열
        """
        if not self.use_gpu:
            # CPU 모드: 순차 처리
            lls = []
            for i in range(len(latent_vars)):
                data_dict = {ind: data_batch[i, j]
                           for j, ind in enumerate(self.config.indicators)}
                data_df = pd.DataFrame([data_dict])
                ll = self.log_likelihood(data_df, latent_vars[i], params)
                lls.append(ll)
            return np.array(lls)

        # GPU 모드: 배치 처리
        zeta = cp.asarray(params['zeta'])      # (n_indicators,)
        sigma_sq = cp.asarray(params['sigma_sq'])  # (n_indicators,)
        data_gpu = cp.asarray(data_batch)      # (n_batch, n_indicators)
        lv_gpu = cp.asarray(latent_vars)       # (n_batch,)

        n_batch = len(latent_vars)
        ll_batch = cp.zeros(n_batch)

        # 각 지표에 대해
        for i in range(self.n_indicators):
            y_obs = data_gpu[:, i]  # (n_batch,)

            # 예측값: Y_pred = ζ * LV
            y_pred = zeta[i] * lv_gpu  # (n_batch,)

            # 잔차
            residual = y_obs - y_pred  # (n_batch,)

            # 정규분포 로그우도
            # log p(y|LV) = -0.5 * log(2π * σ²) - 0.5 * (y - ζ*LV)² / σ²
            ll_i = -0.5 * cp.log(2 * cp.pi * sigma_sq[i])  # 스칼라
            ll_i = ll_i - 0.5 * (residual ** 2) / sigma_sq[i]  # (n_batch,)

            ll_batch = ll_batch + ll_i  # (n_batch,)

        # GPU에서 CPU로 변환
        return cp.asnumpy(ll_batch)

    def get_n_parameters(self) -> int:
        """파라미터 수 반환"""
        n_params = 0

        if self.config.fix_first_loading:
            n_params += self.n_indicators - 1
        else:
            n_params += self.n_indicators

        if not self.config.fix_error_variance:
            n_params += self.n_indicators

        return n_params

