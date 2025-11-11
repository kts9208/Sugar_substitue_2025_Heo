"""
GPU 배치 처리 ICLV 동시추정

SimultaneousEstimator를 상속하여 GPU 배치 처리로 가속합니다.
개인별 우도 계산 부분만 GPU 배치로 오버라이드합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.special import logsumexp
import logging
import gc

from .simultaneous_estimator_fixed import SimultaneousEstimator
from .gpu_measurement_equations import GPUMultiLatentMeasurement
from . import gpu_batch_utils
from scipy.stats import qmc, norm
from .memory_monitor import MemoryMonitor, cleanup_arrays

logger = logging.getLogger(__name__)


class MultiDimensionalHaltonDrawGenerator:
    """
    다중 차원 Halton 시퀀스 생성기

    다중 잠재변수 모델을 위한 다차원 Halton draws를 생성합니다.
    """

    def __init__(self, n_draws: int, n_individuals: int, n_dimensions: int,
                 scramble: bool = True, seed: Optional[int] = None):
        """
        Args:
            n_draws: 개인당 draw 수
            n_individuals: 개인 수
            n_dimensions: 차원 수 (잠재변수 개수)
            scramble: 스크램블 여부
            seed: 난수 시드
        """
        self.n_draws = n_draws
        self.n_individuals = n_individuals
        self.n_dimensions = n_dimensions
        self.scramble = scramble
        self.seed = seed

        self.draws = None
        self._generate_draws()

    def _generate_draws(self):
        """다차원 Halton 시퀀스 생성"""
        logger.info(f"다차원 Halton draws 생성: {self.n_individuals} 개인 × {self.n_draws} draws × {self.n_dimensions} 차원")

        # scipy의 Halton 시퀀스 생성기 사용 (다차원)
        sampler = qmc.Halton(d=self.n_dimensions, scramble=self.scramble, seed=self.seed)

        # 균등분포 [0,1] 샘플 생성
        # (n_individuals * n_draws, n_dimensions)
        uniform_draws = sampler.random(n=self.n_individuals * self.n_draws)

        # 표준정규분포로 변환 (역누적분포함수)
        normal_draws = norm.ppf(uniform_draws)

        # (n_individuals, n_draws, n_dimensions) 형태로 재구성
        self.draws = normal_draws.reshape(self.n_individuals, self.n_draws, self.n_dimensions)

        logger.info(f"다차원 Halton draws 생성 완료: shape={self.draws.shape}")

    def get_draws(self) -> np.ndarray:
        """생성된 draws 반환"""
        return self.draws


class GPUBatchEstimator(SimultaneousEstimator):
    """
    GPU 배치 처리 ICLV 동시추정
    
    SimultaneousEstimator를 상속하여 GPU 배치 처리로 가속합니다.
    개인별 우도 계산 부분만 GPU 배치로 오버라이드합니다.
    """
    
    def __init__(self, config, use_gpu: bool = True,
                 memory_monitor_cpu_threshold_mb: float = 2000,
                 memory_monitor_gpu_threshold_mb: float = 1500):
        """
        Args:
            config: MultiLatentConfig 또는 ICLVConfig
            use_gpu: GPU 사용 여부
            memory_monitor_cpu_threshold_mb: CPU 메모리 임계값 (MB)
            memory_monitor_gpu_threshold_mb: GPU 메모리 임계값 (MB)
        """
        super().__init__(config)
        self.use_gpu = use_gpu and gpu_batch_utils.CUPY_AVAILABLE
        self.gpu_measurement_model = None

        # 메모리 모니터 임계값 저장 (나중에 초기화)
        self.memory_monitor_cpu_threshold_mb = memory_monitor_cpu_threshold_mb
        self.memory_monitor_gpu_threshold_mb = memory_monitor_gpu_threshold_mb
        self.memory_monitor = None  # estimate()에서 초기화

        if self.use_gpu:
            logger.info("GPU 배치 처리 활성화")
        else:
            logger.info("GPU 배치 처리 비활성화 (CPU 모드)")
    
    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None) -> Dict:
        """
        ICLV 모델 추정 (GPU 배치 가속)

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 인스턴스
            structural_model: 구조모델 인스턴스
            choice_model: 선택모델 인스턴스
            log_file: 로그 파일 경로

        Returns:
            추정 결과 딕셔너리
        """
        # GPU 측정모델 생성
        if self.use_gpu:
            if hasattr(self.config, 'measurement_configs'):
                # 다중 잠재변수
                self.gpu_measurement_model = GPUMultiLatentMeasurement(
                    self.config.measurement_configs,
                    use_gpu=True
                )
                logger.info("GPU 측정모델 생성 완료 (다중 잠재변수)")

                # 다중 차원 Halton draws 생성을 위해 structural_model 저장
                self.structural_model_ref = structural_model
                self.use_multi_dimensional_draws = True
            else:
                # 단일 잠재변수 - GPU 배치 처리 미지원
                logger.warning("단일 잠재변수는 GPU 배치 처리 미지원. CPU 모드로 전환.")
                self.use_gpu = False
                self.use_multi_dimensional_draws = False
        else:
            self.use_multi_dimensional_draws = False

        # 부모 클래스의 estimate 호출 전에 데이터 저장
        self.data = data

        # 메모리 모니터 초기화 (iteration_logger 사용 가능한 시점)
        # 부모 클래스의 estimate()에서 iteration_logger가 설정되므로,
        # 여기서는 임시로 logger 사용
        if self.memory_monitor is None:
            self.memory_monitor = MemoryMonitor(
                cpu_threshold_mb=self.memory_monitor_cpu_threshold_mb,
                gpu_threshold_mb=self.memory_monitor_gpu_threshold_mb,
                auto_cleanup=True,
                logger=logger  # 임시로 모듈 logger 사용
            )

        # 다중 차원 Halton draws 생성 (부모 클래스 호출 전에)
        if self.use_multi_dimensional_draws:
            n_individuals = data[self.config.individual_id_column].nunique()
            n_dimensions = structural_model.n_exo + 1  # 외생 LV + 내생 LV 오차항

            logger.info(f"다차원 Halton draws 생성 시작... (n_draws={self.config.estimation.n_draws}, n_individuals={n_individuals}, n_dimensions={n_dimensions})")

            self.halton_generator = MultiDimensionalHaltonDrawGenerator(
                n_draws=self.config.estimation.n_draws,
                n_individuals=n_individuals,
                n_dimensions=n_dimensions,
                scramble=self.config.estimation.scramble_halton
            )

            logger.info("다차원 Halton draws 생성 완료")

        # 부모 클래스의 estimate 호출
        return super().estimate(data, measurement_model, structural_model, choice_model, log_file)
    
    def _joint_log_likelihood(self, params: np.ndarray,
                             measurement_model,
                             structural_model,
                             choice_model) -> float:
        """
        결합 로그우도 계산 (메모리 모니터링 추가)

        부모 클래스의 _joint_log_likelihood를 오버라이드하여
        Halton draws 가져오기 전후 메모리 로그를 추가합니다.
        """
        # 현재 iteration 번호 저장 (개인별 우도 계산 로그에 사용)
        if not hasattr(self, '_current_iteration'):
            self._current_iteration = 0
        self._current_iteration += 1

        # 각 iteration 시작 시 개인별 카운터 리셋
        self._individual_likelihood_count = 0

        # 파라미터 분해
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        # 메모리 체크 (Halton draws 가져오기 전)
        if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
            self.memory_monitor.log_memory_stats(f"Halton draws 가져오기 전 (Iter {self._current_iteration})")

        draws = self.halton_generator.get_draws()

        # 메모리 체크 (Halton draws 가져온 후)
        if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
            self.memory_monitor.log_memory_stats(f"Halton draws 가져온 후 (Iter {self._current_iteration})")

        individual_ids = self.data[self.config.individual_id_column].unique()

        # 순차처리 (GPU 배치는 _compute_individual_likelihood에서 처리)
        total_ll = 0.0
        for i, ind_id in enumerate(individual_ids):
            ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
            ind_draws = draws[i, :]

            person_ll = self._compute_individual_likelihood(
                ind_id, ind_data, ind_draws, param_dict,
                measurement_model, structural_model, choice_model
            )
            total_ll += person_ll

        return total_ll

    def _compute_individual_likelihood(self, ind_id, ind_data, ind_draws,
                                       param_dict, measurement_model,
                                       structural_model, choice_model) -> float:
        """
        개인별 우도 계산 (GPU 배치 가속 버전)

        SimultaneousEstimator의 메서드를 오버라이드하여 GPU 배치 처리를 사용합니다.
        """
        n_draws = len(ind_draws)

        # 메모리 체크 (우도 계산 전)
        mem_info = self.memory_monitor.check_and_cleanup(f"우도 계산 - 개인 {ind_id}")

        if self.use_gpu and self.gpu_measurement_model is not None:
            # GPU 배치 처리
            draw_lls = self._compute_draws_batch_gpu(
                ind_data, ind_draws, param_dict,
                structural_model, choice_model
            )
        else:
            # CPU 순차 처리 (부모 클래스와 동일)
            draw_lls = []
            
            for j in range(n_draws):
                draw = ind_draws[j]
                
                # 구조모델: LV 예측
                if hasattr(structural_model, 'endogenous_lv'):
                    # 다중 잠재변수
                    n_exo = structural_model.n_exo
                    exo_draws = draw[:n_exo]
                    endo_draw = draw[n_exo]
                    lv = structural_model.predict(ind_data, exo_draws, param_dict['structural'], endo_draw)
                else:
                    # 단일 잠재변수
                    lv = structural_model.predict(ind_data, param_dict['structural'], draw)
                
                # 측정모델 우도
                ll_measurement = measurement_model.log_likelihood(
                    ind_data, lv, param_dict['measurement']
                )
                
                # 선택모델 우도 (Panel Product)
                choice_set_lls = []
                for idx in range(len(ind_data)):
                    ll_choice_t = choice_model.log_likelihood(
                        ind_data.iloc[idx:idx+1],
                        lv,
                        param_dict['choice']
                    )
                    choice_set_lls.append(ll_choice_t)
                
                ll_choice = sum(choice_set_lls)
                
                # 구조모델 우도
                if hasattr(structural_model, 'endogenous_lv'):
                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, param_dict['structural'], draw
                    )
                else:
                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, param_dict['structural'], draw
                    )
                
                # 결합 로그우도
                draw_ll = ll_measurement + ll_choice + ll_structural
                
                if not np.isfinite(draw_ll):
                    draw_ll = -1e10
                
                draw_lls.append(draw_ll)
        
        # 개인 우도: log(1/R * sum(exp(draw_lls)))
        person_ll = logsumexp(draw_lls) - np.log(n_draws)
        
        return person_ll
    
    def _compute_draws_batch_gpu(self, ind_data, ind_draws, param_dict,
                                 structural_model, choice_model):
        """
        개인의 모든 draws에 대한 우도를 GPU 배치로 계산

        Args:
            ind_data: 개인 데이터
            ind_draws: 개인의 draws (n_draws, n_dimensions)
            param_dict: 파라미터 딕셔너리
            structural_model: 구조모델 인스턴스
            choice_model: 선택모델 인스턴스

        Returns:
            각 draw의 로그우도 리스트
        """
        # 메모리 체크 (GPU 배치 우도 계산 전) - 로깅 없이 임계값만 체크
        if hasattr(self, 'memory_monitor'):
            # 개인별 카운터 증가
            self._individual_likelihood_count += 1

            # 임계값 체크 및 필요시 정리 (로깅 없음)
            mem_info = self.memory_monitor.check_and_cleanup("GPU 배치 우도 계산")

        n_draws = len(ind_draws)

        # 첫 번째 개인의 첫 번째 draw에 대해서만 상세 로깅
        log_detail = not hasattr(self, '_first_draw_logged')

        # if log_detail:
        #     self.iteration_logger.info("=" * 80)
        #     self.iteration_logger.info("첫 번째 개인의 첫 번째 draw 상세 로깅")
        #     self.iteration_logger.info("=" * 80)
        #     self.iteration_logger.info(f"[파라미터 확인]")
        #     self.iteration_logger.info(f"  측정모델 zeta (health_concern 처음 3개): {param_dict['measurement']['health_concern']['zeta'][:3]}")
        #     self.iteration_logger.info(f"  구조모델 gamma_lv: {param_dict['structural']['gamma_lv']}")
        #     self.iteration_logger.info(f"  구조모델 gamma_x: {param_dict['structural']['gamma_x']}")
        #     self.iteration_logger.info(f"  선택모델 intercept: {param_dict['choice']['intercept']}")
        #     self.iteration_logger.info(f"  선택모델 beta: {param_dict['choice']['beta']}")
        #     self.iteration_logger.info(f"  선택모델 lambda: {param_dict['choice']['lambda']}")

        # 1. 모든 draws에 대한 잠재변수 예측
        lvs_list = []
        for j in range(n_draws):
            draw = ind_draws[j]

            if hasattr(structural_model, 'endogenous_lv'):
                # 다중 잠재변수
                n_exo = structural_model.n_exo
                exo_draws = draw[:n_exo]
                endo_draw = draw[n_exo]
                lv = structural_model.predict(ind_data, exo_draws, param_dict['structural'], endo_draw)

                if log_detail and j == 0:
                    self.iteration_logger.info(f"[구조모델 예측] Draw 0:")
                    self.iteration_logger.info(f"  외생 draws: {exo_draws}")
                    self.iteration_logger.info(f"  내생 draw: {endo_draw}")
                    self.iteration_logger.info(f"  예측된 LV: {lv}")
            else:
                # 단일 잠재변수
                lv = structural_model.predict(ind_data, param_dict['structural'], draw)

            lvs_list.append(lv)
        
        # 2. 측정모델 우도 (GPU 배치)
        if log_detail:
            self.iteration_logger.info("\n[측정모델 우도 계산 시작]")
            self.iteration_logger.info(f"  개인 데이터 shape: {ind_data.shape}")
            self.iteration_logger.info(f"  LV 개수: {len(lvs_list)}")

        ll_measurement_batch = gpu_batch_utils.compute_measurement_batch_gpu(
            self.gpu_measurement_model,
            ind_data,
            lvs_list,
            param_dict['measurement'],
            self.iteration_logger if log_detail else None
        )

        if log_detail:
            self.iteration_logger.info(f"  측정모델 우도 (처음 5개): {ll_measurement_batch[:5]}")
            self.iteration_logger.info(f"  측정모델 우도 범위: [{np.min(ll_measurement_batch):.2f}, {np.max(ll_measurement_batch):.2f}]")
            self.iteration_logger.info(f"  측정모델 우도 평균: {np.mean(ll_measurement_batch):.2f}")

        # 메모리 정리 (측정모델 계산 후)
        gc.collect()

        # 3. 선택모델 우도 (GPU 배치)
        if log_detail:
            self.iteration_logger.info("\n[선택모델 우도 계산 시작]")
            self.iteration_logger.info(f"  선택 상황 수: {len(ind_data)}")

        ll_choice_batch = gpu_batch_utils.compute_choice_batch_gpu(
            ind_data,
            lvs_list,
            param_dict['choice'],
            choice_model,
            self.iteration_logger if log_detail else None
        )

        if log_detail:
            self.iteration_logger.info(f"  선택모델 우도 (처음 5개): {ll_choice_batch[:5]}")
            self.iteration_logger.info(f"  선택모델 우도 범위: [{np.min(ll_choice_batch):.2f}, {np.max(ll_choice_batch):.2f}]")
            self.iteration_logger.info(f"  선택모델 우도 평균: {np.mean(ll_choice_batch):.2f}")

        # 메모리 정리 (선택모델 계산 후)
        gc.collect()

        # 4. 구조모델 우도 (GPU 배치)
        if log_detail:
            self.iteration_logger.info("\n[구조모델 우도 계산 시작]")

        ll_structural_batch = gpu_batch_utils.compute_structural_batch_gpu(
            ind_data,
            lvs_list,
            param_dict['structural'],
            ind_draws,
            structural_model,
            self.iteration_logger if log_detail else None
        )

        if log_detail:
            self.iteration_logger.info(f"  구조모델 우도 (처음 5개): {ll_structural_batch[:5]}")
            self.iteration_logger.info(f"  구조모델 우도 범위: [{np.min(ll_structural_batch):.2f}, {np.max(ll_structural_batch):.2f}]")
            self.iteration_logger.info(f"  구조모델 우도 평균: {np.mean(ll_structural_batch):.2f}")

        # 메모리 정리 (구조모델 계산 후)
        gc.collect()

        # 5. 결합 로그우도
        draw_lls = []
        for j in range(n_draws):
            draw_ll = ll_measurement_batch[j] + ll_choice_batch[j] + ll_structural_batch[j]

            if log_detail and j == 0:
                self.iteration_logger.info("\n[결합 우도 계산] Draw 0:")
                self.iteration_logger.info(f"  측정모델: {ll_measurement_batch[j]:.4f}")
                self.iteration_logger.info(f"  선택모델: {ll_choice_batch[j]:.4f}")
                self.iteration_logger.info(f"  구조모델: {ll_structural_batch[j]:.4f}")
                self.iteration_logger.info(f"  합계: {draw_ll:.4f}")

            if not np.isfinite(draw_ll):
                if log_detail and j == 0:
                    self.iteration_logger.warning(f"  ⚠️ Draw {j}: 비유한 값 감지, -1e10으로 대체")
                draw_ll = -1e10

            draw_lls.append(draw_ll)

        if log_detail:
            self.iteration_logger.info("\n[전체 draws 통계]")
            self.iteration_logger.info(f"  Draw 우도 범위: [{np.min(draw_lls):.2f}, {np.max(draw_lls):.2f}]")
            self.iteration_logger.info(f"  Draw 우도 평균: {np.mean(draw_lls):.2f}")
            self.iteration_logger.info("=" * 80)
            self._first_draw_logged = True

        # 두 번째 함수 호출에서 파라미터 변화 확인
        if hasattr(self, '_first_draw_logged') and not hasattr(self, '_second_draw_logged'):
            self.iteration_logger.info("=" * 80)
            self.iteration_logger.info("두 번째 함수 호출 - 파라미터 변화 확인")
            self.iteration_logger.info("=" * 80)
            self.iteration_logger.info(f"[파라미터 확인]")
            self.iteration_logger.info(f"  측정모델 zeta (health_concern 처음 3개): {param_dict['measurement']['health_concern']['zeta'][:3]}")
            self.iteration_logger.info(f"  구조모델 gamma_lv: {param_dict['structural']['gamma_lv']}")
            self.iteration_logger.info(f"  구조모델 gamma_x: {param_dict['structural']['gamma_x']}")
            self.iteration_logger.info(f"  선택모델 intercept: {param_dict['choice']['intercept']}")
            self.iteration_logger.info(f"  선택모델 beta: {param_dict['choice']['beta']}")
            self.iteration_logger.info(f"  선택모델 lambda: {param_dict['choice']['lambda']}")
            self.iteration_logger.info("=" * 80)
            self._second_draw_logged = True

        return draw_lls

    def _get_initial_parameters(self, measurement_model,
                                structural_model, choice_model) -> np.ndarray:
        """
        초기 파라미터 설정 (다중 잠재변수 지원)
        """
        params = []

        # 다중 잠재변수 측정모델 파라미터
        if hasattr(self.config, 'measurement_configs'):
            # 다중 잠재변수
            for lv_name, config in self.config.measurement_configs.items():
                # measurement_method 확인
                method = getattr(config, 'measurement_method', 'continuous_linear')

                if method == 'continuous_linear':
                    # ContinuousLinearMeasurement
                    n_indicators = len(config.indicators)

                    # 요인적재량 (zeta) - 첫 번째 제외 가능
                    if config.fix_first_loading:
                        params.extend([1.0] * (n_indicators - 1))
                    else:
                        params.extend([1.0] * n_indicators)

                    # 오차분산 (sigma_sq)
                    if not config.fix_error_variance:
                        params.extend([1.0] * n_indicators)

                elif method == 'ordered_probit':
                    # OrderedProbitMeasurement
                    n_indicators = len(config.indicators)
                    n_thresholds = config.n_categories - 1

                    # 요인적재량 (zeta)
                    params.extend([1.0] * n_indicators)

                    # 임계값 (tau)
                    for _ in range(n_indicators):
                        if n_thresholds == 4:
                            params.extend([-2, -1, 1, 2])  # 5점 척도
                        elif n_thresholds == 1:
                            params.extend([0.0])  # 2점 척도
                        else:
                            # 일반적인 경우
                            params.extend(list(range(-n_thresholds//2 + 1, n_thresholds//2 + 1)))

                else:
                    raise ValueError(f"지원하지 않는 측정 방법: {method}")
        else:
            # 단일 잠재변수
            n_indicators = len(self.config.measurement.indicators)
            params.extend([1.0] * n_indicators)

            n_thresholds = self.config.measurement.n_categories - 1
            for _ in range(n_indicators):
                params.extend([-2, -1, 1, 2])

        # 구조모델 파라미터
        if hasattr(self.config.structural, 'n_exo'):
            # 다중 잠재변수 구조모델
            n_exo = self.config.structural.n_exo
            n_cov = self.config.structural.n_cov

            # gamma_lv (외생 LV → 내생 LV)
            params.extend([0.0] * n_exo)

            # gamma_x (공변량 → 내생 LV)
            params.extend([0.0] * n_cov)
        else:
            # 단일 잠재변수 구조모델
            n_sociodem = len(self.config.structural.sociodemographics)
            params.extend([0.0] * n_sociodem)

        # 선택모델 파라미터
        # - 절편
        params.append(0.0)

        # - 속성 계수 (beta)
        n_attributes = len(self.config.choice.choice_attributes)
        params.extend([0.0] * n_attributes)

        # - 잠재변수 계수 (lambda)
        params.append(1.0)

        return np.array(params)

    def _get_parameter_bounds(self, measurement_model,
                              structural_model, choice_model) -> list:
        """
        파라미터 bounds 설정 (다중 잠재변수 지원)
        """
        bounds = []

        # 다중 잠재변수 측정모델 파라미터
        if hasattr(self.config, 'measurement_configs'):
            # 다중 잠재변수
            for lv_name, config in self.config.measurement_configs.items():
                # measurement_method 확인
                method = getattr(config, 'measurement_method', 'continuous_linear')

                if method == 'continuous_linear':
                    # ContinuousLinearMeasurement
                    n_indicators = len(config.indicators)

                    # 요인적재량 (zeta): [-10, 10]
                    if config.fix_first_loading:
                        bounds.extend([(-10.0, 10.0)] * (n_indicators - 1))
                    else:
                        bounds.extend([(-10.0, 10.0)] * n_indicators)

                    # 오차분산 (sigma_sq): [0.01, 100]
                    if not config.fix_error_variance:
                        bounds.extend([(0.01, 100.0)] * n_indicators)

                elif method == 'ordered_probit':
                    # OrderedProbitMeasurement
                    n_indicators = len(config.indicators)
                    n_thresholds = config.n_categories - 1

                    # 요인적재량 (zeta): [0.1, 10]
                    bounds.extend([(0.1, 10.0)] * n_indicators)

                    # 임계값 (tau): [-10, 10]
                    for _ in range(n_indicators):
                        bounds.extend([(-10.0, 10.0)] * n_thresholds)

                else:
                    raise ValueError(f"지원하지 않는 측정 방법: {method}")
        else:
            # 단일 잠재변수
            n_indicators = len(self.config.measurement.indicators)
            bounds.extend([(0.1, 10.0)] * n_indicators)

            n_thresholds = self.config.measurement.n_categories - 1
            for _ in range(n_indicators):
                bounds.extend([(-10.0, 10.0)] * n_thresholds)

        # 구조모델 파라미터
        if hasattr(self.config.structural, 'n_exo'):
            # 다중 잠재변수 구조모델
            n_exo = self.config.structural.n_exo
            n_cov = self.config.structural.n_cov

            # gamma_lv: unbounded
            bounds.extend([(None, None)] * n_exo)

            # gamma_x: unbounded
            bounds.extend([(None, None)] * n_cov)
        else:
            # 단일 잠재변수 구조모델
            n_sociodem = len(self.config.structural.sociodemographics)
            bounds.extend([(None, None)] * n_sociodem)

        # 선택모델 파라미터
        # - 절편: unbounded
        bounds.append((None, None))

        # - 속성 계수 (beta): unbounded
        n_attributes = len(self.config.choice.choice_attributes)
        bounds.extend([(None, None)] * n_attributes)

        # - 잠재변수 계수 (lambda): unbounded
        bounds.append((None, None))

        return bounds

    def _unpack_parameters(self, params: np.ndarray,
                          measurement_model,
                          structural_model,
                          choice_model) -> Dict[str, Dict]:
        """
        파라미터 벡터를 딕셔너리로 변환 (다중 잠재변수 지원)
        """
        # 디버깅: 파라미터 언팩 호출 확인 (간소화)
        if hasattr(self, 'iteration_logger'):
            if not hasattr(self, '_unpack_count'):
                self._unpack_count = 0
            self._unpack_count += 1
            # 처음 3번만 로깅
            if self._unpack_count <= 3:
                self.iteration_logger.info(f"[파라미터 언팩 #{self._unpack_count}] 처음 5개: {params[:5]}, 마지막 5개: {params[-5:]}")

            # 메모리 체크 (파라미터 언팩 시) - 매번 로깅
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.log_memory_stats(f"파라미터 언팩 #{self._unpack_count}")

                # 항상 임계값 체크 및 필요시 정리
                mem_info = self.memory_monitor.check_and_cleanup(f"파라미터 언팩 #{self._unpack_count}")

        idx = 0
        param_dict = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }

        # 다중 잠재변수 측정모델 파라미터
        if hasattr(self.config, 'measurement_configs'):
            # 다중 잠재변수
            for lv_idx, (lv_name, config) in enumerate(self.config.measurement_configs.items()):
                # measurement_method 확인
                method = getattr(config, 'measurement_method', 'continuous_linear')

                if method == 'continuous_linear':
                    # ContinuousLinearMeasurement
                    n_indicators = len(config.indicators)

                    # 요인적재량 (zeta)
                    if config.fix_first_loading:
                        zeta = np.ones(n_indicators)
                        zeta[0] = 1.0  # 고정
                        zeta[1:] = params[idx:idx + n_indicators - 1]
                        idx += n_indicators - 1
                    else:
                        zeta = params[idx:idx + n_indicators]
                        idx += n_indicators

                    # 오차분산 (sigma_sq)
                    if config.fix_error_variance:
                        sigma_sq = np.ones(n_indicators) * config.initial_error_variance
                    else:
                        sigma_sq = params[idx:idx + n_indicators]
                        idx += n_indicators

                    param_dict['measurement'][lv_name] = {'zeta': zeta, 'sigma_sq': sigma_sq}

                    # 첫 번째 LV에 대해서만 상세 로깅
                    if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
                        if self._unpack_count <= 3 and lv_idx == 0:
                            self.iteration_logger.info(f"  측정모델 {lv_name}: zeta[0]={zeta[0]:.4f}, sigma_sq[0]={sigma_sq[0]:.4f}")

                elif method == 'ordered_probit':
                    # OrderedProbitMeasurement
                    n_indicators = len(config.indicators)
                    n_thresholds = config.n_categories - 1

                    # 요인적재량 (zeta)
                    zeta = params[idx:idx+n_indicators]
                    idx += n_indicators

                    # 임계값 (tau)
                    tau_list = []
                    for i in range(n_indicators):
                        tau_list.append(params[idx:idx+n_thresholds])
                        idx += n_thresholds
                    tau = np.array(tau_list)

                    param_dict['measurement'][lv_name] = {'zeta': zeta, 'tau': tau}

                    # 첫 번째 LV에 대해서만 상세 로깅
                    if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
                        if self._unpack_count <= 3 and lv_idx == 0:
                            self.iteration_logger.info(f"  측정모델 {lv_name}: zeta[0]={zeta[0]:.4f}, tau[0,0]={tau[0,0]:.4f}")

                else:
                    raise ValueError(f"지원하지 않는 측정 방법: {method}")
        else:
            # 단일 잠재변수
            n_indicators = len(self.config.measurement.indicators)
            zeta = params[idx:idx+n_indicators]
            idx += n_indicators

            n_thresholds = self.config.measurement.n_categories - 1
            tau_list = []
            for i in range(n_indicators):
                tau_list.append(params[idx:idx+n_thresholds])
                idx += n_thresholds
            tau = np.array(tau_list)

            param_dict['measurement'] = {'zeta': zeta, 'tau': tau}

        # 구조모델 파라미터
        if hasattr(self.config.structural, 'n_exo'):
            # 다중 잠재변수 구조모델
            n_exo = self.config.structural.n_exo
            n_cov = self.config.structural.n_cov

            # gamma_lv (외생 LV → 내생 LV)
            gamma_lv = params[idx:idx+n_exo]
            idx += n_exo

            # gamma_x (공변량 → 내생 LV)
            gamma_x = params[idx:idx+n_cov]
            idx += n_cov

            param_dict['structural'] = {'gamma_lv': gamma_lv, 'gamma_x': gamma_x}

            # 상세 로깅 (간소화)
            if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
                if self._unpack_count <= 3:
                    self.iteration_logger.info(f"  구조모델: gamma_lv[0]={gamma_lv[0]:.6f}, gamma_x[0]={gamma_x[0]:.6f}")
        else:
            # 단일 잠재변수 구조모델
            n_sociodem = len(self.config.structural.sociodemographics)
            gamma = params[idx:idx+n_sociodem]
            idx += n_sociodem

            param_dict['structural'] = {'gamma': gamma}

        # 선택모델 파라미터
        intercept = params[idx]
        idx += 1

        n_attributes = len(self.config.choice.choice_attributes)
        beta = params[idx:idx+n_attributes]
        idx += n_attributes

        lambda_lv = params[idx]
        idx += 1

        param_dict['choice'] = {
            'intercept': intercept,
            'beta': beta,
            'lambda': lambda_lv
        }

        # 상세 로깅 (간소화)
        if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
            if self._unpack_count <= 3:
                self.iteration_logger.info(f"  선택모델: intercept={intercept:.6f}, beta[0]={beta[0]:.6f}, lambda={lambda_lv:.6f}")

        return param_dict

