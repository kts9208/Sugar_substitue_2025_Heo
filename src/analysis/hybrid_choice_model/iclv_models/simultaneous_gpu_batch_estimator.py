"""
동시추정 GPU 배치 처리 ICLV Estimator

SimultaneousEstimator를 상속하여 GPU 배치 처리로 가속합니다.
개인별 우도 계산 부분만 GPU 배치로 오버라이드합니다.

주의: 이 클래스는 동시추정(Simultaneous Estimation) 전용입니다.
순차추정(Sequential Estimation)에는 SequentialEstimator를 사용하세요.
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


class SimultaneousGPUBatchEstimator(SimultaneousEstimator):
    """
    동시추정 GPU 배치 처리 ICLV Estimator

    SimultaneousEstimator를 상속하여 GPU 배치 처리로 가속합니다.
    개인별 우도 계산 부분만 GPU 배치로 오버라이드합니다.

    주의: 이 클래스는 동시추정(Simultaneous Estimation) 전용입니다.
    순차추정(Sequential Estimation)에는 SequentialEstimator를 사용하세요.
    """
    
    def __init__(self, config, use_gpu: bool = True,
                 memory_monitor_cpu_threshold_mb: float = 2000,
                 memory_monitor_gpu_threshold_mb: float = 1500,
                 use_full_parallel: bool = True):
        """
        Args:
            config: MultiLatentConfig 또는 ICLVConfig
            use_gpu: GPU 사용 여부
            memory_monitor_cpu_threshold_mb: CPU 메모리 임계값 (MB)
            memory_monitor_gpu_threshold_mb: GPU 메모리 임계값 (MB)
            use_full_parallel: 완전 병렬 처리 사용 여부 (Advanced Indexing, 기본값: True)
        """
        super().__init__(config)
        self.use_gpu = use_gpu and gpu_batch_utils.CUPY_AVAILABLE
        self.use_full_parallel = use_full_parallel
        self.gpu_measurement_model = None

        # 메모리 모니터 임계값 저장 (나중에 초기화)
        self.memory_monitor_cpu_threshold_mb = memory_monitor_cpu_threshold_mb
        self.memory_monitor_gpu_threshold_mb = memory_monitor_gpu_threshold_mb
        self.memory_monitor = None  # estimate()에서 초기화

        # 사용자 정의 초기값 저장
        self.user_initial_params = None

        if self.use_gpu:
            if self.use_full_parallel:
                logger.info("✨ GPU 완전 병렬 처리 활성화 (Advanced Indexing)")
            else:
                logger.info("GPU 배치 처리 활성화")
        else:
            logger.info("GPU 배치 처리 비활성화 (CPU 모드)")
    
    def estimate(self, data: pd.DataFrame,
                measurement_model,
                structural_model,
                choice_model,
                log_file: Optional[str] = None,
                initial_params: Optional[np.ndarray] = None) -> Dict:
        """
        ICLV 모델 추정 (GPU 배치 가속)

        Args:
            data: 전체 데이터
            measurement_model: 측정모델 인스턴스
            structural_model: 구조모델 인스턴스
            choice_model: 선택모델 인스턴스
            log_file: 로그 파일 경로
            initial_params: 사용자 정의 초기 파라미터 (선택사항)

        Returns:
            추정 결과 딕셔너리
        """
        # 사용자 정의 초기값 저장
        self.user_initial_params = initial_params
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

            # ✅ 계층적 구조 지원
            if structural_model.is_hierarchical:
                # 1차 LV + 2차+ LV 오차항
                n_first_order = len(structural_model.exogenous_lvs)
                n_higher_order = len(structural_model.get_higher_order_lvs())
                n_dimensions = n_first_order + n_higher_order

                logger.info(f"계층적 구조: 1차 LV={n_first_order}, 2차+ LV={n_higher_order}, 총 차원={n_dimensions}")
            else:
                # 병렬 구조 (하위 호환)
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
    
    def _log_parameters(self, param_dict: Dict, iteration: int):
        """
        파라미터 값 로깅

        Args:
            param_dict: 파라미터 딕셔너리
            iteration: 현재 iteration 번호
        """
        if not hasattr(self, 'iteration_logger') or self.iteration_logger is None:
            return

        # 로깅 레벨 확인
        log_level = getattr(self.config.estimation, 'gradient_log_level', 'DETAILED')

        if log_level not in ['MODERATE', 'DETAILED']:
            return

        self.iteration_logger.info("\n" + "="*80)
        self.iteration_logger.info(f"Iteration {iteration} - 파라미터 값")
        self.iteration_logger.info("="*80)

        # 측정모델 파라미터
        self.iteration_logger.info("\n[측정모델 파라미터]")
        for lv_idx, (lv_name, lv_params) in enumerate(param_dict['measurement'].items()):
            if log_level == 'DETAILED' or lv_idx == 0:
                self.iteration_logger.info(f"  {lv_name}:")
                zeta = lv_params['zeta']
                # 전체 파라미터 출력 (초기값 설정용)
                self.iteration_logger.info(f"    - zeta: {zeta}")

                if 'sigma_sq' in lv_params:
                    sigma_sq = lv_params['sigma_sq']
                    self.iteration_logger.info(f"    - sigma_sq: {sigma_sq}")
                elif 'tau' in lv_params:
                    tau = lv_params['tau']
                    self.iteration_logger.info(f"    - tau shape: {tau.shape}")

        # 구조모델 파라미터
        self.iteration_logger.info("\n[구조모델 파라미터]")
        if hasattr(self.config.structural, 'is_hierarchical') and self.config.structural.is_hierarchical:
            # 계층적 구조
            for key, value in param_dict['structural'].items():
                if key.startswith('gamma_'):
                    self.iteration_logger.info(f"  {key}: {value:.6f}")
        else:
            # 병렬 구조
            if 'gamma_lv' in param_dict['structural']:
                self.iteration_logger.info(f"  gamma_lv: {param_dict['structural']['gamma_lv']}")
            if 'gamma_x' in param_dict['structural']:
                self.iteration_logger.info(f"  gamma_x: {param_dict['structural']['gamma_x']}")

        # 선택모델 파라미터
        self.iteration_logger.info("\n[선택모델 파라미터]")
        choice_params = param_dict['choice']
        self.iteration_logger.info(f"  intercept: {choice_params['intercept']:.6f}")
        self.iteration_logger.info(f"  beta: {choice_params['beta']}")

        if 'lambda_main' in choice_params:
            # 조절효과 모델
            self.iteration_logger.info(f"  lambda_main: {choice_params['lambda_main']:.6f}")
            for key in choice_params:
                if key.startswith('lambda_mod_'):
                    self.iteration_logger.info(f"  {key}: {choice_params[key]:.6f}")
        else:
            # 기본 모델
            self.iteration_logger.info(f"  lambda: {choice_params['lambda']:.6f}")

        self.iteration_logger.info("="*80)

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

        # 파라미터 로깅 (처음 3번 또는 10의 배수 iteration)
        if self._current_iteration <= 3 or self._current_iteration % 10 == 0:
            self._log_parameters(param_dict, self._current_iteration)

        # 메모리 체크 (Halton draws 가져오기 전) - 비활성화
        # if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
        #     self.memory_monitor.log_memory_stats(f"Halton draws 가져오기 전 (Iter {self._current_iteration})")

        draws = self.halton_generator.get_draws()

        # 메모리 체크 (Halton draws 가져온 후) - 비활성화
        # if hasattr(self, 'memory_monitor') and hasattr(self, '_likelihood_call_count'):
        #     self.memory_monitor.log_memory_stats(f"Halton draws 가져온 후 (Iter {self._current_iteration})")

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
                if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
                    # ✅ 계층적 구조
                    n_first_order = len(structural_model.exogenous_lvs)
                    exo_draws = draw[:n_first_order]

                    # 2차+ LV 오차항
                    higher_order_draws = {}
                    higher_order_lvs = structural_model.get_higher_order_lvs()
                    for i, lv_name in enumerate(higher_order_lvs):
                        higher_order_draws[lv_name] = draw[n_first_order + i]

                    lv = structural_model.predict(
                        ind_data, exo_draws, param_dict['structural'],
                        higher_order_draws=higher_order_draws
                    )

                elif hasattr(structural_model, 'endogenous_lv'):
                    # 병렬 구조 (하위 호환)
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
                if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
                    # ✅ 계층적 구조
                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, exo_draws, param_dict['structural'],
                        higher_order_draws=higher_order_draws
                    )
                elif hasattr(structural_model, 'endogenous_lv'):
                    # 병렬 구조
                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, exo_draws, param_dict['structural'], endo_draw
                    )
                else:
                    # 단일 잠재변수
                    ll_structural = structural_model.log_likelihood(
                        ind_data, lv, param_dict['structural'], draw
                    )

                # 결합 로그우도
                draw_ll = ll_measurement + ll_choice + ll_structural

                # 🔍 디버깅: 첫 번째 draw의 우도 분해
                if j == 0 and not hasattr(self, '_ll_debug_logged'):
                    self._ll_debug_logged = True
                    print(f"[DEBUG LL Components] Measurement={ll_measurement:.4f}, Choice={ll_choice:.4f}, Structural={ll_structural:.4f}, Total={draw_ll:.4f}")

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

            if hasattr(structural_model, 'is_hierarchical') and structural_model.is_hierarchical:
                # ✅ 계층적 구조
                n_first_order = len(structural_model.exogenous_lvs)
                exo_draws = draw[:n_first_order]

                # 2차+ LV 오차항
                higher_order_draws = {}
                higher_order_lvs = structural_model.get_higher_order_lvs()
                for i, lv_name in enumerate(higher_order_lvs):
                    higher_order_draws[lv_name] = draw[n_first_order + i]

                lv = structural_model.predict(
                    ind_data, exo_draws, param_dict['structural'],
                    higher_order_draws=higher_order_draws
                )

                if log_detail and j == 0:
                    self.iteration_logger.info(f"[구조모델 예측 - 계층적] Draw 0:")
                    self.iteration_logger.info(f"  1차 LV draws: {exo_draws}")
                    self.iteration_logger.info(f"  2차+ LV 오차항: {higher_order_draws}")
                    self.iteration_logger.info(f"  예측된 LV: {lv}")

            elif hasattr(structural_model, 'endogenous_lv'):
                # 병렬 구조 (하위 호환)
                n_exo = structural_model.n_exo
                exo_draws = draw[:n_exo]
                endo_draw = draw[n_exo]
                lv = structural_model.predict(ind_data, exo_draws, param_dict['structural'], endo_draw)

                if log_detail and j == 0:
                    self.iteration_logger.info(f"[구조모델 예측 - 병렬] Draw 0:")
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

            # ✅ 계층적 구조 지원
            if hasattr(self.config.structural, 'is_hierarchical') and self.config.structural.is_hierarchical:
                # 계층적 구조: 개별 경로 파라미터
                first_param = list(param_dict['structural'].keys())[0]
                self.iteration_logger.info(f"  구조모델 (계층적) {first_param}: {param_dict['structural'][first_param]}")
            else:
                # 병렬 구조 (하위 호환)
                self.iteration_logger.info(f"  구조모델 gamma_lv: {param_dict['structural']['gamma_lv']}")
                self.iteration_logger.info(f"  구조모델 gamma_x: {param_dict['structural']['gamma_x']}")

            self.iteration_logger.info(f"  선택모델 intercept: {param_dict['choice']['intercept']}")
            self.iteration_logger.info(f"  선택모델 beta: {param_dict['choice']['beta']}")

            # ✅ 조절효과 지원
            if 'lambda_main' in param_dict['choice']:
                self.iteration_logger.info(f"  선택모델 lambda_main: {param_dict['choice']['lambda_main']}")
                for key in param_dict['choice']:
                    if key.startswith('lambda_mod_'):
                        self.iteration_logger.info(f"  선택모델 {key}: {param_dict['choice'][key]}")
            else:
                self.iteration_logger.info(f"  선택모델 lambda: {param_dict['choice']['lambda']}")

            self.iteration_logger.info("=" * 80)
            self._second_draw_logged = True

        return draw_lls

    def _get_initial_parameters(self, measurement_model,
                                structural_model, choice_model) -> np.ndarray:
        """
        초기 파라미터 설정 (다중 잠재변수 지원)

        ✅ 사용자 정의 초기값이 있으면 우선 사용
        ✅ 없으면 최종 수렴값 (Iteration 24) 기반 초기값 사용
        """
        # 사용자 정의 초기값이 있으면 그것을 사용
        if self.user_initial_params is not None:
            logger.info(f"사용자 정의 초기값 사용 (파라미터 수: {len(self.user_initial_params)})")
            return self.user_initial_params

        from .initial_values_final import (
            get_zeta_initial_value,
            get_sigma_sq_initial_value,
            ZETA_INITIAL_VALUES,
            SIGMA_SQ_INITIAL_VALUES
        )

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

                    # 요인적재량 (zeta)
                    # ✅ Iteration 40 기반 초기값 사용
                    if lv_name in ZETA_INITIAL_VALUES:
                        zeta_values = ZETA_INITIAL_VALUES[lv_name]['values']
                        if config.fix_first_loading:
                            # 첫 번째는 1.0으로 고정 (파라미터 벡터에 포함하지 않음)
                            params.extend(zeta_values)
                        else:
                            # 첫 번째도 포함
                            params.extend([1.0] + zeta_values)
                    else:
                        # 기본값 (이전 방식)
                        zeta_init = get_zeta_initial_value(lv_name, default=0.05)
                        if config.fix_first_loading:
                            params.extend([zeta_init] * (n_indicators - 1))
                        else:
                            params.extend([zeta_init] * n_indicators)

                    # 오차분산 (sigma_sq)
                    # ✅ Iteration 40 기반 초기값 사용
                    if lv_name in SIGMA_SQ_INITIAL_VALUES:
                        sigma_sq_values = SIGMA_SQ_INITIAL_VALUES[lv_name]['values']
                        if not config.fix_error_variance:
                            params.extend(sigma_sq_values)
                    else:
                        # 기본값 (이전 방식)
                        sigma_sq_init = get_sigma_sq_initial_value(lv_name, default=0.03)
                        if not config.fix_error_variance:
                            params.extend([sigma_sq_init] * n_indicators)

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
        if hasattr(self.config.structural, 'is_hierarchical') and self.config.structural.is_hierarchical:
            # ✅ 계층적 구조
            from .initial_values_final import get_gamma_initial_value

            for path in self.config.structural.hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                for pred in predictors:
                    # ✅ 최종 수렴값 기반 초기값 사용
                    path_name = f'{pred}_to_{target}'
                    gamma_init = get_gamma_initial_value(path_name, default=0.5)
                    params.append(gamma_init)
        elif hasattr(self.config.structural, 'n_exo'):
            # 병렬 구조 (하위 호환)
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
        from .initial_values_final import get_choice_initial_value

        # - 절편
        # ✅ 최종 수렴값 기반 초기값 사용
        params.append(get_choice_initial_value('intercept', default=0.0))

        # - 속성 계수 (beta)
        # ✅ 최종 수렴값 기반 초기값 사용
        n_attributes = len(self.config.choice.choice_attributes)
        for attr in self.config.choice.choice_attributes:
            if 'price' in attr.lower():
                params.append(get_choice_initial_value('beta_price', default=-0.26))
            elif 'sugar' in attr.lower():
                params.append(get_choice_initial_value('beta_sugar_free', default=0.23))
            elif 'health' in attr.lower():
                params.append(get_choice_initial_value('beta_health_label', default=0.23))
            else:
                # 기타 속성
                params.append(0.2)

        # - 잠재변수 계수
        # ✅ 모든 LV 주효과 지원
        if hasattr(self.config.choice, 'all_lvs_as_main') and self.config.choice.all_lvs_as_main:
            # 모든 LV 주효과 모델: lambda_{lv_name}
            if hasattr(self.config.choice, 'main_lvs'):
                for lv_name in self.config.choice.main_lvs:
                    # 각 LV별 초기값 (1.0)
                    params.append(1.0)
        elif hasattr(self.config.choice, 'moderation_enabled') and self.config.choice.moderation_enabled:
            # ✅ 조절효과 모델 - 최종 수렴값 기반 초기값 사용
            params.append(get_choice_initial_value('lambda_main', default=0.45))

            # lambda_mod (조절효과 계수)
            for mod_lv in self.config.choice.moderator_lvs:
                if 'price' in mod_lv.lower():
                    params.append(get_choice_initial_value('lambda_mod_perceived_price', default=-1.50))
                elif 'knowledge' in mod_lv.lower():
                    params.append(get_choice_initial_value('lambda_mod_nutrition_knowledge', default=1.05))
                else:
                    params.append(0.0)
        else:
            # 기본 모델 (하위 호환)
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
                        # 첫 번째는 고정 (파라미터 벡터에 포함하지 않음)
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
        if hasattr(self.config.structural, 'is_hierarchical') and self.config.structural.is_hierarchical:
            # ✅ 계층적 구조
            for path in self.config.structural.hierarchical_paths:
                predictors = path['predictors']

                for pred in predictors:
                    # gamma: unbounded
                    bounds.append((None, None))
        elif hasattr(self.config.structural, 'n_exo'):
            # 병렬 구조 (하위 호환)
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

        # - 잠재변수 계수
        if hasattr(self.config.choice, 'moderation_enabled') and self.config.choice.moderation_enabled:
            # ✅ 조절효과 모델
            # lambda_main: unbounded
            bounds.append((None, None))

            # lambda_mod: unbounded
            for mod_lv in self.config.choice.moderator_lvs:
                bounds.append((None, None))
        else:
            # 기본 모델 (하위 호환)
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
        if hasattr(self, 'iteration_logger') and self.iteration_logger is not None:
            if not hasattr(self, '_unpack_count'):
                self._unpack_count = 0
            self._unpack_count += 1
            # ✅ 파라미터 언팩 로깅 비활성화 (메모리 로깅 포함)
            # 처음 3번만 로깅
            # if self._unpack_count <= 3:
            #     self.iteration_logger.info(f"[파라미터 언팩 #{self._unpack_count}] 처음 5개: {params[:5]}, 마지막 5개: {params[-5:]}")

            # 메모리 체크 (파라미터 언팩 시) - 비활성화
            # if hasattr(self, 'memory_monitor'):
            #     self.memory_monitor.log_memory_stats(f"파라미터 언팩 #{self._unpack_count}")
            #
            #     # 항상 임계값 체크 및 필요시 정리
            #     mem_info = self.memory_monitor.check_and_cleanup(f"파라미터 언팩 #{self._unpack_count}")

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
        if hasattr(self.config.structural, 'is_hierarchical') and self.config.structural.is_hierarchical:
            # ✅ 계층적 구조
            for path in self.config.structural.hierarchical_paths:
                target = path['target']
                predictors = path['predictors']

                for pred in predictors:
                    param_name = f'gamma_{pred}_to_{target}'
                    param_dict['structural'][param_name] = params[idx]
                    idx += 1

            # 상세 로깅
            if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
                if self._unpack_count <= 3:
                    first_param = list(param_dict['structural'].keys())[0]
                    self.iteration_logger.info(f"  구조모델 (계층적): {first_param}={param_dict['structural'][first_param]:.6f}")

        elif hasattr(self.config.structural, 'n_exo'):
            # 병렬 구조 (하위 호환)
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
                    gamma_lv_str = f"gamma_lv[0]={gamma_lv[0]:.6f}" if len(gamma_lv) > 0 else "gamma_lv=[]"
                    gamma_x_str = f"gamma_x[0]={gamma_x[0]:.6f}" if len(gamma_x) > 0 else "gamma_x=[]"
                    self.iteration_logger.info(f"  구조모델: {gamma_lv_str}, {gamma_x_str}")
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

        # 잠재변수 계수
        if hasattr(self.config.choice, 'moderation_enabled') and self.config.choice.moderation_enabled:
            # ✅ 조절효과 모델
            lambda_main = params[idx]
            idx += 1

            param_dict['choice'] = {
                'intercept': intercept,
                'beta': beta,
                'lambda_main': lambda_main
            }

            # 조절효과 계수
            for mod_lv in self.config.choice.moderator_lvs:
                param_name = f'lambda_mod_{mod_lv}'
                param_dict['choice'][param_name] = params[idx]
                idx += 1

            # 상세 로깅
            if hasattr(self, 'iteration_logger') and hasattr(self, '_unpack_count'):
                if self._unpack_count <= 3:
                    first_mod = self.config.choice.moderator_lvs[0]
                    self.iteration_logger.info(f"  선택모델 (조절): intercept={intercept:.6f}, lambda_main={lambda_main:.6f}, lambda_mod_{first_mod}={param_dict['choice'][f'lambda_mod_{first_mod}']:.6f}")
        else:
            # 기본 모델 (하위 호환)
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

    def _structure_statistics(self, estimates, std_errors, t_stats, p_values,
                              measurement_model, structural_model, choice_model):
        """
        파라미터별 통계량을 구조화된 딕셔너리로 변환 (다중 잠재변수 지원)

        Args:
            estimates: 추정값 벡터
            std_errors: 표준오차 벡터
            t_stats: t-통계량 벡터
            p_values: p-value 벡터
            measurement_model: 측정모델
            structural_model: 구조모델
            choice_model: 선택모델

        Returns:
            구조화된 통계량 딕셔너리
            {
                'measurement': {
                    'lv_name1': {'zeta': {...}, 'sigma_sq': {...}},
                    'lv_name2': {'zeta': {...}, 'sigma_sq': {...}},
                    ...
                },
                'structural': {'gamma_pred_to_target': {...}, ...},
                'choice': {'intercept': {...}, 'beta': {...}, 'lambda_main': {...}, ...}
            }
        """
        # 파라미터 언팩 (다중 잠재변수 지원)
        param_dict = self._unpack_parameters(
            estimates, measurement_model, structural_model, choice_model
        )

        # 동일한 방식으로 표준오차, t-통계량, p-value 언팩
        se_dict = self._unpack_parameters(
            std_errors, measurement_model, structural_model, choice_model
        )
        t_dict = self._unpack_parameters(
            t_stats, measurement_model, structural_model, choice_model
        )
        p_dict = self._unpack_parameters(
            p_values, measurement_model, structural_model, choice_model
        )

        # 구조화된 결과 생성
        structured = {
            'measurement': {},
            'structural': {},
            'choice': {}
        }

        # 측정모델 (다중 잠재변수 지원)
        if 'measurement' in param_dict:
            if hasattr(self.config, 'measurement_configs'):
                # 다중 잠재변수
                for lv_name in param_dict['measurement'].keys():
                    structured['measurement'][lv_name] = {}

                    # zeta (요인적재량)
                    if 'zeta' in param_dict['measurement'][lv_name]:
                        structured['measurement'][lv_name]['zeta'] = {
                            'estimate': param_dict['measurement'][lv_name]['zeta'],
                            'std_error': se_dict['measurement'][lv_name]['zeta'],
                            't_statistic': t_dict['measurement'][lv_name]['zeta'],
                            'p_value': p_dict['measurement'][lv_name]['zeta']
                        }

                    # sigma_sq (오차분산) - continuous_linear 방식
                    if 'sigma_sq' in param_dict['measurement'][lv_name]:
                        structured['measurement'][lv_name]['sigma_sq'] = {
                            'estimate': param_dict['measurement'][lv_name]['sigma_sq'],
                            'std_error': se_dict['measurement'][lv_name]['sigma_sq'],
                            't_statistic': t_dict['measurement'][lv_name]['sigma_sq'],
                            'p_value': p_dict['measurement'][lv_name]['sigma_sq']
                        }

                    # tau (임계값) - ordered_probit 방식
                    if 'tau' in param_dict['measurement'][lv_name]:
                        structured['measurement'][lv_name]['tau'] = {
                            'estimate': param_dict['measurement'][lv_name]['tau'],
                            'std_error': se_dict['measurement'][lv_name]['tau'],
                            't_statistic': t_dict['measurement'][lv_name]['tau'],
                            'p_value': p_dict['measurement'][lv_name]['tau']
                        }
            else:
                # 단일 잠재변수 (하위 호환)
                for key in param_dict['measurement']:
                    structured['measurement'][key] = {
                        'estimate': param_dict['measurement'][key],
                        'std_error': se_dict['measurement'][key],
                        't_statistic': t_dict['measurement'][key],
                        'p_value': p_dict['measurement'][key]
                    }

        # 구조모델 (계층적 구조 지원)
        if 'structural' in param_dict:
            for key in param_dict['structural']:
                structured['structural'][key] = {
                    'estimate': param_dict['structural'][key],
                    'std_error': se_dict['structural'][key],
                    't_statistic': t_dict['structural'][key],
                    'p_value': p_dict['structural'][key]
                }

        # 선택모델 (조절효과 지원)
        if 'choice' in param_dict:
            for key in param_dict['choice']:
                structured['choice'][key] = {
                    'estimate': param_dict['choice'][key],
                    'std_error': se_dict['choice'][key],
                    't_statistic': t_dict['choice'][key],
                    'p_value': p_dict['choice'][key]
                }

        return structured

