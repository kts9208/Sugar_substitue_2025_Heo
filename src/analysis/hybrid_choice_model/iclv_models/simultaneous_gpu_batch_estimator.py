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

        # ✅ 대안별 모델 (ASC) 또는 Binary 모델 (intercept)
        if 'asc_sugar' in choice_params:
            # Multinomial Logit with ASC
            self.iteration_logger.info(f"  asc_sugar: {choice_params['asc_sugar']:.6f}")
            self.iteration_logger.info(f"  asc_sugar_free: {choice_params['asc_sugar_free']:.6f}")
        elif 'intercept' in choice_params:
            # Binary Logit with intercept
            self.iteration_logger.info(f"  intercept: {choice_params['intercept']:.6f}")

        self.iteration_logger.info(f"  beta: {choice_params['beta']}")

        # ✅ 대안별 LV 계수 (theta_*) 또는 일반 LV 계수 (lambda_*)
        for key in sorted(choice_params.keys()):
            if key.startswith('theta_'):
                self.iteration_logger.info(f"  {key}: {choice_params[key]:.6f}")
            elif key.startswith('lambda_'):
                self.iteration_logger.info(f"  {key}: {choice_params[key]:.6f}")

        # ✅ LV-Attribute 상호작용 (gamma_*)
        for key in sorted(choice_params.keys()):
            if key.startswith('gamma_') and not '_to_' in key:
                self.iteration_logger.info(f"  {key}: {choice_params[key]:.6f}")

        self.iteration_logger.info("="*80)

    def _joint_log_likelihood(self, params: np.ndarray,
                             measurement_model,
                             structural_model,
                             choice_model) -> float:
        """
        결합 로그우도 계산 (완전 GPU 병렬화)

        🚀 모든 개인 × 모든 draws를 한 번에 GPU로 계산
        """
        # 현재 iteration 번호 저장
        if not hasattr(self, '_current_iteration'):
            self._current_iteration = 0
        self._current_iteration += 1

        # 파라미터 분해
        param_dict = self._unpack_parameters(
            params, measurement_model, structural_model, choice_model
        )

        # 파라미터 로깅 (처음 3번 또는 10의 배수 iteration)
        if self._current_iteration <= 3 or self._current_iteration % 10 == 0:
            self._log_parameters(param_dict, self._current_iteration)

        draws = self.halton_generator.get_draws()
        individual_ids = self.data[self.config.individual_id_column].unique()

        # ✅ 완전 GPU 병렬화: 모든 개인을 한 번에 처리
        if self.use_gpu and self.use_full_parallel:
            # 모든 개인 데이터 준비
            all_ind_data = []
            for ind_id in individual_ids:
                ind_data = self.data[self.data[self.config.individual_id_column] == ind_id]
                all_ind_data.append(ind_data)

            # gpu_gradient_batch의 완전 병렬화 함수 사용
            from . import gpu_gradient_batch

            # 로깅 레벨 설정
            log_level = 'DETAILED' if self._current_iteration == 1 else 'MINIMAL'

            total_ll = gpu_gradient_batch.compute_all_individuals_likelihood_full_batch_gpu(
                self.gpu_measurement_model,
                all_ind_data,
                draws,
                param_dict,
                structural_model,
                choice_model,
                iteration_logger=self.iteration_logger if hasattr(self, 'iteration_logger') else None,
                log_level=log_level
            )
        else:
            # 기존 방식: 개인별 순차 처리
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

            # ✅ 유연한 리스트 기반: 모든 lambda_* 파라미터 출력
            for key in sorted(param_dict['choice'].keys()):
                if key.startswith('lambda_'):
                    self.iteration_logger.info(f"  선택모델 {key}: {param_dict['choice'][key]}")

            # ✅ 유연한 리스트 기반: 모든 gamma_* 파라미터 출력 (LV-Attribute 상호작용)
            for key in sorted(param_dict['choice'].keys()):
                if key.startswith('gamma_') and not '_to_' in key:
                    self.iteration_logger.info(f"  선택모델 {key}: {param_dict['choice'][key]}")

            self.iteration_logger.info("=" * 80)
            self._second_draw_logged = True

        return draw_lls

    # ❌ 제거됨: _compute_all_individuals_likelihood_full_batch_gpu
    # ✅ gpu_gradient_batch.compute_all_individuals_likelihood_full_batch_gpu 사용
    # (중복 제거, 기존 인프라 활용)

    # ❌ 제거됨: _get_initial_parameters
    # ✅ 부모 클래스(SimultaneousEstimatorFixed)의 메서드 사용
    # (ParameterManager 기반, 중복 로직 제거)

    # ❌ 제거됨: _get_parameter_bounds
    # ✅ 부모 클래스(SimultaneousEstimatorFixed)의 메서드 사용
    # (ParameterManager 기반, optimizer와 무관하게 동일한 로직 사용)

    # ❌ 제거됨: _unpack_parameters (197 lines)
    # ✅ 부모 클래스(SimultaneousEstimatorFixed)의 메서드 사용
    # (ParameterManager 기반, 유연한 리스트 기반 시스템)

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

