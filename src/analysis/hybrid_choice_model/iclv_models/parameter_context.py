"""
파라미터 변환 및 관리를 담당하는 컨텍스트 클래스 (동시추정 전용)

파라미터 타입 간 변환 로직을 단일 클래스로 캡슐화하여
단일 책임 원칙을 준수합니다.

동시추정 전제:
- 측정모델 파라미터는 항상 CFA 결과로 고정
- 구조모델 + 선택모델 파라미터만 최적화

파라미터 타입:
- params_scaled: L-BFGS-B가 최적화하는 파라미터 (스케일됨, 최적화 대상만)
- params_opt: 최적화 파라미터 (언스케일됨, 최적화 대상만)
- params_full: 전체 파라미터 (언스케일됨, 측정모델 포함)

Author: Taeseok Kim
Date: 2025-01-19
"""

import numpy as np
import logging


class ParameterContext:
    """
    파라미터 변환 및 관리를 담당하는 컨텍스트 클래스 (동시추정 전용)

    전제:
    - 측정모델 파라미터는 항상 고정 (CFA 결과)
    - 구조모델 + 선택모델 파라미터만 최적화

    책임:
    1. 파라미터 타입 변환 (scaled ↔ unscaled, optimized ↔ full)
    2. 파라미터 타입 추적 (어떤 파라미터인지 명확히)
    3. 일관된 인터페이스 제공

    장점:
    - 파라미터 변환 로직이 한 곳에 집중됨
    - 타입 불일치 에러 방지
    - 코드 중복 제거
    - 유지보수성 향상
    """

    def __init__(self, param_manager, param_scaler,
                 measurement_model,
                 logger: logging.Logger = None):
        """
        초기화 (동시추정 전용)

        ✅ 측정모델 파라미터는 measurement_model에서 직접 추출

        Args:
            param_manager: ParameterManager 인스턴스
            param_scaler: ParameterScaler 인스턴스
            measurement_model: 측정모델 (CFA 결과 포함)
            logger: 로거 (선택)
        """
        self.param_manager = param_manager
        self.param_scaler = param_scaler
        self.measurement_model = measurement_model
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(
            f"ParameterContext 초기화: 측정모델 파라미터는 CFA 결과 사용"
        )
    
    # ===== 파라미터 변환 메서드 =====

    def to_full_external(self, params_scaled: np.ndarray) -> np.ndarray:
        """
        L-BFGS-B 파라미터 → 전체 외부 파라미터 (동시추정 전용)

        ✅ 동시추정에서는 params_scaled가 이미 최적화 파라미터만 포함 (8개)
        ✅ 측정모델 파라미터는 추가하지 않음 (이미 분리됨)

        변환 과정:
        params_scaled (n_opt개, scaled) → params_opt (n_opt개, unscaled)

        Args:
            params_scaled: 스케일된 최적화 파라미터 (8개)

        Returns:
            언스케일된 최적화 파라미터 (8개, 측정모델 제외)
        """
        # 언스케일링만 수행 (측정모델 파라미터 추가 안 함)
        params_opt = self.param_scaler.unscale_parameters(params_scaled)

        return params_opt
    
    def to_optimized_external(self, params_scaled: np.ndarray) -> np.ndarray:
        """
        L-BFGS-B 파라미터 → 최적화 외부 파라미터
        
        변환 과정:
        params_scaled (n_opt개, scaled) → params_opt (n_opt개, unscaled)
        
        Args:
            params_scaled: 스케일된 최적화 파라미터
        
        Returns:
            언스케일된 최적화 파라미터
        """
        return self.param_scaler.unscale_parameters(params_scaled)
    
    def extract_optimized_gradient(self, grad_full: np.ndarray) -> np.ndarray:
        """
        전체 그래디언트 → 최적화 그래디언트 (동시추정 전용)

        ✅ 동시추정에서는 grad_full이 이미 최적화 그래디언트만 포함 (8개)
        ✅ 측정모델 그래디언트는 이미 제거됨

        Args:
            grad_full: 최적화 파라미터에 대한 그래디언트 (8개)

        Returns:
            그대로 반환 (이미 최적화 그래디언트)
        """
        # 동시추정에서는 이미 최적화 그래디언트만 포함
        return grad_full
    
    def scale_gradient(self, grad_external: np.ndarray) -> np.ndarray:
        """
        외부 그래디언트 → 스케일된 그래디언트
        
        변환 과정:
        grad_external (n_opt개, unscaled) → grad_scaled (n_opt개, scaled)
        
        Args:
            grad_external: 언스케일된 그래디언트
        
        Returns:
            스케일된 그래디언트
        """
        return self.param_scaler.scale_gradient(grad_external)
    
    def get_info(self) -> dict:
        """
        파라미터 컨텍스트 정보 반환

        Returns:
            정보 딕셔너리
        """
        return {
            'n_optimized': len(self.param_scaler.scales) if hasattr(self.param_scaler, 'scales') else 0
        }

