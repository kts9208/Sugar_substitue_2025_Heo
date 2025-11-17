"""
GPU 계산 상태 관리 모듈

GPU 병렬화 상태를 명시적으로 관리하여 조건 분기 로직을 단순화합니다.
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUComputeState:
    """
    GPU 계산 상태를 명시적으로 관리하는 클래스
    
    Attributes:
        enabled: GPU 사용 여부
        measurement_model: GPU 측정모델 객체
        full_parallel: 완전 병렬 처리 사용 여부
    """
    enabled: bool
    measurement_model: Optional[Any]
    full_parallel: bool
    
    def is_ready(self) -> bool:
        """
        GPU 계산이 가능한 상태인지 확인
        
        Returns:
            GPU 계산 가능 여부
        """
        return self.enabled and self.measurement_model is not None
    
    def get_mode_name(self) -> str:
        """
        현재 계산 모드 이름 반환
        
        Returns:
            모드 이름 문자열
        """
        if not self.enabled:
            return "CPU"
        if self.measurement_model is None:
            return "CPU (GPU 모델 없음)"
        if self.full_parallel:
            return "GPU (완전 병렬)"
        return "GPU (배치)"
    
    def get_status_dict(self) -> dict:
        """
        상태 정보를 딕셔너리로 반환
        
        Returns:
            상태 정보 딕셔너리
        """
        return {
            'enabled': self.enabled,
            'measurement_model_available': self.measurement_model is not None,
            'full_parallel': self.full_parallel,
            'is_ready': self.is_ready(),
            'mode': self.get_mode_name()
        }
    
    def log_status(self, logger_instance: logging.Logger, prefix: str = ""):
        """
        상태 정보를 로깅
        
        Args:
            logger_instance: 로거 객체
            prefix: 로그 메시지 접두사
        """
        status = self.get_status_dict()
        logger_instance.info(f"{prefix}GPU 계산 상태:")
        logger_instance.info(f"{prefix}  모드: {status['mode']}")
        logger_instance.info(f"{prefix}  enabled: {status['enabled']}")
        logger_instance.info(f"{prefix}  measurement_model: {status['measurement_model_available']}")
        logger_instance.info(f"{prefix}  full_parallel: {status['full_parallel']}")
        logger_instance.info(f"{prefix}  is_ready: {status['is_ready']}")
    
    @classmethod
    def from_joint_gradient(cls, joint_grad, gpu_measurement_model=None):
        """
        MultiLatentJointGradient 객체로부터 상태 생성
        
        Args:
            joint_grad: MultiLatentJointGradient 객체
            gpu_measurement_model: GPU 측정모델 (선택적)
        
        Returns:
            GPUComputeState 객체
        """
        enabled = hasattr(joint_grad, 'use_gpu') and joint_grad.use_gpu
        measurement_model = gpu_measurement_model or getattr(joint_grad, 'gpu_measurement_model', None)
        full_parallel = hasattr(joint_grad, 'use_full_parallel') and joint_grad.use_full_parallel
        
        return cls(
            enabled=enabled,
            measurement_model=measurement_model,
            full_parallel=full_parallel
        )
    
    @classmethod
    def create_cpu_mode(cls):
        """
        CPU 모드 상태 생성
        
        Returns:
            CPU 모드 GPUComputeState 객체
        """
        return cls(enabled=False, measurement_model=None, full_parallel=False)
    
    @classmethod
    def create_gpu_mode(cls, measurement_model, full_parallel: bool = True):
        """
        GPU 모드 상태 생성
        
        Args:
            measurement_model: GPU 측정모델
            full_parallel: 완전 병렬 처리 사용 여부
        
        Returns:
            GPU 모드 GPUComputeState 객체
        """
        return cls(enabled=True, measurement_model=measurement_model, full_parallel=full_parallel)

