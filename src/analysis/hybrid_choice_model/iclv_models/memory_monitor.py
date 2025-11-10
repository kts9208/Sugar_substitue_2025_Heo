"""
메모리 모니터링 및 관리 유틸리티

GPU 및 CPU 메모리 사용량을 모니터링하고 과부하를 방지합니다.
"""

import psutil
import gc
import logging
from typing import Optional, Dict
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """메모리 사용량 모니터링 및 관리"""
    
    def __init__(self, 
                 cpu_threshold_mb: float = 1000,  # CPU 메모리 임계값 (MB)
                 gpu_threshold_mb: float = 1000,  # GPU 메모리 임계값 (MB)
                 auto_cleanup: bool = True,       # 자동 정리 활성화
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            cpu_threshold_mb: CPU 메모리 임계값 (MB) - 이 값 이상 사용 시 경고
            gpu_threshold_mb: GPU 메모리 임계값 (MB) - 이 값 이상 사용 시 경고
            auto_cleanup: 임계값 초과 시 자동으로 가비지 컬렉션 수행
            logger: 로거 인스턴스
        """
        self.cpu_threshold_mb = cpu_threshold_mb
        self.gpu_threshold_mb = gpu_threshold_mb
        self.auto_cleanup = auto_cleanup
        self.logger = logger or logging.getLogger(__name__)
        
        # 메모리 사용 기록
        self.cpu_memory_history = []
        self.gpu_memory_history = []
        
        # 프로세스 정보
        self.process = psutil.Process()
        
    def get_cpu_memory_mb(self) -> float:
        """현재 프로세스의 CPU 메모리 사용량 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_mb(self) -> Optional[float]:
        """현재 GPU 메모리 사용량 (MB)"""
        # nvidia-smi 사용 (가장 정확)
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                nvidia_mem_mb = float(result.stdout.strip().split('\n')[0])
                return nvidia_mem_mb
        except:
            pass

        # nvidia-smi 실패 시 CuPy 사용
        if not CUPY_AVAILABLE:
            return None

        try:
            mempool = cp.get_default_memory_pool()
            # used_bytes()와 total_bytes() 모두 확인
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()
            # 둘 중 큰 값 사용
            gpu_mem_mb = max(used_bytes, total_bytes) / 1024 / 1024
            return gpu_mem_mb
        except Exception as e:
            return None

    def get_gpu_total_memory_mb(self) -> Optional[float]:
        """GPU 전체 메모리 용량 (MB)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                total_mem_mb = float(result.stdout.strip().split('\n')[0])
                return total_mem_mb
        except:
            pass

        return None
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """시스템 전체 메모리 정보"""
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / 1024 / 1024,
            'available_mb': mem.available / 1024 / 1024,
            'used_mb': mem.used / 1024 / 1024,
            'percent': mem.percent
        }
    
    def check_and_cleanup(self, context: str = "") -> Dict[str, float]:
        """
        메모리 사용량 체크 및 필요시 정리
        
        Args:
            context: 로그 메시지에 포함할 컨텍스트 정보
            
        Returns:
            메모리 사용량 정보 딕셔너리
        """
        cpu_mem = self.get_cpu_memory_mb()
        gpu_mem = self.get_gpu_memory_mb()
        
        # 기록 저장
        self.cpu_memory_history.append(cpu_mem)
        if gpu_mem is not None:
            self.gpu_memory_history.append(gpu_mem)
        
        # 임계값 체크
        cpu_exceeded = cpu_mem > self.cpu_threshold_mb
        gpu_exceeded = gpu_mem is not None and gpu_mem > self.gpu_threshold_mb
        
        if cpu_exceeded or gpu_exceeded:
            msg = f"[메모리 경고] {context}"
            if cpu_exceeded:
                msg += f" | CPU: {cpu_mem:.1f}MB (임계값: {self.cpu_threshold_mb}MB)"
            if gpu_exceeded:
                msg += f" | GPU: {gpu_mem:.1f}MB (임계값: {self.gpu_threshold_mb}MB)"
            
            self.logger.warning(msg)
            
            # 자동 정리
            if self.auto_cleanup:
                self.cleanup_memory()
                
                # 정리 후 재확인
                cpu_mem_after = self.get_cpu_memory_mb()
                gpu_mem_after = self.get_gpu_memory_mb()
                
                freed_cpu = cpu_mem - cpu_mem_after
                freed_gpu = (gpu_mem - gpu_mem_after) if gpu_mem is not None and gpu_mem_after is not None else 0
                
                self.logger.info(
                    f"[메모리 정리 완료] CPU: {freed_cpu:.1f}MB 해제, "
                    f"GPU: {freed_gpu:.1f}MB 해제"
                )
                
                cpu_mem = cpu_mem_after
                gpu_mem = gpu_mem_after
        
        return {
            'cpu_mb': cpu_mem,
            'gpu_mb': gpu_mem,
            'cpu_exceeded': cpu_exceeded,
            'gpu_exceeded': gpu_exceeded
        }
    
    def cleanup_memory(self):
        """메모리 정리 (가비지 컬렉션)"""
        # Python 가비지 컬렉션
        gc.collect()
        
        # GPU 메모리 풀 정리
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except Exception as e:
                self.logger.warning(f"GPU 메모리 정리 실패: {e}")
    
    def log_memory_stats(self, context: str = ""):
        """메모리 사용 통계 로깅"""
        cpu_mem = self.get_cpu_memory_mb()
        gpu_mem = self.get_gpu_memory_mb()
        gpu_total = self.get_gpu_total_memory_mb()
        sys_mem = self.get_system_memory_info()

        msg = f"[메모리 상태] {context}\n"
        msg += f"  프로세스 CPU: {cpu_mem:.1f}MB\n"
        msg += f"  시스템 전체: {sys_mem['used_mb']:.1f}MB / {sys_mem['total_mb']:.1f}MB ({sys_mem['percent']:.1f}%)\n"
        msg += f"  시스템 여유: {sys_mem['available_mb']:.1f}MB"

        if gpu_mem is not None:
            if gpu_total is not None:
                gpu_percent = (gpu_mem / gpu_total) * 100
                msg += f"\n  GPU: {gpu_mem:.1f}MB / {gpu_total:.1f}MB ({gpu_percent:.1f}%)"
            else:
                msg += f"\n  GPU: {gpu_mem:.1f}MB"

        self.logger.info(msg)
    
    def get_memory_summary(self) -> Dict:
        """메모리 사용 요약 정보"""
        summary = {
            'current_cpu_mb': self.get_cpu_memory_mb(),
            'current_gpu_mb': self.get_gpu_memory_mb(),
        }
        
        if self.cpu_memory_history:
            summary['cpu_max_mb'] = max(self.cpu_memory_history)
            summary['cpu_avg_mb'] = np.mean(self.cpu_memory_history)
        
        if self.gpu_memory_history:
            summary['gpu_max_mb'] = max(self.gpu_memory_history)
            summary['gpu_avg_mb'] = np.mean(self.gpu_memory_history)
        
        return summary


def cleanup_arrays(*arrays):
    """
    배열들을 명시적으로 삭제하고 메모리 정리
    
    Args:
        *arrays: 삭제할 배열들
    """
    for arr in arrays:
        if arr is not None:
            del arr
    
    gc.collect()
    
    if CUPY_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        except:
            pass


def get_array_memory_mb(arr) -> float:
    """배열의 메모리 사용량 (MB) 계산"""
    if arr is None:
        return 0.0
    
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr.nbytes / 1024 / 1024
    elif isinstance(arr, np.ndarray):
        return arr.nbytes / 1024 / 1024
    else:
        return 0.0

