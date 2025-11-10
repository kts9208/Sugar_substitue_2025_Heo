"""
메모리 모니터 단독 테스트
"""

import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.memory_monitor import MemoryMonitor

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("메모리 모니터 테스트")
print("="*70)

# 메모리 모니터 생성
monitor = MemoryMonitor(
    cpu_threshold_mb=1000,
    gpu_threshold_mb=500,
    auto_cleanup=True,
    logger=logger
)

print("\n1. 초기 메모리 상태")
monitor.log_memory_stats("초기 상태")

print("\n2. CPU 메모리 확인")
cpu_mem = monitor.get_cpu_memory_mb()
print(f"   CPU 메모리: {cpu_mem:.1f}MB")

print("\n3. GPU 메모리 확인")
gpu_mem = monitor.get_gpu_memory_mb()
if gpu_mem is not None:
    print(f"   GPU 메모리: {gpu_mem:.1f}MB")
else:
    print("   GPU 메모리: 사용 불가")

print("\n4. 시스템 메모리 확인")
sys_mem = monitor.get_system_memory_info()
print(f"   전체: {sys_mem['total_mb']:.1f}MB")
print(f"   사용: {sys_mem['used_mb']:.1f}MB ({sys_mem['percent']:.1f}%)")
print(f"   여유: {sys_mem['available_mb']:.1f}MB")

print("\n5. 대용량 배열 생성")
import numpy as np
large_array = np.random.rand(5000, 5000)  # ~200MB
print(f"   배열 크기: {large_array.nbytes / 1024 / 1024:.1f}MB")

print("\n6. 배열 생성 후 메모리 상태")
monitor.log_memory_stats("배열 생성 후")

print("\n7. 메모리 체크 및 정리")
mem_info = monitor.check_and_cleanup("대용량 배열 생성 후")
print(f"   CPU: {mem_info['cpu_mb']:.1f}MB (초과: {mem_info['cpu_exceeded']})")
if mem_info['gpu_mb'] is not None:
    print(f"   GPU: {mem_info['gpu_mb']:.1f}MB (초과: {mem_info['gpu_exceeded']})")

print("\n8. 배열 삭제")
del large_array

print("\n9. 정리 후 메모리 상태")
monitor.log_memory_stats("정리 후")

print("\n10. 메모리 요약")
summary = monitor.get_memory_summary()
print(f"   현재 CPU: {summary['current_cpu_mb']:.1f}MB")
if summary['current_gpu_mb'] is not None:
    print(f"   현재 GPU: {summary['current_gpu_mb']:.1f}MB")
if 'cpu_max_mb' in summary:
    print(f"   최대 CPU: {summary['cpu_max_mb']:.1f}MB")
    print(f"   평균 CPU: {summary['cpu_avg_mb']:.1f}MB")

print("\n" + "="*70)
print("테스트 완료")
print("="*70)

