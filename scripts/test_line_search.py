"""
Line Search 동작 확인 유닛 테스트

scipy L-BFGS-B의 line search가 maxls 옵션을 제대로 적용하는지 확인합니다.
"""

import numpy as np
from scipy import optimize
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class LineSearchMonitor:
    """Line search 과정을 모니터링하는 클래스"""
    
    def __init__(self):
        self.func_call_count = 0
        self.grad_call_count = 0
        self.major_iter_count = 0
        self.line_search_call_count = 0
        self.current_major_iter_start_call = 0
        self.func_values = []
        self.grad_norms = []
        
    def reset_line_search(self):
        """Major iteration 시작 시 line search 카운터 리셋"""
        self.current_major_iter_start_call = self.func_call_count
        self.line_search_call_count = 0
        
    def objective_function(self, x):
        """목적 함수 (Rosenbrock 함수)"""
        self.func_call_count += 1
        
        # Line search 중인지 판단
        calls_since_major_start = self.func_call_count - self.current_major_iter_start_call
        
        if calls_since_major_start == 1:
            context = f"Major Iteration #{self.major_iter_count + 1} 시작"
            self.line_search_call_count = 0
        elif calls_since_major_start > 1:
            self.line_search_call_count += 1
            context = f"Line Search 함수 호출 #{self.line_search_call_count}"
        else:
            context = "초기 함수값 계산"
        
        # Rosenbrock 함수 계산
        f = optimize.rosen(x)
        
        self.func_values.append(f)
        
        logger.info(f"[{context}] 함수 호출 #{self.func_call_count}: f = {f:.6f}")
        
        return f
    
    def gradient_function(self, x):
        """Gradient 함수"""
        self.grad_call_count += 1
        
        grad = optimize.rosen_der(x)
        grad_norm = np.linalg.norm(grad)
        
        self.grad_norms.append(grad_norm)
        
        logger.info(f"[Gradient 계산 #{self.grad_call_count}] ||∇f|| = {grad_norm:.6e}")
        
        return grad
    
    def callback(self, xk):
        """Major iteration 완료 시 호출"""
        self.major_iter_count += 1
        
        f_current = self.objective_function(xk)
        
        logger.info(
            f"\n{'='*80}\n"
            f"[Major Iteration #{self.major_iter_count} 완료]\n"
            f"  함수값: {f_current:.6f}\n"
            f"  Line Search 함수 호출: {self.line_search_call_count}회\n"
            f"  총 함수 호출: {self.func_call_count}회\n"
            f"  총 Gradient 호출: {self.grad_call_count}회\n"
            f"{'='*80}\n"
        )
        
        # 다음 major iteration 준비
        self.reset_line_search()


def test_line_search_maxls(maxls_value=10):
    """
    Line search maxls 옵션 테스트
    
    Parameters:
    -----------
    maxls_value : int
        Line search 최대 횟수
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Line Search 테스트: maxls = {maxls_value}")
    logger.info(f"{'#'*80}\n")
    
    # 모니터 생성
    monitor = LineSearchMonitor()
    
    # 초기값 (Rosenbrock 함수의 최적값에서 멀리 떨어진 값)
    x0 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    
    logger.info(f"초기값: {x0}")
    logger.info(f"초기 함수값: {optimize.rosen(x0):.6f}")
    logger.info(f"초기 Gradient norm: {np.linalg.norm(optimize.rosen_der(x0)):.6e}\n")
    
    # 초기 함수 호출 카운터 설정
    monitor.current_major_iter_start_call = monitor.func_call_count
    
    # BFGS 최적화 실행
    result = optimize.minimize(
        monitor.objective_function,
        x0,
        method='BFGS',  # L-BFGS-B 대신 BFGS 사용 (더 명확한 로깅)
        jac=monitor.gradient_function,
        callback=monitor.callback,
        options={
            'maxiter': 5,  # 5번의 major iteration만 실행
            'gtol': 1e-5,
            'disp': True
        }
    )
    
    # 결과 요약
    logger.info(f"\n{'#'*80}")
    logger.info(f"# 최적화 결과 요약")
    logger.info(f"{'#'*80}")
    logger.info(f"성공 여부: {result.success}")
    logger.info(f"종료 메시지: {result.message}")
    logger.info(f"Major iterations: {monitor.major_iter_count}")
    logger.info(f"총 함수 호출: {monitor.func_call_count}회")
    logger.info(f"총 Gradient 호출: {monitor.grad_call_count}회")
    logger.info(f"최종 함수값: {result.fun:.6f}")
    logger.info(f"최종 해: {result.x}")
    
    # Line search 통계
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Line Search 통계")
    logger.info(f"{'#'*80}")
    logger.info(f"Major iteration당 평균 함수 호출: {monitor.func_call_count / max(monitor.major_iter_count, 1):.2f}회")
    
    return monitor


def test_line_search_comparison():
    """
    여러 maxls 값으로 line search 비교 테스트
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Line Search maxls 비교 테스트")
    logger.info(f"{'='*80}\n")
    
    # 다양한 maxls 값 테스트
    maxls_values = [5, 10, 20]
    
    results = {}
    
    for maxls in maxls_values:
        logger.info(f"\n\n{'='*80}")
        logger.info(f"테스트: maxls = {maxls}")
        logger.info(f"{'='*80}\n")
        
        monitor = LineSearchMonitor()
        x0 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        
        monitor.current_major_iter_start_call = monitor.func_call_count
        
        # L-BFGS-B로 테스트 (maxls 옵션 지원)
        result = optimize.minimize(
            monitor.objective_function,
            x0,
            method='L-BFGS-B',
            jac=monitor.gradient_function,
            callback=monitor.callback,
            options={
                'maxiter': 3,
                'maxls': maxls,
                'gtol': 1e-5,
                'disp': False
            }
        )
        
        results[maxls] = {
            'major_iters': monitor.major_iter_count,
            'func_calls': monitor.func_call_count,
            'grad_calls': monitor.grad_call_count,
            'final_f': result.fun,
            'success': result.success
        }
        
        logger.info(f"\nmaxls={maxls} 결과:")
        logger.info(f"  Major iterations: {monitor.major_iter_count}")
        logger.info(f"  총 함수 호출: {monitor.func_call_count}회")
        logger.info(f"  총 Gradient 호출: {monitor.grad_call_count}회")
        logger.info(f"  최종 함수값: {result.fun:.6f}")
    
    # 비교 요약
    logger.info(f"\n\n{'='*80}")
    logger.info(f"비교 요약")
    logger.info(f"{'='*80}")
    logger.info(f"{'maxls':<10} {'Major Iter':<15} {'함수 호출':<15} {'Gradient 호출':<15} {'최종 함수값':<15}")
    logger.info(f"{'-'*80}")
    
    for maxls in maxls_values:
        r = results[maxls]
        logger.info(
            f"{maxls:<10} {r['major_iters']:<15} {r['func_calls']:<15} "
            f"{r['grad_calls']:<15} {r['final_f']:<15.6f}"
        )
    
    return results


if __name__ == "__main__":
    # 테스트 1: 단일 maxls 값으로 상세 테스트
    logger.info("="*80)
    logger.info("테스트 1: BFGS Line Search 상세 모니터링")
    logger.info("="*80)
    monitor = test_line_search_maxls(maxls_value=10)
    
    # 테스트 2: 여러 maxls 값 비교
    logger.info("\n\n")
    logger.info("="*80)
    logger.info("테스트 2: L-BFGS-B maxls 비교")
    logger.info("="*80)
    results = test_line_search_comparison()
    
    logger.info("\n\n테스트 완료!")

