"""
BFGS Hessian 역행렬 간단 테스트

목적: BFGS 정상 종료 후 hess_inv가 올바르게 제공되는지 간단히 테스트
- 조기 종료 비활성화
- 간단한 최적화 문제
- BFGS 방법 사용
"""

import numpy as np
from scipy.optimize import minimize
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStoppingWrapper:
    """
    조기 종료 Wrapper (정상 종료 방식)
    """
    
    def __init__(self, func, grad_func, patience=5, tol=1e-6):
        self.func = func
        self.grad_func = grad_func
        self.patience = patience
        self.tol = tol
        
        self.best_f = np.inf
        self.best_x = None
        self.no_improvement_count = 0
        self.func_call_count = 0
        self.grad_call_count = 0
        self.early_stopped = False
    
    def objective(self, x):
        """목적 함수 wrapper"""
        # 이미 조기 종료된 경우
        if self.early_stopped:
            return 1e10
        
        self.func_call_count += 1
        current_f = self.func(x)
        
        # 개선 체크
        if current_f < self.best_f - self.tol:
            self.best_f = current_f
            self.best_x = x.copy()
            self.no_improvement_count = 0
            logger.info(f"Iter {self.func_call_count}: f = {current_f:.6f} [NEW BEST]")
        else:
            self.no_improvement_count += 1
            logger.info(f"Iter {self.func_call_count}: f = {current_f:.6f} (no improvement: {self.no_improvement_count}/{self.patience})")
        
        # 조기 종료 조건
        if self.no_improvement_count >= self.patience:
            self.early_stopped = True
            logger.info(f"조기 종료: {self.patience}회 연속 개선 없음 (Best f={self.best_f:.6f})")
            return 1e10
        
        return current_f
    
    def gradient(self, x):
        """그래디언트 wrapper"""
        if self.early_stopped:
            return np.zeros_like(x)
        
        self.grad_call_count += 1
        return self.grad_func(x)
    
    def callback(self, xk):
        """Callback"""
        if self.early_stopped and self.best_x is not None:
            xk[:] = self.best_x


def test_case_1_rosenbrock():
    """
    테스트 케이스 1: Rosenbrock 함수 (조기 종료 없음)
    """
    logger.info("=" * 80)
    logger.info("테스트 케이스 1: Rosenbrock 함수 (조기 종료 비활성화)")
    logger.info("=" * 80)
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dfdx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dfdx1 = 200 * (x[1] - x[0]**2)
        return np.array([dfdx0, dfdx1])
    
    x0 = np.array([0.0, 0.0])
    
    logger.info(f"초기값: {x0}")
    logger.info(f"초기 함수값: {rosenbrock(x0):.6f}")
    
    # BFGS 실행 (조기 종료 없음)
    result = minimize(
        rosenbrock,
        x0,
        method='BFGS',
        jac=rosenbrock_grad,
        options={
            'maxiter': 100,
            'ftol': 1e-3,  # 상대적 변화 0.1%
            'gtol': 1e-3,
            'disp': True
        }
    )
    
    logger.info("\n결과:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Message: {result.message}")
    logger.info(f"  Iterations: {result.nit}")
    logger.info(f"  Function evaluations: {result.nfev}")
    logger.info(f"  Final x: {result.x}")
    logger.info(f"  Final f: {result.fun:.6f}")
    
    # Hessian 역행렬 확인
    logger.info("\nHessian 역행렬 확인:")
    if hasattr(result, 'hess_inv'):
        if result.hess_inv is not None:
            logger.info(f"  ✅ result.hess_inv 존재!")
            logger.info(f"  Type: {type(result.hess_inv)}")
            logger.info(f"  Shape: {result.hess_inv.shape}")
            logger.info(f"  Values:\n{result.hess_inv}")
            
            # 대각 원소
            diag = np.diag(result.hess_inv)
            logger.info(f"  대각 원소: {diag}")
            
            # 표준오차
            se = np.sqrt(np.abs(diag))
            logger.info(f"  표준오차: {se}")
            
            logger.info("\n✅ 테스트 1 성공: BFGS가 정상 종료 후 hess_inv를 제공했습니다!")
        else:
            logger.error("  ❌ result.hess_inv가 None입니다!")
    else:
        logger.error("  ❌ result에 hess_inv 속성이 없습니다!")


def test_case_2_early_stopping():
    """
    테스트 케이스 2: 조기 종료 활성화
    """
    logger.info("\n" + "=" * 80)
    logger.info("테스트 케이스 2: Rosenbrock 함수 (조기 종료 활성화)")
    logger.info("=" * 80)
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dfdx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dfdx1 = 200 * (x[1] - x[0]**2)
        return np.array([dfdx0, dfdx1])
    
    x0 = np.array([0.0, 0.0])
    
    logger.info(f"초기값: {x0}")
    logger.info(f"초기 함수값: {rosenbrock(x0):.6f}")
    logger.info(f"조기 종료: patience=5, tol=1e-6")
    
    # Wrapper 생성
    wrapper = EarlyStoppingWrapper(
        func=rosenbrock,
        grad_func=rosenbrock_grad,
        patience=5,
        tol=1e-6
    )
    
    # BFGS 실행
    result = minimize(
        wrapper.objective,
        x0,
        method='BFGS',
        jac=wrapper.gradient,
        callback=wrapper.callback,
        options={
            'maxiter': 100,
            'ftol': 1e-3,  # 상대적 변화 0.1%
            'gtol': 1e-3,
            'disp': True
        }
    )
    
    # 조기 종료된 경우 최적 파라미터로 복원
    if wrapper.early_stopped:
        from scipy.optimize import OptimizeResult
        result = OptimizeResult(
            x=wrapper.best_x,
            success=True,
            message=f"Early stopping: {wrapper.patience}회 연속 개선 없음",
            fun=wrapper.best_f,
            nit=wrapper.func_call_count,
            nfev=wrapper.func_call_count,
            njev=wrapper.grad_call_count,
            hess_inv=None
        )
    
    logger.info("\n결과:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Message: {result.message}")
    logger.info(f"  조기 종료: {wrapper.early_stopped}")
    logger.info(f"  Function evaluations: {wrapper.func_call_count}")
    logger.info(f"  Final x: {result.x}")
    logger.info(f"  Final f: {result.fun:.6f}")
    
    # Hessian 역행렬 확인
    logger.info("\nHessian 역행렬 확인:")
    if hasattr(result, 'hess_inv'):
        if result.hess_inv is not None:
            logger.info(f"  ✅ result.hess_inv 존재!")
            logger.info(f"  Type: {type(result.hess_inv)}")
            logger.info(f"  Shape: {result.hess_inv.shape}")
            logger.info(f"  Values:\n{result.hess_inv}")
            
            logger.info("\n✅ 테스트 2 성공: 조기 종료 후에도 hess_inv를 제공했습니다!")
        else:
            logger.warning("  ⚠️ result.hess_inv가 None입니다 (조기 종료 시 예상됨)")
            logger.info("  조기 종료 시에는 BFGS hess_inv가 제공되지 않을 수 있습니다.")
    else:
        logger.error("  ❌ result에 hess_inv 속성이 없습니다!")


def test_case_3_quadratic():
    """
    테스트 케이스 3: 간단한 이차 함수 (빠른 수렴)
    """
    logger.info("\n" + "=" * 80)
    logger.info("테스트 케이스 3: 이차 함수 (빠른 수렴)")
    logger.info("=" * 80)
    
    # f(x) = x^T A x + b^T x
    A = np.array([[2.0, 0.5], [0.5, 3.0]])
    b = np.array([1.0, -2.0])
    
    def quadratic(x):
        return 0.5 * x.T @ A @ x + b.T @ x
    
    def quadratic_grad(x):
        return A @ x + b
    
    x0 = np.array([5.0, 5.0])
    
    logger.info(f"초기값: {x0}")
    logger.info(f"초기 함수값: {quadratic(x0):.6f}")
    
    # BFGS 실행
    result = minimize(
        quadratic,
        x0,
        method='BFGS',
        jac=quadratic_grad,
        options={
            'maxiter': 50,
            'ftol': 1e-3,
            'gtol': 1e-3,
            'disp': True
        }
    )
    
    logger.info("\n결과:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Iterations: {result.nit}")
    logger.info(f"  Final x: {result.x}")
    logger.info(f"  Final f: {result.fun:.6f}")
    
    # Hessian 역행렬 확인
    logger.info("\nHessian 역행렬 확인:")
    if hasattr(result, 'hess_inv'):
        if result.hess_inv is not None:
            logger.info(f"  ✅ result.hess_inv 존재!")
            logger.info(f"  Shape: {result.hess_inv.shape}")
            logger.info(f"  Values:\n{result.hess_inv}")
            
            # 실제 Hessian 역행렬과 비교
            true_hess_inv = np.linalg.inv(A)
            logger.info(f"\n  실제 Hessian 역행렬:\n{true_hess_inv}")
            logger.info(f"\n  차이:\n{result.hess_inv - true_hess_inv}")
            logger.info(f"  최대 차이: {np.max(np.abs(result.hess_inv - true_hess_inv)):.6e}")
            
            logger.info("\n✅ 테스트 3 성공!")
        else:
            logger.error("  ❌ result.hess_inv가 None입니다!")
    else:
        logger.error("  ❌ result에 hess_inv 속성이 없습니다!")


if __name__ == '__main__':
    logger.info("BFGS Hessian 역행렬 테스트 시작\n")
    
    # 테스트 1: 조기 종료 없음
    test_case_1_rosenbrock()
    
    # 테스트 2: 조기 종료 활성화
    test_case_2_early_stopping()
    
    # 테스트 3: 이차 함수 (빠른 수렴)
    test_case_3_quadratic()
    
    logger.info("\n" + "=" * 80)
    logger.info("모든 테스트 완료!")
    logger.info("=" * 80)

