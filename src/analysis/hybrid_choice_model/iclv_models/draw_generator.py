"""
Draw Generator for Simulation-Based Estimation

시뮬레이션 기반 추정을 위한 난수 생성기입니다.
동시추정(Simultaneous Estimation)에서만 사용됩니다.

단일책임 원칙:
- Halton 시퀀스 생성
- 다차원 draws 생성 (다중 잠재변수용)
- 개인별 draws 관리
"""

import numpy as np
from typing import Optional
from scipy.stats import norm, qmc
import logging

logger = logging.getLogger(__name__)


class HaltonDrawGenerator:
    """
    Halton 시퀀스 생성기
    
    준난수(Quasi-random) 시퀀스를 생성하여 시뮬레이션 정확도를 향상시킵니다.
    일반 난수보다 공간을 더 균등하게 커버합니다.
    
    참조: Apollo 패키지의 Halton draws
    
    Usage:
        >>> # 단일 차원 (단일 잠재변수)
        >>> generator = HaltonDrawGenerator(n_draws=100, n_individuals=500)
        >>> draws = generator.get_draws()  # shape: (500, 100)
        
        >>> # 다차원 (다중 잠재변수)
        >>> generator = HaltonDrawGenerator(
        ...     n_draws=100, 
        ...     n_individuals=500,
        ...     n_dimensions=5  # 5개 잠재변수
        ... )
        >>> draws = generator.get_draws()  # shape: (500, 100, 5)
    """
    
    def __init__(self, n_draws: int, n_individuals: int, 
                 n_dimensions: int = 1,
                 scramble: bool = True, 
                 seed: Optional[int] = None):
        """
        Args:
            n_draws: 개인당 draw 수
            n_individuals: 개인 수
            n_dimensions: 차원 수 (잠재변수 개수)
            scramble: 스크램블 여부 (권장)
            seed: 난수 시드
        """
        self.n_draws = n_draws
        self.n_individuals = n_individuals
        self.n_dimensions = n_dimensions
        self.scramble = scramble
        self.seed = seed
        
        self.draws = None
        self._generate_draws()
    
    def _generate_draws(self) -> None:
        """Halton 시퀀스 생성"""
        logger.info(
            f"Halton draws 생성: {self.n_individuals} 개인 × "
            f"{self.n_draws} draws × {self.n_dimensions} 차원"
        )
        
        # scipy의 Halton 시퀀스 생성기 사용
        sampler = qmc.Halton(
            d=self.n_dimensions, 
            scramble=self.scramble, 
            seed=self.seed
        )
        
        # 균등분포 [0,1] 샘플 생성
        uniform_draws = sampler.random(n=self.n_individuals * self.n_draws)
        
        # 표준정규분포로 변환 (역누적분포함수)
        normal_draws = norm.ppf(uniform_draws)
        
        # 형태 재구성
        if self.n_dimensions == 1:
            # 단일 차원: (n_individuals, n_draws)
            self.draws = normal_draws.reshape(self.n_individuals, self.n_draws)
        else:
            # 다차원: (n_individuals, n_draws, n_dimensions)
            self.draws = normal_draws.reshape(
                self.n_individuals, self.n_draws, self.n_dimensions
            )
        
        logger.info(f"Halton draws 생성 완료: shape={self.draws.shape}")
    
    def get_draws(self) -> np.ndarray:
        """
        생성된 draws 반환
        
        Returns:
            - 단일 차원: (n_individuals, n_draws)
            - 다차원: (n_individuals, n_draws, n_dimensions)
        """
        return self.draws
    
    def get_draw_for_individual(self, individual_idx: int) -> np.ndarray:
        """
        특정 개인의 draws 반환
        
        Args:
            individual_idx: 개인 인덱스 (0-based)
        
        Returns:
            - 단일 차원: (n_draws,)
            - 다차원: (n_draws, n_dimensions)
        """
        return self.draws[individual_idx]


class RandomDrawGenerator:
    """
    일반 난수 생성기 (비교용)
    
    Halton 시퀀스 대신 일반 난수를 사용합니다.
    """
    
    def __init__(self, n_draws: int, n_individuals: int,
                 n_dimensions: int = 1,
                 seed: Optional[int] = None):
        """
        Args:
            n_draws: 개인당 draw 수
            n_individuals: 개인 수
            n_dimensions: 차원 수
            seed: 난수 시드
        """
        self.n_draws = n_draws
        self.n_individuals = n_individuals
        self.n_dimensions = n_dimensions
        self.seed = seed
        
        self.draws = None
        self._generate_draws()
    
    def _generate_draws(self) -> None:
        """일반 난수 생성"""
        logger.info(
            f"Random draws 생성: {self.n_individuals} 개인 × "
            f"{self.n_draws} draws × {self.n_dimensions} 차원"
        )
        
        rng = np.random.default_rng(self.seed)
        
        if self.n_dimensions == 1:
            self.draws = rng.standard_normal(
                size=(self.n_individuals, self.n_draws)
            )
        else:
            self.draws = rng.standard_normal(
                size=(self.n_individuals, self.n_draws, self.n_dimensions)
            )
        
        logger.info(f"Random draws 생성 완료: shape={self.draws.shape}")
    
    def get_draws(self) -> np.ndarray:
        """생성된 draws 반환"""
        return self.draws
    
    def get_draw_for_individual(self, individual_idx: int) -> np.ndarray:
        """특정 개인의 draws 반환"""
        return self.draws[individual_idx]

