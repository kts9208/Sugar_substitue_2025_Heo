"""
Data Standardization for ICLV Models

This module implements z-score standardization for choice attributes
to improve numerical stability during optimization.

Author: Sugar Substitute Research Team
Date: 2025-01-22
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging


class DataStandardizer:
    """
    Z-score standardization for choice attributes
    
    Standardizes data variables (e.g., price, health_label) to have
    mean 0 and standard deviation 1, improving numerical stability
    during optimization.
    
    Standardization formula:
        z = (x - mean(x)) / std(x)
        
    Inverse transformation:
        x = z * std(x) + mean(x)
    
    This is separate from parameter scaling (ParameterScaler),
    which operates in the optimization space.
    """
    
    def __init__(self, variables_to_standardize: List[str],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize data standardizer
        
        Args:
            variables_to_standardize: List of variable names to standardize
                Example: ['price', 'health_label']
            logger: Optional logger for debugging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.variables_to_standardize = variables_to_standardize
        self.stats = {}  # {var_name: {'mean': float, 'std': float}}
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'DataStandardizer':
        """
        Compute mean and standard deviation from data
        
        Args:
            data: DataFrame containing variables to standardize
            
        Returns:
            self (for method chaining)
        """
        self.logger.info("=" * 80)
        self.logger.info("데이터 변수 Z-score 표준화 파라미터 계산")
        self.logger.info("=" * 80)
        
        self.stats = {}
        
        for var in self.variables_to_standardize:
            if var not in data.columns:
                self.logger.warning(f"  변수 '{var}'가 데이터에 없습니다. 건너뜁니다.")
                continue
            
            # NaN 제외하고 통계 계산
            values = data[var].dropna()
            
            if len(values) == 0:
                self.logger.warning(f"  변수 '{var}'에 유효한 값이 없습니다. 건너뜁니다.")
                continue
            
            mean = values.mean()
            std = values.std(ddof=0)  # 모집단 표준편차 (N으로 나눔)
            
            self.stats[var] = {
                'mean': mean,
                'std': std,
                'min': values.min(),
                'max': values.max(),
                'n_valid': len(values)
            }
            
            self.logger.info(
                f"  {var:20s}: mean={mean:10.4f}, std={std:10.4f}, "
                f"range=[{values.min():10.2f}, {values.max():10.2f}], n={len(values)}"
            )
        
        self.is_fitted = True
        self.logger.info("=" * 80)
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score standardization to data
        
        Args:
            data: DataFrame to standardize
            
        Returns:
            Standardized DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("DataStandardizer must be fitted before transform. Call fit() first.")
        
        df = data.copy()
        
        for var, params in self.stats.items():
            if var not in df.columns:
                self.logger.warning(f"  변수 '{var}'가 데이터에 없습니다. 건너뜁니다.")
                continue
            
            mean = params['mean']
            std = params['std']
            
            # NaN이 아닌 값만 표준화
            mask = df[var].notna()
            
            if std > 1e-10:  # 표준편차가 0이 아닌 경우만
                df.loc[mask, var] = (df.loc[mask, var] - mean) / std
            else:
                # 표준편차가 0에 가까우면 중심화만 적용
                self.logger.warning(
                    f"  변수 '{var}': 표준편차가 0에 가까워 중심화만 적용 (std={std:.6e})"
                )
                df.loc[mask, var] = df.loc[mask, var] - mean
        
        return df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            data: DataFrame to fit and standardize
            
        Returns:
            Standardized DataFrame
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse z-score standardization (standardized → original scale)
        
        Args:
            data: Standardized DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("DataStandardizer must be fitted before inverse_transform.")
        
        df = data.copy()
        
        for var, params in self.stats.items():
            if var not in df.columns:
                continue
            
            mean = params['mean']
            std = params['std']
            
            # NaN이 아닌 값만 역변환
            mask = df[var].notna()
            
            if std > 1e-10:
                df.loc[mask, var] = df.loc[mask, var] * std + mean
            else:
                df.loc[mask, var] = df.loc[mask, var] + mean
        
        return df
    
    def get_standardization_params(self) -> Dict[str, Dict[str, float]]:
        """
        Get standardization parameters
        
        Returns:
            Dictionary mapping variable names to their statistics
            Example: {'price': {'mean': 4000.0, 'std': 1000.0, ...}, ...}
        """
        return self.stats.copy()
    
    def log_standardization_comparison(self, data_original: pd.DataFrame,
                                        data_standardized: pd.DataFrame):
        """
        Log comparison between original and standardized data

        Args:
            data_original: Original data
            data_standardized: Standardized data
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("데이터 표준화 전후 비교")
        self.logger.info("=" * 80)

        # 표준화 파라미터 출력
        self.logger.info("\n[표준화 파라미터]")
        self.logger.info("-" * 80)
        self.logger.info(f"{'변수':<20s} {'평균 (mean)':>15s} {'표준편차 (std)':>15s} {'샘플 수':>10s}")
        self.logger.info("-" * 80)

        for var, params in self.stats.items():
            n_samples = params.get('n_valid', params.get('n_samples', 0))
            self.logger.info(
                f"{var:<20s} {params['mean']:15.4f} {params['std']:15.4f} {n_samples:10d}"
            )

        self.logger.info("-" * 80)

        # 원본 vs 표준화 비교
        self.logger.info("\n[원본 데이터 vs 표준화 데이터]")
        self.logger.info("-" * 80)
        self.logger.info(f"{'변수':<20s} {'원본 평균':>12s} {'원본 표준편차':>12s} "
                        f"{'표준화 평균':>14s} {'표준화 표준편차':>14s}")
        self.logger.info("-" * 80)

        for var in self.stats.keys():
            if var not in data_original.columns or var not in data_standardized.columns:
                continue

            orig_values = data_original[var].dropna()
            std_values = data_standardized[var].dropna()

            self.logger.info(
                f"{var:<20s} {orig_values.mean():12.4f} {orig_values.std():12.4f} "
                f"{std_values.mean():14.8f} {std_values.std():14.8f}"
            )

        self.logger.info("-" * 80)

        # 샘플 데이터 출력 (처음 5개)
        self.logger.info("\n[샘플 데이터 비교 (처음 5개 행)]")
        self.logger.info("-" * 80)

        for var in self.stats.keys():
            if var not in data_original.columns or var not in data_standardized.columns:
                continue

            self.logger.info(f"\n변수: {var}")
            self.logger.info(f"  원본:     {list(data_original[var].head().values)}")
            self.logger.info(f"  표준화:   {list(data_standardized[var].head().values)}")

        self.logger.info("\n" + "=" * 80)

