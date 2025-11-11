"""
Apollo-style Parameter Scaling for ICLV Models

This module implements parameter scaling similar to Apollo R package,
which scales parameters to improve numerical stability during optimization.

Author: Based on Apollo R package methodology
Date: 2025-01-10
"""

import numpy as np
from typing import Dict, Optional
import logging


class ParameterScaler:
    """
    Apollo-style parameter scaler
    
    Scales parameters based on their initial values to improve
    numerical stability during optimization. This is similar to
    the scaling mechanism used in Apollo R package.
    
    Scaling formula:
        θ_internal = θ_external / scale
        θ_external = θ_internal * scale
        
    Gradient scaling (chain rule):
        ∂LL/∂θ_internal = ∂LL/∂θ_external * scale
    """
    
    def __init__(self, initial_params: np.ndarray, param_names: list, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize parameter scaler
        
        Args:
            initial_params: Initial parameter values (1D array)
            param_names: List of parameter names
            logger: Optional logger for debugging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.param_names = param_names
        self.n_params = len(initial_params)
        
        # Compute scale factors based on initial parameter values
        # Apollo R 방식 (scaleHessian = TRUE, default):
        # - 초기값이 0이 아닌 파라미터: abs(initial_value)로 스케일링
        # - 초기값이 0인 파라미터: 1.0으로 스케일링 (스케일링 안함)
        self.scales = np.ones(self.n_params)

        for i, (name, value) in enumerate(zip(param_names, initial_params)):
            if abs(value) > 1e-10:
                # Use absolute value of initial parameter as scale
                self.scales[i] = abs(value)
            else:
                # Apollo R 방식: 초기값이 0인 파라미터는 1.0 (스케일링 안함)
                self.scales[i] = 1.0
        
        # Log scaling information
        self.logger.info("=" * 80)
        self.logger.info("Apollo-style Parameter Scaling Initialized")
        self.logger.info("=" * 80)
        self.logger.info(f"Total parameters: {self.n_params}")
        self.logger.info("")
        self.logger.info("Scale factors:")
        for i, (name, scale) in enumerate(zip(param_names, self.scales)):
            self.logger.info(f"  {name:30s}: {scale:12.6f}")
        self.logger.info("=" * 80)
    
    def scale_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Scale parameters for optimization (External → Internal)
        
        θ_internal = θ_external / scale
        
        This makes parameters closer to O(1) for optimization.
        
        Args:
            params: External parameter values (1D array)
            
        Returns:
            Scaled (internal) parameter values
        """
        return params / self.scales
    
    def unscale_parameters(self, params_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale parameters for likelihood calculation (Internal → External)
        
        θ_external = θ_internal * scale
        
        Args:
            params_scaled: Internal (scaled) parameter values
            
        Returns:
            Unscaled (external) parameter values
        """
        return params_scaled * self.scales
    
    def scale_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Scale gradient using chain rule
        
        ∂LL/∂θ_internal = ∂LL/∂θ_external * scale
        
        Because:
            θ_external = θ_internal * scale
            ∂θ_external/∂θ_internal = scale
            
        By chain rule:
            ∂LL/∂θ_internal = (∂LL/∂θ_external) * (∂θ_external/∂θ_internal)
                             = (∂LL/∂θ_external) * scale
        
        Args:
            grad: Gradient with respect to external parameters
            
        Returns:
            Gradient with respect to internal (scaled) parameters
        """
        return grad * self.scales
    
    def get_scale_info(self) -> Dict[str, float]:
        """
        Get scaling information as dictionary
        
        Returns:
            Dictionary mapping parameter names to scale factors
        """
        return {name: scale for name, scale in zip(self.param_names, self.scales)}
    
    def log_parameter_comparison(self, params_external: np.ndarray, 
                                  params_internal: np.ndarray):
        """
        Log comparison between external and internal parameters
        
        Args:
            params_external: External (unscaled) parameters
            params_internal: Internal (scaled) parameters
        """
        self.logger.info("")
        self.logger.info("Parameter Scaling Comparison:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Parameter':<30s} {'External':>12s} {'Internal':>12s} {'Scale':>12s}")
        self.logger.info("-" * 80)
        
        for i, name in enumerate(self.param_names):
            self.logger.info(
                f"{name:<30s} {params_external[i]:12.6f} {params_internal[i]:12.6f} "
                f"{self.scales[i]:12.6f}"
            )
        self.logger.info("-" * 80)
    
    def log_gradient_comparison(self, grad_external: np.ndarray, 
                                 grad_internal: np.ndarray):
        """
        Log comparison between external and internal gradients
        
        Args:
            grad_external: Gradient w.r.t. external parameters
            grad_internal: Gradient w.r.t. internal parameters
        """
        self.logger.info("")
        self.logger.info("Gradient Scaling Comparison:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Parameter':<30s} {'Grad(External)':>15s} {'Grad(Internal)':>15s} {'Scale':>12s}")
        self.logger.info("-" * 80)
        
        for i, name in enumerate(self.param_names):
            self.logger.info(
                f"{name:<30s} {grad_external[i]:15.6e} {grad_internal[i]:15.6e} "
                f"{self.scales[i]:12.6f}"
            )
        self.logger.info("-" * 80)
        self.logger.info(f"External gradient norm: {np.linalg.norm(grad_external):.6e}")
        self.logger.info(f"Internal gradient norm: {np.linalg.norm(grad_internal):.6e}")
        self.logger.info(f"External gradient max:  {np.max(np.abs(grad_external)):.6e}")
        self.logger.info(f"Internal gradient max:  {np.max(np.abs(grad_internal)):.6e}")
        self.logger.info("-" * 80)


class AdaptiveParameterScaler(ParameterScaler):
    """
    Adaptive parameter scaler based on initial gradient magnitudes
    
    This scaler computes scale factors based on the magnitude of
    initial gradients, aiming to make all gradients approximately
    the same order of magnitude.
    """
    
    def __init__(self, initial_params: np.ndarray, initial_gradients: np.ndarray,
                 param_names: list, target_grad_magnitude: float = 1000.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize adaptive parameter scaler
        
        Args:
            initial_params: Initial parameter values
            initial_gradients: Initial gradient values
            param_names: List of parameter names
            target_grad_magnitude: Target gradient magnitude (default: 1000.0)
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.param_names = param_names
        self.n_params = len(initial_params)
        self.target_grad_magnitude = target_grad_magnitude
        
        # Compute scale factors based on gradient magnitudes
        self.scales = np.ones(self.n_params)
        
        for i, (name, grad) in enumerate(zip(param_names, initial_gradients)):
            grad_mag = abs(grad)
            
            if grad_mag > 1e-10:
                # Scale to make gradient ~target_grad_magnitude
                self.scales[i] = grad_mag / target_grad_magnitude
            else:
                # For zero gradients, use scale = 1.0
                self.scales[i] = 1.0
        
        # Log scaling information
        self.logger.info("=" * 80)
        self.logger.info("Adaptive Parameter Scaling Initialized")
        self.logger.info("=" * 80)
        self.logger.info(f"Total parameters: {self.n_params}")
        self.logger.info(f"Target gradient magnitude: {target_grad_magnitude:.2f}")
        self.logger.info("")
        self.logger.info("Scale factors (based on initial gradients):")
        for i, (name, scale, grad) in enumerate(zip(param_names, self.scales, initial_gradients)):
            self.logger.info(
                f"  {name:30s}: scale={scale:12.6f}, initial_grad={grad:15.6e}"
            )
        self.logger.info("=" * 80)

