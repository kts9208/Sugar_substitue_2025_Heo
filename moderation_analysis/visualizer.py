"""
Moderation Analysis Visualizer Module

조절효과 분석 결과를 시각화하는 모듈입니다.
단순기울기 그래프, 상호작용 플롯, 조절효과 히트맵 등을 생성합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import warnings

from .config import ModerationAnalysisConfig

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False


class ModerationVisualizer:
    """조절효과 분석 시각화 클래스"""
    
    def __init__(self, config: Optional[ModerationAnalysisConfig] = None):
        """
        시각화기 초기화
        
        Args:
            config (Optional[ModerationAnalysisConfig]): 분석 설정
        """
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 설정
        self._setup_plot_style()
        
        logger.info("조절효과 시각화기 초기화 완료")
    
    def _setup_plot_style(self):
        """플롯 스타일 설정"""
        try:
            plt.style.use(self.config.plot_style)
        except:
            plt.style.use('default')
            logger.warning(f"스타일 '{self.config.plot_style}'를 찾을 수 없어 기본 스타일을 사용합니다.")
        
        # 기본 설정
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
    
    def create_moderation_plot(self, data: pd.DataFrame, results: Dict[str, Any],
                             save_path: Optional[Path] = None) -> Path:
        """
        조절효과 시각화 (상호작용 플롯)
        
        Args:
            data (pd.DataFrame): 분석 데이터
            results (Dict[str, Any]): 분석 결과
            save_path (Optional[Path]): 저장 경로
            
        Returns:
            Path: 저장된 파일 경로
        """
        logger.info("조절효과 플롯 생성 시작")
        
        variables = results['variables']
        independent_var = variables['independent']
        dependent_var = variables['dependent']
        moderator_var = variables['moderator']
        
        # 플롯 생성
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # 조절변수 값들 (low, mean, high)
        moderator_values = self._get_moderator_plot_values(data, moderator_var)
        
        # 독립변수 범위
        x_range = np.linspace(data[independent_var].min(), data[independent_var].max(), 100)
        
        # 각 조절변수 수준에서의 회귀선 그리기
        colors = ['red', 'blue', 'green']
        labels = ['Low (-1SD)', 'Mean', 'High (+1SD)']
        
        for i, (level, mod_value) in enumerate(moderator_values.items()):
            y_pred = self._predict_values(x_range, mod_value, results)
            ax.plot(x_range, y_pred, color=colors[i], label=f'{labels[i]} ({mod_value:.2f})', linewidth=2)
        
        # 실제 데이터 포인트 추가 (샘플링)
        sample_data = data.sample(min(200, len(data)))  # 최대 200개 포인트
        ax.scatter(sample_data[independent_var], sample_data[dependent_var], 
                  alpha=0.3, color='gray', s=20)
        
        # 플롯 설정
        ax.set_xlabel(f'{independent_var} (독립변수)', fontsize=12)
        ax.set_ylabel(f'{dependent_var} (종속변수)', fontsize=12)
        ax.set_title(f'조절효과 분석: {moderator_var}의 조절효과', fontsize=14, fontweight='bold')
        ax.legend(title=f'{moderator_var} (조절변수)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 유의성 정보 추가
        moderation_test = results.get('moderation_test', {})
        significance_text = "유의함" if moderation_test.get('significant', False) else "유의하지 않음"
        p_value = moderation_test.get('p_value', np.nan)
        
        ax.text(0.02, 0.98, f'상호작용 효과: {significance_text} (p={p_value:.4f})', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        if save_path is None:
            save_path = self.results_dir / f"moderation_plot_{independent_var}_x_{moderator_var}.png"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"조절효과 플롯 저장: {save_path}")
        return save_path
    
    def _get_moderator_plot_values(self, data: pd.DataFrame, moderator_var: str) -> Dict[str, float]:
        """플롯용 조절변수 값들 계산"""
        moderator_data = data[moderator_var]
        mean_val = moderator_data.mean()
        std_val = moderator_data.std()
        
        return {
            'low': mean_val - std_val,
            'mean': mean_val,
            'high': mean_val + std_val
        }
    
    def _predict_values(self, x_values: np.ndarray, moderator_value: float, 
                       results: Dict[str, Any]) -> np.ndarray:
        """예측값 계산"""
        coefficients = results.get('coefficients', {})
        
        # 필요한 계수들 추출
        intercept = 0.0  # 절편은 보통 별도로 처리
        main_effect = 0.0
        moderator_effect = 0.0
        interaction_effect = 0.0
        
        variables = results['variables']
        independent_var = variables['independent']
        moderator_var = variables['moderator']
        interaction_var = variables['interaction']
        
        for var_name, coeff_info in coefficients.items():
            if var_name == independent_var:
                main_effect = coeff_info['estimate']
            elif var_name == moderator_var:
                moderator_effect = coeff_info['estimate']
            elif var_name == interaction_var:
                interaction_effect = coeff_info['estimate']
        
        # 예측값 계산: Y = b0 + b1*X + b2*Z + b3*X*Z
        y_pred = (intercept + 
                 main_effect * x_values + 
                 moderator_effect * moderator_value + 
                 interaction_effect * x_values * moderator_value)
        
        return y_pred
    
    def create_simple_slopes_plot(self, results: Dict[str, Any],
                                save_path: Optional[Path] = None) -> Path:
        """
        단순기울기 분석 시각화
        
        Args:
            results (Dict[str, Any]): 분석 결과
            save_path (Optional[Path]): 저장 경로
            
        Returns:
            Path: 저장된 파일 경로
        """
        logger.info("단순기울기 플롯 생성 시작")
        
        simple_slopes = results.get('simple_slopes', {})
        if not simple_slopes:
            logger.warning("단순기울기 데이터가 없습니다.")
            return None
        
        # 데이터 준비
        levels = []
        slopes = []
        errors = []
        p_values = []
        significant = []
        
        for level, slope_info in simple_slopes.items():
            levels.append(level.replace('_', ' ').title())
            slopes.append(slope_info.get('simple_slope', 0))
            errors.append(slope_info.get('std_error', 0))
            p_values.append(slope_info.get('p_value', 1))
            significant.append(slope_info.get('significant', False))
        
        # 플롯 생성
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # 막대 색상 (유의한 것은 진한 색, 유의하지 않은 것은 연한 색)
        colors = ['darkblue' if sig else 'lightblue' for sig in significant]
        
        # 막대 그래프
        bars = ax.bar(levels, slopes, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        
        # 0선 추가
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 유의성 표시
        for i, (bar, p_val, sig) in enumerate(zip(bars, p_values, significant)):
            height = bar.get_height()
            symbol = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.01,
                   symbol, ha='center', va='bottom', fontweight='bold')
        
        # 플롯 설정
        variables = results['variables']
        ax.set_xlabel(f'{variables["moderator"]} 수준', fontsize=12)
        ax.set_ylabel('단순기울기 (Simple Slope)', fontsize=12)
        ax.set_title(f'단순기울기 분석: {variables["independent"]} → {variables["dependent"]}', 
                    fontsize=14, fontweight='bold')
        
        # 범례
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkblue', alpha=0.7, label='유의함 (p < 0.05)'),
                          Patch(facecolor='lightblue', alpha=0.7, label='유의하지 않음')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 저장
        if save_path is None:
            save_path = self.results_dir / f"simple_slopes_{variables['independent']}_x_{variables['moderator']}.png"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"단순기울기 플롯 저장: {save_path}")
        return save_path
    
    def create_interaction_heatmap(self, data: pd.DataFrame, results: Dict[str, Any],
                                 save_path: Optional[Path] = None) -> Path:
        """
        상호작용 효과 히트맵 생성
        
        Args:
            data (pd.DataFrame): 분석 데이터
            results (Dict[str, Any]): 분석 결과
            save_path (Optional[Path]): 저장 경로
            
        Returns:
            Path: 저장된 파일 경로
        """
        logger.info("상호작용 히트맵 생성 시작")
        
        variables = results['variables']
        independent_var = variables['independent']
        dependent_var = variables['dependent']
        moderator_var = variables['moderator']
        
        # 그리드 생성
        x_range = np.linspace(data[independent_var].quantile(0.1), 
                             data[independent_var].quantile(0.9), 20)
        z_range = np.linspace(data[moderator_var].quantile(0.1), 
                             data[moderator_var].quantile(0.9), 20)
        
        X, Z = np.meshgrid(x_range, z_range)
        
        # 예측값 계산
        Y_pred = np.zeros_like(X)
        for i in range(len(z_range)):
            Y_pred[i, :] = self._predict_values(x_range, z_range[i], results)
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        im = ax.contourf(X, Z, Y_pred, levels=20, cmap=self.config.color_palette, alpha=0.8)
        contours = ax.contour(X, Z, Y_pred, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{dependent_var} (예측값)', fontsize=12)
        
        # 플롯 설정
        ax.set_xlabel(f'{independent_var} (독립변수)', fontsize=12)
        ax.set_ylabel(f'{moderator_var} (조절변수)', fontsize=12)
        ax.set_title(f'상호작용 효과 히트맵: {independent_var} × {moderator_var}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        if save_path is None:
            save_path = self.results_dir / f"interaction_heatmap_{independent_var}_x_{moderator_var}.png"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"상호작용 히트맵 저장: {save_path}")
        return save_path
    
    def visualize_comprehensive_analysis(self, data: pd.DataFrame, results: Dict[str, Any],
                                       analysis_name: Optional[str] = None) -> Dict[str, Path]:
        """
        포괄적 시각화 (모든 그래프 생성)
        
        Args:
            data (pd.DataFrame): 분석 데이터
            results (Dict[str, Any]): 분석 결과
            analysis_name (Optional[str]): 분석명
            
        Returns:
            Dict[str, Path]: 생성된 그래프 파일 경로들
        """
        logger.info("포괄적 시각화 시작")
        
        if analysis_name is None:
            vars_info = results.get('variables', {})
            analysis_name = f"{vars_info.get('independent', 'X')}_x_{vars_info.get('moderator', 'Z')}"
        
        plot_files = {}
        
        try:
            # 1. 조절효과 플롯
            moderation_plot = self.create_moderation_plot(data, results)
            plot_files['moderation_plot'] = moderation_plot
            
            # 2. 단순기울기 플롯
            simple_slopes_plot = self.create_simple_slopes_plot(results)
            if simple_slopes_plot:
                plot_files['simple_slopes_plot'] = simple_slopes_plot
            
            # 3. 상호작용 히트맵
            heatmap_plot = self.create_interaction_heatmap(data, results)
            plot_files['interaction_heatmap'] = heatmap_plot
            
            logger.info(f"포괄적 시각화 완료: {len(plot_files)}개 그래프 생성")
            return plot_files
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            raise


# 편의 함수들
def create_moderation_plot(data: pd.DataFrame, results: Dict[str, Any],
                         save_path: Optional[Path] = None,
                         config: Optional[ModerationAnalysisConfig] = None) -> Path:
    """조절효과 플롯 생성 편의 함수"""
    visualizer = ModerationVisualizer(config)
    return visualizer.create_moderation_plot(data, results, save_path)


def create_simple_slopes_plot(results: Dict[str, Any], save_path: Optional[Path] = None,
                            config: Optional[ModerationAnalysisConfig] = None) -> Path:
    """단순기울기 플롯 생성 편의 함수"""
    visualizer = ModerationVisualizer(config)
    return visualizer.create_simple_slopes_plot(results, save_path)


def create_interaction_heatmap(data: pd.DataFrame, results: Dict[str, Any],
                             save_path: Optional[Path] = None,
                             config: Optional[ModerationAnalysisConfig] = None) -> Path:
    """상호작용 히트맵 생성 편의 함수"""
    visualizer = ModerationVisualizer(config)
    return visualizer.create_interaction_heatmap(data, results, save_path)


def visualize_moderation_analysis(data: pd.DataFrame, results: Dict[str, Any],
                                analysis_name: Optional[str] = None,
                                config: Optional[ModerationAnalysisConfig] = None) -> Dict[str, Path]:
    """조절효과 분석 포괄적 시각화 편의 함수"""
    visualizer = ModerationVisualizer(config)
    return visualizer.visualize_comprehensive_analysis(data, results, analysis_name)
