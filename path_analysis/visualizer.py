"""
Path Analysis Visualization Module

경로분석 결과를 시각화하는 모듈입니다.
경로 다이어그램, 효과 차트, 적합도 지수 시각화 등을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings

# semopy 가시화 관련 임포트
try:
    import semopy
    from semopy import Model, semplot
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

# 그래프 관련 임포트
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False


class PathAnalysisVisualizer:
    """경로분석 시각화 클래스"""
    
    def __init__(self, output_dir: str = "path_analysis_results/visualizations"):
        """
        초기화
        
        Args:
            output_dir (str): 시각화 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"PathAnalysisVisualizer 초기화 완료: {self.output_dir}")
    
    def create_comprehensive_visualization(self, 
                                         analysis_results: Dict[str, Any],
                                         filename_prefix: str = "path_analysis") -> Dict[str, str]:
        """
        종합적인 시각화 생성
        
        Args:
            analysis_results (Dict[str, Any]): 분석 결과
            filename_prefix (str): 파일명 접두사
            
        Returns:
            Dict[str, str]: 생성된 시각화 파일들의 경로
        """
        logger.info("종합적인 경로분석 시각화 시작")
        
        visualization_files = {}
        
        try:
            # 1. 경로 다이어그램 (semopy 사용)
            if 'model_object' in analysis_results and SEMOPY_AVAILABLE:
                path_diagram = self.create_path_diagram(
                    analysis_results['model_object'],
                    f"{filename_prefix}_path_diagram"
                )
                if path_diagram:
                    visualization_files['path_diagram'] = str(path_diagram)
            
            # 2. 적합도 지수 시각화
            if 'fit_indices' in analysis_results:
                fit_plot = self.plot_fit_indices(
                    analysis_results['fit_indices'],
                    f"{filename_prefix}_fit_indices"
                )
                visualization_files['fit_indices_plot'] = str(fit_plot)
            
            # 3. 경로계수 시각화
            if 'path_coefficients' in analysis_results:
                coeff_plot = self.plot_path_coefficients(
                    analysis_results['path_coefficients'],
                    f"{filename_prefix}_path_coefficients"
                )
                visualization_files['path_coefficients_plot'] = str(coeff_plot)
            
            # 4. 효과 분석 시각화
            if 'effects_analysis' in analysis_results:
                effects_plot = self.plot_effects_analysis(
                    analysis_results['effects_analysis'],
                    f"{filename_prefix}_effects"
                )
                visualization_files['effects_plot'] = str(effects_plot)
            
            # 5. 매개효과 시각화
            if 'effects_analysis' in analysis_results and 'mediation_analysis' in analysis_results['effects_analysis']:
                mediation_plot = self.plot_mediation_analysis(
                    analysis_results['effects_analysis']['mediation_analysis'],
                    f"{filename_prefix}_mediation"
                )
                visualization_files['mediation_plot'] = str(mediation_plot)
            
            logger.info(f"시각화 완료: {len(visualization_files)}개 파일 생성")
            return visualization_files
            
        except Exception as e:
            logger.error(f"시각화 중 오류: {e}")
            raise
    
    def create_path_diagram(self, 
                          model: Model,
                          filename: str,
                          format: str = 'png') -> Optional[Path]:
        """
        경로 다이어그램 생성 (semopy 사용)
        
        Args:
            model (Model): 적합된 semopy 모델
            filename (str): 파일명
            format (str): 파일 형식
            
        Returns:
            Optional[Path]: 생성된 파일 경로
        """
        if not SEMOPY_AVAILABLE:
            logger.warning("semopy를 사용할 수 없어 경로 다이어그램을 생성할 수 없습니다.")
            return None
        
        try:
            file_path = self.output_dir / f"{filename}.{format}"
            
            # semplot을 사용한 다이어그램 생성
            graph = semplot(
                mod=model,
                filename=str(file_path),
                std_ests=True,
                plot_covs=False,
                title="Path Analysis Diagram"
            )
            
            logger.info(f"경로 다이어그램 생성 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"경로 다이어그램 생성 오류: {e}")
            return None
    
    def plot_fit_indices(self, 
                        fit_indices: Dict[str, float],
                        filename: str) -> Path:
        """
        적합도 지수 시각화
        
        Args:
            fit_indices (Dict[str, float]): 적합도 지수들
            filename (str): 파일명
            
        Returns:
            Path: 생성된 파일 경로
        """
        try:
            # 유효한 적합도 지수만 선택
            valid_indices = {k: v for k, v in fit_indices.items() 
                           if not pd.isna(v) and k not in ['chi_square', 'df', 'aic', 'bic']}
            
            if not valid_indices:
                logger.warning("시각화할 적합도 지수가 없습니다.")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 적합도 지수별 기준선 설정
            cutoff_lines = {
                'cfi': [0.90, 0.95],
                'tli': [0.90, 0.95],
                'rmsea': [0.08, 0.06],
                'srmr': [0.10, 0.08],
                'p_value': [0.05]
            }
            
            indices = list(valid_indices.keys())
            values = list(valid_indices.values())
            
            # 막대 그래프
            bars = ax.bar(indices, values, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # 기준선 추가
            for i, index_name in enumerate(indices):
                if index_name in cutoff_lines:
                    for cutoff in cutoff_lines[index_name]:
                        ax.axhline(y=cutoff, color='red', linestyle='--', alpha=0.7)
                        ax.text(i, cutoff, f'{cutoff}', ha='center', va='bottom', 
                               color='red', fontweight='bold')
            
            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title('Model Fit Indices', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.set_xlabel('Fit Index', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"적합도 지수 시각화 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"적합도 지수 시각화 오류: {e}")
            raise
    
    def plot_path_coefficients(self, 
                             path_coefficients: Dict[str, Any],
                             filename: str) -> Path:
        """
        경로계수 시각화
        
        Args:
            path_coefficients (Dict[str, Any]): 경로계수 정보
            filename (str): 파일명
            
        Returns:
            Path: 생성된 파일 경로
        """
        try:
            if 'paths' not in path_coefficients or 'coefficients' not in path_coefficients:
                logger.warning("경로계수 데이터가 불완전합니다.")
                return None
            
            paths = path_coefficients['paths']
            coefficients = path_coefficients['coefficients']
            p_values = path_coefficients.get('p_values', {})
            
            # 데이터 준비
            path_labels = [f"{from_var} → {to_var}" for from_var, to_var in paths]
            coeff_values = [coefficients.get(i, 0) for i in range(len(paths))]
            p_vals = [p_values.get(i, 1) for i in range(len(paths))]
            
            # 유의성에 따른 색상 설정
            colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 
                     else 'lightblue' for p in p_vals]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 수평 막대 그래프
            bars = ax.barh(path_labels, coeff_values, color=colors, alpha=0.7, edgecolor='black')
            
            # 0 기준선
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 계수 값 표시
            for bar, coeff, p_val in zip(bars, coeff_values, p_vals):
                width = bar.get_width()
                sig_stars = self._get_significance_stars(p_val)
                ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                       f'{coeff:.3f}{sig_stars}', ha='left' if width >= 0 else 'right', 
                       va='center', fontweight='bold')
            
            ax.set_title('Path Coefficients', fontsize=14, fontweight='bold')
            ax.set_xlabel('Coefficient Value', fontsize=12)
            ax.set_ylabel('Path', fontsize=12)
            
            # 범례 추가
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='p < 0.001'),
                plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='p < 0.01'),
                plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.7, label='p < 0.05'),
                plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='p ≥ 0.05')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"경로계수 시각화 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"경로계수 시각화 오류: {e}")
            raise
    
    def plot_effects_analysis(self, 
                            effects_analysis: Dict[str, Any],
                            filename: str) -> Path:
        """
        효과 분석 시각화
        
        Args:
            effects_analysis (Dict[str, Any]): 효과 분석 결과
            filename (str): 파일명
            
        Returns:
            Path: 생성된 파일 경로
        """
        try:
            effects_data = []
            
            # 직접효과
            if 'direct_effects' in effects_analysis:
                direct = effects_analysis['direct_effects']
                effects_data.append({
                    'Effect_Type': 'Direct',
                    'Value': direct.get('coefficient', 0),
                    'P_Value': direct.get('p_value', 1)
                })
            
            # 간접효과
            if 'indirect_effects' in effects_analysis:
                indirect = effects_analysis['indirect_effects']
                effects_data.append({
                    'Effect_Type': 'Indirect',
                    'Value': indirect.get('total_indirect_effect', 0),
                    'P_Value': 1.0  # 간접효과의 p값은 별도 계산 필요
                })
            
            # 총효과
            if 'total_effects' in effects_analysis:
                total = effects_analysis['total_effects']
                effects_data.append({
                    'Effect_Type': 'Total',
                    'Value': total.get('total_effect', 0),
                    'P_Value': 1.0
                })
            
            if not effects_data:
                logger.warning("시각화할 효과 데이터가 없습니다.")
                return None
            
            df = pd.DataFrame(effects_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 막대 그래프
            colors = ['steelblue', 'orange', 'green'][:len(df)]
            bars = ax.bar(df['Effect_Type'], df['Value'], color=colors, alpha=0.7, edgecolor='black')
            
            # 0 기준선
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 값 표시
            for bar, value, p_val in zip(bars, df['Value'], df['P_Value']):
                height = bar.get_height()
                sig_stars = self._get_significance_stars(p_val)
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                       f'{value:.3f}{sig_stars}', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            ax.set_title('Effects Analysis', fontsize=14, fontweight='bold')
            ax.set_ylabel('Effect Size', fontsize=12)
            ax.set_xlabel('Effect Type', fontsize=12)
            
            plt.tight_layout()
            
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"효과 분석 시각화 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"효과 분석 시각화 오류: {e}")
            raise
    
    def plot_mediation_analysis(self, 
                              mediation_analysis: Dict[str, Any],
                              filename: str) -> Path:
        """
        매개효과 분석 시각화
        
        Args:
            mediation_analysis (Dict[str, Any]): 매개효과 분석 결과
            filename (str): 파일명
            
        Returns:
            Path: 생성된 파일 경로
        """
        try:
            if 'sobel_tests' not in mediation_analysis:
                logger.warning("Sobel test 결과가 없습니다.")
                return None
            
            sobel_tests = mediation_analysis['sobel_tests']
            
            # 데이터 준비
            mediators = list(sobel_tests.keys())
            z_scores = [sobel_tests[med].get('z_score', 0) for med in mediators]
            p_values = [sobel_tests[med].get('p_value', 1) for med in mediators]
            indirect_effects = [sobel_tests[med].get('indirect_effect', 0) for med in mediators]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 1. 간접효과 크기
            colors1 = ['red' if p < 0.05 else 'lightblue' for p in p_values]
            bars1 = ax1.bar(mediators, indirect_effects, color=colors1, alpha=0.7, edgecolor='black')
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_title('Indirect Effects by Mediator', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Indirect Effect Size', fontsize=10)
            ax1.set_xlabel('Mediator', fontsize=10)
            
            # 값 표시
            for bar, effect, p_val in zip(bars1, indirect_effects, p_values):
                height = bar.get_height()
                sig_stars = self._get_significance_stars(p_val)
                ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                        f'{effect:.3f}{sig_stars}', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            # 2. Sobel test Z-scores
            colors2 = ['red' if abs(z) > 1.96 else 'lightblue' for z in z_scores]
            bars2 = ax2.bar(mediators, z_scores, color=colors2, alpha=0.7, edgecolor='black')
            
            ax2.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='p < 0.05')
            ax2.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax2.set_title('Sobel Test Z-scores', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z-score', fontsize=10)
            ax2.set_xlabel('Mediator', fontsize=10)
            ax2.legend()
            
            # 값 표시
            for bar, z_score in zip(bars2, z_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                        f'{z_score:.2f}', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"매개효과 분석 시각화 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"매개효과 분석 시각화 오류: {e}")
            raise
    
    def _get_significance_stars(self, p_value: float) -> str:
        """유의도 별표 반환"""
        if pd.isna(p_value):
            return ""
        elif p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "."
        else:
            return ""


# 편의 함수들
def create_path_diagram(model: Model,
                       filename: str = "path_diagram",
                       output_dir: str = "path_analysis_results/visualizations") -> Optional[Path]:
    """경로 다이어그램 생성 편의 함수"""
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_path_diagram(model, filename)


def visualize_effects(effects_analysis: Dict[str, Any],
                     filename: str = "effects_analysis",
                     output_dir: str = "path_analysis_results/visualizations") -> Path:
    """효과 분석 시각화 편의 함수"""
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.plot_effects_analysis(effects_analysis, filename)
