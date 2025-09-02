"""
Factor Analysis Visualization Module using semopy

이 모듈은 semopy를 사용한 요인분석 결과를 다양한 방식으로 가시화하는 기능을 제공합니다.
재사용성, 모듈화, 단일책임 원칙을 고려하여 설계되었습니다.
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
    from semopy import Model
    from semopy.stats import calc_stats
    # semopy 가시화 기능
    try:
        from semopy.inspector import inspect
        SEMOPY_INSPECTOR_AVAILABLE = True
    except ImportError:
        SEMOPY_INSPECTOR_AVAILABLE = False
        
    try:
        import graphviz
        GRAPHVIZ_AVAILABLE = True
    except ImportError:
        GRAPHVIZ_AVAILABLE = False
        
except ImportError as e:
    logging.error("semopy 라이브러리를 찾을 수 없습니다. pip install semopy로 설치해주세요.")
    raise e

from .config import FactorAnalysisConfig, get_default_config

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False


class FactorLoadingPlotter:
    """Factor Loading 시각화 전담 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Factor Loading Plotter 초기화
        
        Args:
            figsize (Tuple[int, int]): 그래프 크기
            style (str): seaborn 스타일
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
    
    def plot_loading_heatmap(self, loadings_df: pd.DataFrame, 
                           title: str = "Factor Loadings Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Factor Loading 히트맵 생성
        
        Args:
            loadings_df (pd.DataFrame): Factor loading 데이터
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        Returns:
            plt.Figure: 생성된 그래프
        """
        # 데이터 피벗 (Factor x Item 매트릭스)
        pivot_data = loadings_df.pivot(index='Item', columns='Factor', values='Loading')
        pivot_data = pivot_data.fillna(0)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 히트맵 생성
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Factor Loading'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Factors', fontsize=12)
        ax.set_ylabel('Items', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"히트맵 저장 완료: {save_path}")
        
        return fig
    
    def plot_loading_barplot(self, loadings_df: pd.DataFrame,
                           factor_name: Optional[str] = None,
                           title: str = "Factor Loadings",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Factor Loading 막대 그래프 생성
        
        Args:
            loadings_df (pd.DataFrame): Factor loading 데이터
            factor_name (Optional[str]): 특정 요인만 표시 (None이면 모든 요인)
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        Returns:
            plt.Figure: 생성된 그래프
        """
        # 특정 요인만 필터링
        if factor_name:
            plot_data = loadings_df[loadings_df['Factor'] == factor_name].copy()
            title = f"{title} - {factor_name}"
        else:
            plot_data = loadings_df.copy()
        
        # 유의성에 따른 색상 설정
        if 'Significant' in plot_data.columns:
            plot_data['Color'] = plot_data['Significant'].map({True: 'steelblue', False: 'lightcoral'})
        else:
            plot_data['Color'] = 'steelblue'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 막대 그래프 생성
        bars = ax.bar(range(len(plot_data)), 
                     plot_data['Loading'], 
                     color=plot_data['Color'],
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=0.5)
        
        # 축 설정
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['Item'], rotation=45, ha='right')
        ax.set_ylabel('Factor Loading', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 0선 추가
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 값 라벨 추가
        for i, (bar, loading) in enumerate(zip(bars, plot_data['Loading'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                   f'{loading:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # 범례 추가 (유의성이 있는 경우)
        if 'Significant' in plot_data.columns:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Significant (p < 0.05)'),
                             Patch(facecolor='lightcoral', alpha=0.7, label='Not Significant')]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"막대 그래프 저장 완료: {save_path}")
        
        return fig


class ModelDiagramGenerator:
    """SEM 모델 다이어그램 생성 전담 클래스"""
    
    def __init__(self):
        """Model Diagram Generator 초기화"""
        self.graphviz_available = GRAPHVIZ_AVAILABLE
        if not self.graphviz_available:
            logger.warning("graphviz가 설치되지 않아 모델 다이어그램 생성이 제한됩니다.")
    
    def generate_path_diagram(self, model: Model, 
                            title: str = "SEM Path Diagram",
                            save_path: Optional[str] = None) -> Optional[Any]:
        """
        SEM 경로 다이어그램 생성
        
        Args:
            model (Model): 적합된 semopy 모델
            title (str): 다이어그램 제목
            save_path (Optional[str]): 저장 경로
            
        Returns:
            Optional[Any]: 생성된 다이어그램 (graphviz 객체)
        """
        if not self.graphviz_available:
            logger.error("graphviz가 설치되지 않아 경로 다이어그램을 생성할 수 없습니다.")
            return None
        
        try:
            # semopy의 내장 시각화 기능 사용
            if hasattr(model, 'draw'):
                diagram = model.draw()
                
                if save_path:
                    # 확장자에 따라 저장 형식 결정
                    if save_path.endswith('.png'):
                        diagram.render(save_path.replace('.png', ''), format='png', cleanup=True)
                    elif save_path.endswith('.pdf'):
                        diagram.render(save_path.replace('.pdf', ''), format='pdf', cleanup=True)
                    else:
                        diagram.render(save_path, cleanup=True)
                    
                    logger.info(f"경로 다이어그램 저장 완료: {save_path}")
                
                return diagram
            else:
                logger.warning("semopy 모델에 draw 메서드가 없습니다.")
                return None
                
        except Exception as e:
            logger.error(f"경로 다이어그램 생성 중 오류: {e}")
            return None
    
    def create_custom_diagram(self, loadings_df: pd.DataFrame,
                            title: str = "Factor Model Diagram",
                            save_path: Optional[str] = None) -> Optional[str]:
        """
        커스텀 모델 다이어그램 생성 (텍스트 기반)
        
        Args:
            loadings_df (pd.DataFrame): Factor loading 데이터
            title (str): 다이어그램 제목
            save_path (Optional[str]): 저장 경로
            
        Returns:
            Optional[str]: 생성된 다이어그램 텍스트
        """
        try:
            diagram_text = [f"=== {title} ===\n"]
            
            # 요인별로 그룹화
            for factor in loadings_df['Factor'].unique():
                factor_data = loadings_df[loadings_df['Factor'] == factor]
                diagram_text.append(f"\n[{factor}]")
                
                for _, row in factor_data.iterrows():
                    loading = row['Loading']
                    item = row['Item']
                    
                    # 유의성 표시
                    if 'Significant' in row and row['Significant']:
                        sig_mark = " ***" if 'P_value' in row and row['P_value'] < 0.001 else " *"
                    else:
                        sig_mark = ""
                    
                    # 화살표와 loading 값
                    arrow = "──>" if loading > 0 else "──┤"
                    diagram_text.append(f"  {arrow} {item}: {loading:.3f}{sig_mark}")
            
            diagram_str = "\n".join(diagram_text)
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(diagram_str)
                logger.info(f"텍스트 다이어그램 저장 완료: {save_path}")
            
            return diagram_str
            
        except Exception as e:
            logger.error(f"커스텀 다이어그램 생성 중 오류: {e}")
            return None


class FitIndicesVisualizer:
    """적합도 지수 시각화 전담 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Fit Indices Visualizer 초기화
        
        Args:
            figsize (Tuple[int, int]): 그래프 크기
        """
        self.figsize = figsize
        
        # 적합도 지수 기준값
        self.fit_criteria = {
            'CFI': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
            'TLI': {'excellent': 0.95, 'good': 0.90, 'acceptable': 0.85},
            'RMSEA': {'excellent': 0.05, 'good': 0.08, 'acceptable': 0.10},
            'SRMR': {'excellent': 0.05, 'good': 0.08, 'acceptable': 0.10}
        }
    
    def plot_fit_indices(self, fit_indices: Dict[str, float],
                        title: str = "Model Fit Indices",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        적합도 지수 시각화
        
        Args:
            fit_indices (Dict[str, float]): 적합도 지수 딕셔너리
            title (str): 그래프 제목
            save_path (Optional[str]): 저장 경로
            
        Returns:
            plt.Figure: 생성된 그래프
        """
        # 시각화할 지수만 필터링
        plot_indices = {k: v for k, v in fit_indices.items() 
                       if k in self.fit_criteria and pd.notna(v)}
        
        if not plot_indices:
            logger.warning("시각화할 적합도 지수가 없습니다.")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        indices = list(plot_indices.keys())
        values = list(plot_indices.values())
        
        # 막대 색상 결정 (기준에 따라)
        colors = []
        for idx, value in zip(indices, values):
            criteria = self.fit_criteria.get(idx, {})
            if idx in ['RMSEA', 'SRMR']:  # 낮을수록 좋은 지수
                if value <= criteria.get('excellent', 0):
                    colors.append('green')
                elif value <= criteria.get('good', 0):
                    colors.append('orange')
                elif value <= criteria.get('acceptable', 0):
                    colors.append('red')
                else:
                    colors.append('darkred')
            else:  # 높을수록 좋은 지수
                if value >= criteria.get('excellent', 1):
                    colors.append('green')
                elif value >= criteria.get('good', 1):
                    colors.append('orange')
                elif value >= criteria.get('acceptable', 1):
                    colors.append('red')
                else:
                    colors.append('darkred')
        
        # 막대 그래프 생성
        bars = ax.bar(indices, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 기준선 추가
        for i, (idx, value) in enumerate(zip(indices, values)):
            criteria = self.fit_criteria.get(idx, {})
            if idx in ['RMSEA', 'SRMR']:
                # 낮을수록 좋은 지수의 기준선
                if 'excellent' in criteria:
                    ax.axhline(y=criteria['excellent'], color='green', linestyle='--', alpha=0.5)
                if 'acceptable' in criteria:
                    ax.axhline(y=criteria['acceptable'], color='red', linestyle='--', alpha=0.5)
            else:
                # 높을수록 좋은 지수의 기준선
                if 'excellent' in criteria:
                    ax.axhline(y=criteria['excellent'], color='green', linestyle='--', alpha=0.5)
                if 'acceptable' in criteria:
                    ax.axhline(y=criteria['acceptable'], color='red', linestyle='--', alpha=0.5)
        
        # 값 라벨 추가
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Index Value', fontsize=12)
        ax.set_ylim(0, max(1.1, max(values) * 1.1))
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Excellent'),
            Patch(facecolor='orange', alpha=0.7, label='Good'),
            Patch(facecolor='red', alpha=0.7, label='Acceptable'),
            Patch(facecolor='darkred', alpha=0.7, label='Poor')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"적합도 지수 그래프 저장 완료: {save_path}")
        
        return fig


class FactorAnalysisVisualizer:
    """Factor Analysis 결과 종합 가시화 클래스"""

    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        """
        Factor Analysis Visualizer 초기화

        Args:
            config (Optional[FactorAnalysisConfig]): 설정 객체
        """
        self.config = config if config is not None else get_default_config()

        # 하위 가시화 클래스들 초기화
        self.loading_plotter = FactorLoadingPlotter()
        self.diagram_generator = ModelDiagramGenerator()
        self.fit_visualizer = FitIndicesVisualizer()

        logger.info("Factor Analysis Visualizer 초기화 완료")

    def visualize_analysis_results(self, analysis_results: Dict[str, Any],
                                 output_dir: Optional[Union[str, Path]] = None,
                                 show_plots: bool = True) -> Dict[str, Any]:
        """
        분석 결과 종합 가시화

        Args:
            analysis_results (Dict[str, Any]): factor_analyzer의 분석 결과
            output_dir (Optional[Union[str, Path]]): 출력 디렉토리
            show_plots (bool): 그래프 화면 표시 여부

        Returns:
            Dict[str, Any]: 생성된 시각화 결과 정보
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        visualization_results = {
            'plots_generated': [],
            'diagrams_generated': [],
            'errors': []
        }

        try:
            # 1. Factor Loadings 시각화
            if 'factor_loadings' in analysis_results:
                loadings_df = analysis_results['factor_loadings']
                if isinstance(loadings_df, pd.DataFrame) and len(loadings_df) > 0:
                    self._visualize_factor_loadings(loadings_df, output_path,
                                                   visualization_results, show_plots)

            # 2. 적합도 지수 시각화
            if 'fit_indices' in analysis_results:
                fit_indices = analysis_results['fit_indices']
                if isinstance(fit_indices, dict) and fit_indices:
                    self._visualize_fit_indices(fit_indices, output_path,
                                               visualization_results, show_plots)

            # 3. 모델 다이어그램 생성 (factor_loadings 기반)
            if 'factor_loadings' in analysis_results:
                loadings_df = analysis_results['factor_loadings']
                if isinstance(loadings_df, pd.DataFrame) and len(loadings_df) > 0:
                    self._generate_model_diagrams(None, loadings_df, output_path,
                                                 visualization_results)

            logger.info(f"가시화 완료: {len(visualization_results['plots_generated'])}개 그래프, "
                       f"{len(visualization_results['diagrams_generated'])}개 다이어그램 생성")

        except Exception as e:
            error_msg = f"가시화 중 오류 발생: {e}"
            logger.error(error_msg)
            visualization_results['errors'].append(error_msg)

        return visualization_results

    def _visualize_factor_loadings(self, loadings_df: pd.DataFrame,
                                  output_path: Optional[Path],
                                  results: Dict[str, Any],
                                  show_plots: bool):
        """Factor Loadings 시각화 처리"""
        try:
            # 히트맵 생성
            save_path = str(output_path / "factor_loadings_heatmap.png") if output_path else None
            heatmap_fig = self.loading_plotter.plot_loading_heatmap(
                loadings_df,
                title="Factor Loadings Heatmap",
                save_path=save_path
            )
            results['plots_generated'].append('factor_loadings_heatmap')

            if show_plots:
                plt.show()
            else:
                plt.close(heatmap_fig)

            # 요인별 막대 그래프 생성
            factors = loadings_df['Factor'].unique()
            for factor in factors:
                save_path = str(output_path / f"factor_loadings_{factor}.png") if output_path else None
                bar_fig = self.loading_plotter.plot_loading_barplot(
                    loadings_df,
                    factor_name=factor,
                    title=f"Factor Loadings - {factor}",
                    save_path=save_path
                )
                results['plots_generated'].append(f'factor_loadings_{factor}')

                if show_plots:
                    plt.show()
                else:
                    plt.close(bar_fig)

        except Exception as e:
            error_msg = f"Factor loadings 시각화 오류: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

    def _visualize_fit_indices(self, fit_indices: Dict[str, float],
                              output_path: Optional[Path],
                              results: Dict[str, Any],
                              show_plots: bool):
        """적합도 지수 시각화 처리"""
        try:
            save_path = str(output_path / "model_fit_indices.png") if output_path else None
            fit_fig = self.fit_visualizer.plot_fit_indices(
                fit_indices,
                title="Model Fit Indices",
                save_path=save_path
            )

            if fit_fig:
                results['plots_generated'].append('model_fit_indices')

                if show_plots:
                    plt.show()
                else:
                    plt.close(fit_fig)

        except Exception as e:
            error_msg = f"적합도 지수 시각화 오류: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

    def _generate_model_diagrams(self, model: Any, loadings_df: pd.DataFrame,
                                output_path: Optional[Path],
                                results: Dict[str, Any]):
        """모델 다이어그램 생성 처리"""
        try:
            # semopy 경로 다이어그램 (모델이 있는 경우)
            if model and hasattr(model, 'draw'):
                save_path = str(output_path / "sem_path_diagram.png") if output_path else None
                diagram = self.diagram_generator.generate_path_diagram(
                    model,
                    title="SEM Path Diagram",
                    save_path=save_path
                )
                if diagram:
                    results['diagrams_generated'].append('sem_path_diagram')

            # 커스텀 텍스트 다이어그램
            save_path = str(output_path / "factor_model_diagram.txt") if output_path else None
            text_diagram = self.diagram_generator.create_custom_diagram(
                loadings_df,
                title="Factor Model Structure",
                save_path=save_path
            )
            if text_diagram:
                results['diagrams_generated'].append('factor_model_diagram')

        except Exception as e:
            error_msg = f"모델 다이어그램 생성 오류: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)


# 편의 함수들
def visualize_factor_analysis(analysis_results: Dict[str, Any],
                            output_dir: Optional[Union[str, Path]] = None,
                            show_plots: bool = True,
                            config: Optional[FactorAnalysisConfig] = None) -> Dict[str, Any]:
    """
    Factor Analysis 결과 가시화 편의 함수

    Args:
        analysis_results (Dict[str, Any]): factor_analyzer의 분석 결과
        output_dir (Optional[Union[str, Path]]): 출력 디렉토리
        show_plots (bool): 그래프 화면 표시 여부
        config (Optional[FactorAnalysisConfig]): 설정 객체

    Returns:
        Dict[str, Any]: 가시화 결과 정보
    """
    visualizer = FactorAnalysisVisualizer(config)
    return visualizer.visualize_analysis_results(analysis_results, output_dir, show_plots)


def create_loading_heatmap(loadings_df: pd.DataFrame,
                          title: str = "Factor Loadings Heatmap",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Factor Loading 히트맵 생성 편의 함수

    Args:
        loadings_df (pd.DataFrame): Factor loading 데이터
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로
        figsize (Tuple[int, int]): 그래프 크기

    Returns:
        plt.Figure: 생성된 그래프
    """
    plotter = FactorLoadingPlotter(figsize=figsize)
    return plotter.plot_loading_heatmap(loadings_df, title, save_path)


def create_loading_barplot(loadings_df: pd.DataFrame,
                          factor_name: Optional[str] = None,
                          title: str = "Factor Loadings",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Factor Loading 막대 그래프 생성 편의 함수

    Args:
        loadings_df (pd.DataFrame): Factor loading 데이터
        factor_name (Optional[str]): 특정 요인만 표시
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로
        figsize (Tuple[int, int]): 그래프 크기

    Returns:
        plt.Figure: 생성된 그래프
    """
    plotter = FactorLoadingPlotter(figsize=figsize)
    return plotter.plot_loading_barplot(loadings_df, factor_name, title, save_path)


def create_fit_indices_plot(fit_indices: Dict[str, float],
                           title: str = "Model Fit Indices",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    적합도 지수 그래프 생성 편의 함수

    Args:
        fit_indices (Dict[str, float]): 적합도 지수 딕셔너리
        title (str): 그래프 제목
        save_path (Optional[str]): 저장 경로
        figsize (Tuple[int, int]): 그래프 크기

    Returns:
        plt.Figure: 생성된 그래프
    """
    visualizer = FitIndicesVisualizer(figsize=figsize)
    return visualizer.plot_fit_indices(fit_indices, title, save_path)


def create_model_diagram(loadings_df: pd.DataFrame,
                        title: str = "Factor Model Diagram",
                        save_path: Optional[str] = None) -> Optional[str]:
    """
    모델 다이어그램 생성 편의 함수

    Args:
        loadings_df (pd.DataFrame): Factor loading 데이터
        title (str): 다이어그램 제목
        save_path (Optional[str]): 저장 경로

    Returns:
        Optional[str]: 생성된 다이어그램 텍스트
    """
    generator = ModelDiagramGenerator()
    return generator.create_custom_diagram(loadings_df, title, save_path)
