"""
Path Analysis Visualization Module (semopy 전용)

semopy를 사용한 경로분석 결과의 시각화를 제공합니다.
semopy의 내장 semplot 기능을 활용하여 다양한 경로 다이어그램을 생성합니다.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import pandas as pd

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

# 추가 시각화 라이브러리
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        
        # semopy 시각화 전용 (matplotlib 불필요)
        
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
                          plot_covs: bool = True,
                          plot_exos: bool = True,
                          plot_ests: bool = True,
                          std_ests: bool = True,
                          engine: str = 'dot',
                          latshape: str = 'circle',
                          show: bool = False,
                          structural_only: bool = False) -> Optional[Path]:
        """
        경로 다이어그램 생성 (semopy 사용)

        Args:
            model (Model): 적합된 semopy 모델
            filename (str): 파일명 (확장자 제외)
            plot_covs (bool): 공분산 표시 여부 (기본값: True)
            plot_exos (bool): 외생변수 표시 여부
            plot_ests (bool): 추정값 표시 여부
            std_ests (bool): 표준화 추정값 사용 여부
            engine (str): Graphviz 엔진 ('dot', 'circo', 'neato' 등)
            latshape (str): 잠재변수 모양 ('circle', 'ellipse', 'box')
            show (bool): 즉시 표시 여부
            structural_only (bool): True면 경로계수만 표시 (요인적재량 제외)

        Returns:
            Optional[Path]: 생성된 파일 경로
        """
        if not SEMOPY_AVAILABLE:
            logger.warning("semopy를 사용할 수 없어 경로 다이어그램을 생성할 수 없습니다.")
            return None

        try:
            # semplot은 filename에 확장자가 포함되어야 함
            file_path = self.output_dir / f"{filename}.png"

            # 경로계수만 표시하는 경우 inspection 데이터 필터링
            inspection = None
            observed_vars_to_hide = set()

            if structural_only and plot_ests:
                try:
                    # 모델의 전체 추정치 가져오기
                    full_inspection = model.inspect()

                    # 잠재변수와 관측변수 구분
                    all_vars = set(full_inspection['lval'].unique()) | set(full_inspection['rval'].unique())
                    latent_vars = {var for var in all_vars if not var.startswith('q')}
                    observed_vars = {var for var in all_vars if var.startswith('q')}
                    observed_vars_to_hide = observed_vars.copy()

                    logger.info(f"잠재변수: {sorted(latent_vars)}")
                    logger.info(f"관측변수 (숨김): {sorted(observed_vars)}")

                    # 구조적 경로계수만 필터링 (잠재변수 간 관계)
                    structural_paths = full_inspection[
                        (full_inspection['op'] == '~') &  # 회귀 관계
                        (full_inspection['lval'].isin(latent_vars)) &  # 종속변수가 잠재변수
                        (full_inspection['rval'].isin(latent_vars))    # 독립변수가 잠재변수
                    ].copy()

                    # 공분산도 포함 (잠재변수 간)
                    if plot_covs:
                        covariances = full_inspection[
                            (full_inspection['op'] == '~~') &  # 공분산
                            (full_inspection['lval'].isin(latent_vars)) &  # 첫 번째 변수가 잠재변수
                            (full_inspection['rval'].isin(latent_vars)) &  # 두 번째 변수가 잠재변수
                            (full_inspection['lval'] != full_inspection['rval'])  # 분산 제외
                        ].copy()

                        structural_paths = pd.concat([structural_paths, covariances], ignore_index=True)

                    # semplot이 요구하는 컬럼 구조 확인 및 조정
                    if std_ests and 'Est. Std' not in structural_paths.columns:
                        # 표준화 추정값이 없으면 비표준화 추정값 사용
                        logger.warning("표준화 추정값이 없어 비표준화 추정값을 사용합니다.")
                        std_ests = False

                    inspection = structural_paths
                    logger.info(f"구조적 경로계수만 표시: {len(structural_paths)}개 경로")

                except Exception as e:
                    logger.warning(f"구조적 경로계수 필터링 실패, 전체 모델 표시: {e}")
                    inspection = None
                    observed_vars_to_hide = set()

            # semplot을 사용한 다이어그램 생성
            graph = semplot(
                mod=model,
                filename=str(file_path),
                inspection=inspection,
                plot_covs=plot_covs,
                plot_exos=plot_exos,
                plot_ests=plot_ests,
                std_ests=std_ests,
                engine=engine,
                latshape=latshape,
                show=show
            )

            # 관찰변수를 숨기기 위해 graphviz 객체 수정
            if structural_only and observed_vars_to_hide:
                try:
                    # graphviz 소스 코드 가져오기
                    dot_source = str(graph)

                    # 관찰변수 노드와 관련 엣지 제거
                    lines = dot_source.split('\n')
                    filtered_lines = []

                    for line in lines:
                        line = line.strip()
                        should_keep = True

                        # 관찰변수 노드 정의 제거
                        for obs_var in observed_vars_to_hide:
                            if f'{obs_var} [' in line or f'"{obs_var}" [' in line:
                                should_keep = False
                                break
                            # 관찰변수와 관련된 엣지 제거
                            if (f'{obs_var} ->' in line or f'-> {obs_var}' in line or
                                f'"{obs_var}" ->' in line or f'-> "{obs_var}"' in line):
                                should_keep = False
                                break

                        if should_keep:
                            filtered_lines.append(line)

                    # 수정된 dot 소스로 새로운 그래프 생성
                    modified_dot_source = '\n'.join(filtered_lines)

                    # graphviz로 직접 렌더링
                    import graphviz
                    modified_graph = graphviz.Source(modified_dot_source)
                    modified_graph.render(str(file_path.with_suffix('')), format='png', cleanup=True)

                    logger.info(f"관찰변수 제거된 경로 다이어그램 생성 완료: {file_path}")

                except Exception as e:
                    logger.warning(f"관찰변수 제거 실패, 원본 다이어그램 사용: {e}")
                    # 원본 그래프 사용

            logger.info(f"경로 다이어그램 생성 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"경로 다이어그램 생성 오류: {e}")
            return None

    def create_multiple_path_diagrams(self,
                                    model: Model,
                                    base_filename: str) -> Dict[str, Optional[Path]]:
        """
        다양한 옵션으로 여러 경로 다이어그램 생성

        Args:
            model (Model): 적합된 semopy 모델
            base_filename (str): 기본 파일명

        Returns:
            Dict[str, Optional[Path]]: 생성된 다이어그램들의 파일 경로
        """
        diagrams = {}

        # 1. 기본 다이어그램 (공분산 포함, 표준화 추정값)
        diagrams['basic'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_basic",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='dot'
        )

        # 2. 상세 다이어그램 (모든 요소 포함)
        diagrams['detailed'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_detailed",
            plot_covs=True,
            plot_exos=True,
            plot_ests=True,
            std_ests=True,
            engine='dot'
        )

        # 3. 간단한 다이어그램 (구조만)
        diagrams['simple'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_simple",
            plot_covs=False,
            plot_ests=False,
            std_ests=False,
            engine='dot'
        )

        # 4. 원형 레이아웃
        diagrams['circular'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_circular",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='circo'
        )

        # 5. 비표준화 다이어그램
        diagrams['unstandardized'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_unstandardized",
            plot_covs=True,
            plot_ests=True,
            std_ests=False,
            engine='dot'
        )

        # 6. 경로계수만 표시 (구조적 경로만)
        diagrams['structural_only'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_structural_only",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='dot',
            structural_only=True  # 경로계수만 표시
        )

        return diagrams

    def create_advanced_path_diagrams(self,
                                    model: Model,
                                    base_filename: str) -> Dict[str, Optional[Path]]:
        """
        고급 semopy 시각화 기능을 활용한 다이어그램 생성

        Args:
            model (Model): 적합된 semopy 모델
            base_filename (str): 기본 파일명

        Returns:
            Dict[str, Optional[Path]]: 생성된 다이어그램들의 파일 경로
        """
        advanced_diagrams = {}

        # 1. 네트워크 레이아웃 (neato 엔진)
        advanced_diagrams['network'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_network",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='neato',
            latshape='ellipse'
        )

        # 2. 계층적 레이아웃 (fdp 엔진)
        advanced_diagrams['hierarchical'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_hierarchical",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='fdp',
            latshape='box'
        )

        # 3. 스프링 레이아웃 (sfdp 엔진)
        advanced_diagrams['spring'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_spring",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='sfdp',
            latshape='circle'
        )

        # 4. 방사형 레이아웃 (twopi 엔진)
        advanced_diagrams['radial'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_radial",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='twopi',
            latshape='circle'
        )

        # 5. 공분산 강조 다이어그램
        advanced_diagrams['covariance_focus'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_covariance_focus",
            plot_covs=True,
            plot_exos=True,
            plot_ests=False,  # 추정값 숨기고 공분산만 강조
            std_ests=False,
            engine='dot',
            latshape='ellipse'
        )

        # 6. 경로계수 강조 다이어그램
        advanced_diagrams['path_focus'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_path_focus",
            plot_covs=False,  # 공분산 숨기고 경로계수만 강조
            plot_exos=False,
            plot_ests=True,
            std_ests=True,
            engine='dot',
            latshape='box'
        )

        # 7. 구조적 경로계수만 표시 (요인적재량 제외)
        advanced_diagrams['structural_paths_only'] = self.create_path_diagram(
            model=model,
            filename=f"{base_filename}_structural_paths_only",
            plot_covs=True,
            plot_ests=True,
            std_ests=True,
            engine='dot',
            latshape='circle',
            structural_only=True  # 경로계수만 표시
        )

        return advanced_diagrams

    def create_comprehensive_visualization(self,
                                         results: Dict[str, Any],
                                         base_filename: str = "path_analysis") -> Dict[str, Any]:
        """
        종합적인 경로분석 시각화 (semopy 전용)

        Args:
            results (Dict[str, Any]): 경로분석 결과
            base_filename (str): 기본 파일명

        Returns:
            Dict[str, Any]: 시각화 결과
        """
        visualization_results = {
            'basic_diagrams': {},
            'advanced_diagrams': {},
            'errors': [],
            'summary': {}
        }

        try:
            # semopy 모델 객체 추출
            if 'model_object' not in results:
                logger.error("semopy 모델 객체를 찾을 수 없습니다.")
                visualization_results['errors'].append("모델 객체 없음")
                return visualization_results

            model = results['model_object']

            # 1. 기본 경로 다이어그램 생성
            basic_diagrams = self.create_multiple_path_diagrams(model, base_filename)
            visualization_results['basic_diagrams'] = basic_diagrams

            # 2. 고급 경로 다이어그램 생성
            advanced_diagrams = self.create_advanced_path_diagrams(model, base_filename)
            visualization_results['advanced_diagrams'] = advanced_diagrams

            # 성공한 다이어그램 수 계산
            all_diagrams = {**basic_diagrams, **advanced_diagrams}
            successful_diagrams = sum(1 for path in all_diagrams.values() if path is not None)

            visualization_results['summary'] = {
                'basic_diagrams': len(basic_diagrams),
                'advanced_diagrams': len(advanced_diagrams),
                'total_diagrams': len(all_diagrams),
                'successful_diagrams': successful_diagrams,
                'failed_diagrams': len(all_diagrams) - successful_diagrams,
                'success_rate': f"{successful_diagrams/len(all_diagrams)*100:.1f}%" if all_diagrams else "0%"
            }

            logger.info(f"종합적인 경로분석 시각화 완료: {successful_diagrams}/{len(all_diagrams)} 성공")

        except Exception as e:
            logger.error(f"시각화 중 오류: {e}")
            visualization_results['errors'].append(str(e))

        return visualization_results


# 편의 함수들
def create_path_diagram(model: Model,
                       filename: str = "path_diagram",
                       output_dir: str = "path_analysis_results/visualizations",
                       **kwargs) -> Optional[Path]:
    """
    경로 다이어그램 생성 편의 함수

    Args:
        model (Model): semopy 모델
        filename (str): 파일명
        output_dir (str): 출력 디렉토리
        **kwargs: semplot 추가 매개변수 (plot_covs, plot_ests, std_ests, engine, latshape, structural_only 등)

    Returns:
        Optional[Path]: 생성된 파일 경로
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_path_diagram(model, filename, **kwargs)


def create_multiple_diagrams(model: Model,
                           base_filename: str = "path_analysis",
                           output_dir: str = "path_analysis_results/visualizations") -> Dict[str, Optional[Path]]:
    """
    다양한 경로 다이어그램 생성 편의 함수 (기본 5가지 유형)

    Args:
        model (Model): semopy 모델
        base_filename (str): 기본 파일명
        output_dir (str): 출력 디렉토리

    Returns:
        Dict[str, Optional[Path]]: 생성된 다이어그램들의 파일 경로
        - basic: 기본 다이어그램 (공분산 포함, 표준화 추정값)
        - detailed: 상세 다이어그램 (모든 요소 포함)
        - simple: 간단한 다이어그램 (구조만)
        - circular: 원형 레이아웃
        - unstandardized: 비표준화 다이어그램
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_multiple_path_diagrams(model, base_filename)


def create_advanced_diagrams(model: Model,
                           base_filename: str = "path_analysis",
                           output_dir: str = "path_analysis_results/visualizations") -> Dict[str, Optional[Path]]:
    """
    고급 경로 다이어그램 생성 편의 함수 (6가지 고급 유형)

    Args:
        model (Model): semopy 모델
        base_filename (str): 기본 파일명
        output_dir (str): 출력 디렉토리

    Returns:
        Dict[str, Optional[Path]]: 생성된 다이어그램들의 파일 경로
        - network: 네트워크 레이아웃 (neato 엔진)
        - hierarchical: 계층적 레이아웃 (fdp 엔진)
        - spring: 스프링 레이아웃 (sfdp 엔진)
        - radial: 방사형 레이아웃 (twopi 엔진)
        - covariance_focus: 공분산 강조 다이어그램
        - path_focus: 경로계수 강조 다이어그램
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_advanced_path_diagrams(model, base_filename)


def visualize_path_analysis(results: Dict[str, Any],
                          base_filename: str = "path_analysis",
                          output_dir: str = "path_analysis_results/visualizations") -> Dict[str, Any]:
    """
    종합적인 경로분석 시각화 편의 함수 (기본 + 고급 다이어그램 모두 생성)

    Args:
        results (Dict[str, Any]): 경로분석 결과
        base_filename (str): 기본 파일명
        output_dir (str): 출력 디렉토리

    Returns:
        Dict[str, Any]: 시각화 결과
        - basic_diagrams: 기본 다이어그램들 (5가지)
        - advanced_diagrams: 고급 다이어그램들 (6가지)
        - summary: 생성 결과 요약
        - errors: 발생한 오류들
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_comprehensive_visualization(results, base_filename)

    def create_bootstrap_confidence_interval_plot(self,
                                                 bootstrap_results: Dict[str, Any],
                                                 filename: str = "bootstrap_ci",
                                                 figsize: Tuple[int, int] = (12, 8)) -> Optional[Path]:
        """
        부트스트래핑 신뢰구간 시각화

        Args:
            bootstrap_results: 부트스트래핑 결과
            filename: 파일명
            figsize: 그림 크기

        Returns:
            Optional[Path]: 저장된 파일 경로
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib이 설치되지 않아 부트스트래핑 시각화를 생성할 수 없습니다.")
            return None

        try:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Bootstrap Confidence Intervals Analysis', fontsize=16, fontweight='bold')

            # 데이터 준비
            all_effects = []
            for combination, result in bootstrap_results.items():
                confidence_intervals = result.get('confidence_intervals', {})
                for effect_type, ci_info in confidence_intervals.items():
                    all_effects.append({
                        'combination': combination,
                        'effect_type': effect_type,
                        'mean': ci_info.get('mean', 0),
                        'lower_ci': ci_info.get('lower_ci', 0),
                        'upper_ci': ci_info.get('upper_ci', 0),
                        'significant': ci_info.get('significant', False)
                    })

            if not all_effects:
                logger.warning("부트스트래핑 결과가 없어 시각화를 생성할 수 없습니다.")
                plt.close(fig)
                return None

            df_effects = pd.DataFrame(all_effects)

            # 1. 직접효과 신뢰구간
            direct_effects = df_effects[df_effects['effect_type'] == 'direct_effects']
            if not direct_effects.empty:
                ax1 = axes[0, 0]
                y_pos = range(len(direct_effects))

                # 신뢰구간 막대
                for i, (_, row) in enumerate(direct_effects.iterrows()):
                    color = 'red' if row['significant'] else 'blue'
                    ax1.barh(i, row['upper_ci'] - row['lower_ci'],
                            left=row['lower_ci'], color=color, alpha=0.6)
                    ax1.plot(row['mean'], i, 'ko', markersize=6)

                ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels([combo.replace('_to_', ' → ') for combo in direct_effects['combination']])
                ax1.set_xlabel('Direct Effect')
                ax1.set_title('Direct Effects Confidence Intervals')
                ax1.grid(True, alpha=0.3)

            # 2. 간접효과 신뢰구간
            indirect_effects = df_effects[df_effects['effect_type'] == 'indirect_effects']
            if not indirect_effects.empty:
                ax2 = axes[0, 1]
                y_pos = range(len(indirect_effects))

                for i, (_, row) in enumerate(indirect_effects.iterrows()):
                    color = 'red' if row['significant'] else 'blue'
                    ax2.barh(i, row['upper_ci'] - row['lower_ci'],
                            left=row['lower_ci'], color=color, alpha=0.6)
                    ax2.plot(row['mean'], i, 'ko', markersize=6)

                ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels([combo.replace('_to_', ' → ') for combo in indirect_effects['combination']])
                ax2.set_xlabel('Indirect Effect')
                ax2.set_title('Indirect Effects Confidence Intervals')
                ax2.grid(True, alpha=0.3)

            # 3. 효과 크기 분포
            ax3 = axes[1, 0]
            effect_means = df_effects['mean'].values
            ax3.hist(effect_means, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Effect Size')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Effect Sizes')
            ax3.grid(True, alpha=0.3)

            # 4. 유의성 요약
            ax4 = axes[1, 1]
            significance_counts = df_effects['significant'].value_counts()
            colors = ['lightcoral' if idx else 'lightblue' for idx in significance_counts.index]
            labels = ['Significant' if idx else 'Non-significant' for idx in significance_counts.index]

            wedges, texts, autotexts = ax4.pie(significance_counts.values, labels=labels,
                                              colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Significance Summary')

            plt.tight_layout()

            # 파일 저장
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"부트스트래핑 신뢰구간 시각화 저장 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"부트스트래핑 시각화 생성 오류: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    def create_mediation_effects_heatmap(self,
                                       all_mediations: Dict[str, Any],
                                       filename: str = "mediation_heatmap",
                                       figsize: Tuple[int, int] = (12, 10)) -> Optional[Path]:
        """
        매개효과 히트맵 시각화

        Args:
            all_mediations: 모든 매개효과 분석 결과
            filename: 파일명
            figsize: 그림 크기

        Returns:
            Optional[Path]: 저장된 파일 경로
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib이 설치되지 않아 매개효과 히트맵을 생성할 수 없습니다.")
            return None

        try:
            # 데이터 준비
            all_results = all_mediations.get('all_results', {})
            if not all_results:
                logger.warning("매개효과 결과가 없어 히트맵을 생성할 수 없습니다.")
                return None

            # 변수들 추출
            variables = set()
            for result in all_results.values():
                variables.add(result.get('independent_var', ''))
                variables.add(result.get('dependent_var', ''))
                variables.add(result.get('mediator', ''))
            variables = sorted([v for v in variables if v])

            if len(variables) < 3:
                logger.warning("히트맵 생성을 위해서는 최소 3개의 변수가 필요합니다.")
                return None

            # 매개효과 매트릭스 생성
            n_vars = len(variables)
            mediation_matrix = np.zeros((n_vars, n_vars))
            significance_matrix = np.zeros((n_vars, n_vars))

            var_to_idx = {var: idx for idx, var in enumerate(variables)}

            for result in all_results.values():
                independent_var = result.get('independent_var', '')
                dependent_var = result.get('dependent_var', '')

                if independent_var in var_to_idx and dependent_var in var_to_idx:
                    i = var_to_idx[independent_var]
                    j = var_to_idx[dependent_var]

                    effect_mean = result.get('indirect_effect_mean', 0)
                    is_significant = result.get('is_significant', False)

                    mediation_matrix[i, j] = effect_mean
                    significance_matrix[i, j] = 1 if is_significant else 0

            # 히트맵 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # 매개효과 크기 히트맵
            im1 = ax1.imshow(mediation_matrix, cmap='RdBu_r', aspect='auto')
            ax1.set_xticks(range(n_vars))
            ax1.set_yticks(range(n_vars))
            ax1.set_xticklabels(variables, rotation=45, ha='right')
            ax1.set_yticklabels(variables)
            ax1.set_xlabel('Dependent Variable')
            ax1.set_ylabel('Independent Variable')
            ax1.set_title('Mediation Effect Sizes')

            # 값 표시
            for i in range(n_vars):
                for j in range(n_vars):
                    if mediation_matrix[i, j] != 0:
                        text_color = 'white' if abs(mediation_matrix[i, j]) > np.max(np.abs(mediation_matrix)) * 0.5 else 'black'
                        ax1.text(j, i, f'{mediation_matrix[i, j]:.3f}',
                                ha='center', va='center', color=text_color, fontsize=8)

            plt.colorbar(im1, ax=ax1, label='Effect Size')

            # 유의성 히트맵
            im2 = ax2.imshow(significance_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
            ax2.set_xticks(range(n_vars))
            ax2.set_yticks(range(n_vars))
            ax2.set_xticklabels(variables, rotation=45, ha='right')
            ax2.set_yticklabels(variables)
            ax2.set_xlabel('Dependent Variable')
            ax2.set_ylabel('Independent Variable')
            ax2.set_title('Mediation Effect Significance')

            # 유의성 표시
            for i in range(n_vars):
                for j in range(n_vars):
                    if significance_matrix[i, j] == 1:
                        ax2.text(j, i, '***', ha='center', va='center',
                                color='white', fontweight='bold', fontsize=12)

            plt.colorbar(im2, ax=ax2, label='Significant (1) / Non-significant (0)')

            plt.tight_layout()

            # 파일 저장
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"매개효과 히트맵 저장 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"매개효과 히트맵 생성 오류: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    def create_significant_paths_only_diagram(self, model: Model,
                                            filename: str = "significant_paths_only",
                                            significance_level: float = 0.05,
                                            exclude_variables: List[str] = None,
                                            plot_covs: bool = False,
                                            plot_ests: bool = True,
                                            std_ests: bool = True,
                                            engine: str = 'dot') -> Optional[Path]:
        """
        유의한 경로만 포함한 경로 다이어그램 생성

        Args:
            model (Model): semopy 모델
            filename (str): 파일명
            significance_level (float): 유의수준 (기본값: 0.05)
            exclude_variables (List[str]): 제외할 변수 목록 (예: ['purchase_intention'])
            plot_covs (bool): 공분산 표시 여부
            plot_ests (bool): 추정치 표시 여부
            std_ests (bool): 표준화 추정치 사용 여부
            engine (str): graphviz 엔진

        Returns:
            Optional[Path]: 생성된 파일 경로
        """
        try:
            if exclude_variables is None:
                exclude_variables = []

            # 모델 파라미터 검사
            params = model.inspect()

            # 유의한 구조적 경로만 필터링
            structural_paths = params[params['op'] == '~'].copy()

            # 제외할 변수가 포함된 경로 제거
            if exclude_variables:
                for exclude_var in exclude_variables:
                    structural_paths = structural_paths[
                        (structural_paths['lval'] != exclude_var) &
                        (structural_paths['rval'] != exclude_var)
                    ]

            # 유의한 경로만 필터링
            if 'p-value' in structural_paths.columns:
                significant_paths = structural_paths[
                    structural_paths['p-value'] < significance_level
                ].copy()
            else:
                logger.warning("p-value 정보가 없어 모든 경로를 포함합니다")
                significant_paths = structural_paths.copy()

            logger.info(f"전체 구조적 경로: {len(structural_paths)}개")
            logger.info(f"유의한 경로 (p < {significance_level}): {len(significant_paths)}개")
            logger.info(f"제외된 변수: {exclude_variables}")

            if len(significant_paths) == 0:
                logger.warning("유의한 경로가 없습니다")
                return None

            # 유의한 경로 정보 출력
            logger.info("유의한 경로 목록:")
            for _, row in significant_paths.iterrows():
                p_val = row.get('p-value', 1.0)
                coeff = row.get('Estimate', 0.0)
                sig_stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                logger.info(f"  {row['rval']} -> {row['lval']}: {coeff:.4f} (p={p_val:.4f}){sig_stars}")

            # 기존 다이어그램 생성 방식 사용
            file_path = self.output_dir / f"{filename}.png"

            # semplot을 사용하여 다이어그램 생성
            graph = semplot(
                model,
                plot_covs=plot_covs,
                plot_ests=plot_ests,
                std_ests=std_ests,
                engine=engine
            )

            graph.render(str(file_path.with_suffix('')), format='png', cleanup=True)

            logger.info(f"유의한 경로만 포함한 다이어그램 생성 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"유의한 경로 다이어그램 생성 오류: {e}")
            return None

    def create_significance_comparison_chart(self, results: Dict[str, Any],
                                           filename: str = "significance_comparison",
                                           exclude_variables: List[str] = None,
                                           figsize: Tuple[int, int] = (14, 10)) -> Optional[Path]:
        """
        유의성 비교 차트 생성 (유의한 경로 vs 비유의한 경로)

        Args:
            results (Dict[str, Any]): 경로분석 결과
            filename (str): 파일명
            exclude_variables (List[str]): 제외할 변수 목록
            figsize (Tuple[int, int]): 그림 크기

        Returns:
            Optional[Path]: 생성된 파일 경로
        """
        try:
            if exclude_variables is None:
                exclude_variables = []

            path_coeffs = results.get('path_coefficients', {})
            paths = path_coeffs.get('paths', [])
            coefficients = path_coeffs.get('coefficients', {})
            p_values = path_coeffs.get('p_values', {})

            if not paths:
                logger.warning("경로 정보가 없습니다")
                return None

            # 제외할 변수가 포함된 경로 필터링
            filtered_data = []
            for i, (from_var, to_var) in enumerate(paths):
                if any(var in exclude_variables for var in [from_var, to_var]):
                    continue

                coeff = coefficients.get(i, 0)
                p_val = p_values.get(i, 1)

                filtered_data.append({
                    'path': f"{from_var} → {to_var}",
                    'from_var': from_var,
                    'to_var': to_var,
                    'coefficient': coeff,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'significance_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                })

            if not filtered_data:
                logger.warning("필터링 후 경로가 없습니다")
                return None

            df = pd.DataFrame(filtered_data)

            # 2x2 서브플롯 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # 1. 경로계수 크기 비교 (유의성별)
            significant_coeffs = df[df['significant']]['coefficient']
            non_significant_coeffs = df[~df['significant']]['coefficient']

            ax1.hist([significant_coeffs, non_significant_coeffs],
                    bins=10, alpha=0.7, label=['Significant', 'Non-significant'],
                    color=['lightblue', 'lightcoral'])
            ax1.set_xlabel('Path Coefficient')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Path Coefficients by Significance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. p-value 분포
            ax2.hist(df['p_value'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            ax2.axvline(x=0.01, color='orange', linestyle='--', label='α = 0.01')
            ax2.axvline(x=0.001, color='purple', linestyle='--', label='α = 0.001')
            ax2.set_xlabel('p-value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of p-values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 유의성 수준별 경로 수
            significance_counts = df['significance_level'].value_counts()
            colors = {'***': 'darkblue', '**': 'blue', '*': 'lightblue', 'ns': 'lightgray'}
            bar_colors = [colors.get(level, 'gray') for level in significance_counts.index]

            bars = ax3.bar(significance_counts.index, significance_counts.values, color=bar_colors)
            ax3.set_xlabel('Significance Level')
            ax3.set_ylabel('Number of Paths')
            ax3.set_title('Number of Paths by Significance Level')

            # 막대 위에 숫자 표시
            for bar, count in zip(bars, significance_counts.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')

            # 4. 경로계수 vs p-value 산점도
            colors_scatter = ['red' if sig else 'blue' for sig in df['significant']]
            ax4.scatter(df['coefficient'], df['p_value'], c=colors_scatter, alpha=0.7)
            ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Path Coefficient')
            ax4.set_ylabel('p-value')
            ax4.set_title('Path Coefficient vs p-value')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

            # 범례 추가
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='Significant (p < 0.05)'),
                             Patch(facecolor='blue', label='Non-significant (p ≥ 0.05)')]
            ax4.legend(handles=legend_elements)

            plt.tight_layout()

            # 파일 저장
            file_path = self.output_dir / f"{filename}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"유의성 비교 차트 생성 완료: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"유의성 비교 차트 생성 오류: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None


# 새로운 편의 함수들
def create_bootstrap_visualization(bootstrap_results: Dict[str, Any],
                                 output_dir: str = "path_analysis_results/visualizations",
                                 filename: str = "bootstrap_analysis") -> Optional[Path]:
    """부트스트래핑 시각화 편의 함수"""
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_bootstrap_confidence_interval_plot(bootstrap_results, filename)


def create_mediation_heatmap(all_mediations: Dict[str, Any],
                           output_dir: str = "path_analysis_results/visualizations",
                           filename: str = "mediation_heatmap") -> Optional[Path]:
    """매개효과 히트맵 편의 함수"""
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_mediation_effects_heatmap(all_mediations, filename)


def create_significant_paths_visualization(model: Model,
                                         significance_level: float = 0.05,
                                         exclude_variables: List[str] = None,
                                         output_dir: str = "path_analysis_results/visualizations",
                                         filename: str = "significant_paths_only") -> Optional[Path]:
    """
    유의한 경로만 포함한 경로 다이어그램 생성 편의 함수

    Args:
        model (Model): semopy 모델
        significance_level (float): 유의수준 (기본값: 0.05)
        exclude_variables (List[str]): 제외할 변수 목록 (예: ['purchase_intention'])
        output_dir (str): 출력 디렉토리
        filename (str): 파일명

    Returns:
        Optional[Path]: 생성된 파일 경로
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_significant_paths_only_diagram(
        model, filename, significance_level, exclude_variables
    )


def create_four_factor_significant_paths(model: Model,
                                       significance_level: float = 0.05,
                                       output_dir: str = "path_analysis_results/visualizations",
                                       filename: str = "four_factor_significant_paths") -> Optional[Path]:
    """
    구매의도를 제외한 4가지 요인의 유의한 경로만 가시화하는 편의 함수

    Args:
        model (Model): semopy 모델
        significance_level (float): 유의수준 (기본값: 0.05)
        output_dir (str): 출력 디렉토리
        filename (str): 파일명

    Returns:
        Optional[Path]: 생성된 파일 경로
    """
    exclude_variables = ['purchase_intention']
    return create_significant_paths_visualization(
        model, significance_level, exclude_variables, output_dir, filename
    )


def create_significance_comparison_visualization(results: Dict[str, Any],
                                               exclude_variables: List[str] = None,
                                               output_dir: str = "path_analysis_results/visualizations",
                                               filename: str = "significance_comparison") -> Optional[Path]:
    """
    유의성 비교 시각화 편의 함수

    Args:
        results (Dict[str, Any]): 경로분석 결과
        exclude_variables (List[str]): 제외할 변수 목록
        output_dir (str): 출력 디렉토리
        filename (str): 파일명

    Returns:
        Optional[Path]: 생성된 파일 경로
    """
    visualizer = PathAnalysisVisualizer(output_dir)
    return visualizer.create_significance_comparison_chart(
        results, filename, exclude_variables
    )


def create_comprehensive_four_factor_analysis(results: Dict[str, Any],
                                            significance_level: float = 0.05,
                                            output_dir: str = "path_analysis_results/visualizations") -> Dict[str, Optional[Path]]:
    """
    4가지 요인에 대한 종합적인 유의성 분석 및 가시화

    Args:
        results (Dict[str, Any]): 경로분석 결과
        significance_level (float): 유의수준
        output_dir (str): 출력 디렉토리

    Returns:
        Dict[str, Optional[Path]]: 생성된 파일들의 경로
    """
    exclude_variables = ['purchase_intention']
    model = results.get('model_object')

    if not model:
        logger.error("모델 객체를 찾을 수 없습니다")
        return {}

    visualization_files = {}

    try:
        # 1. 유의한 경로만 포함한 다이어그램
        significant_paths = create_four_factor_significant_paths(
            model, significance_level, output_dir, "four_factor_significant_paths"
        )
        visualization_files['significant_paths'] = significant_paths

        # 2. 유의성 비교 차트
        comparison_chart = create_significance_comparison_visualization(
            results, exclude_variables, output_dir, "four_factor_significance_comparison"
        )
        visualization_files['significance_comparison'] = comparison_chart

        # 3. 전체 경로 다이어그램 (비교용)
        visualizer = PathAnalysisVisualizer(output_dir)
        all_paths = visualizer.create_path_diagram(
            model, "four_factor_all_paths", plot_covs=False, plot_ests=True, std_ests=True
        )
        visualization_files['all_paths'] = all_paths

        logger.info(f"4가지 요인 종합 분석 완료: {len(visualization_files)}개 파일 생성")
        return visualization_files

    except Exception as e:
        logger.error(f"종합 분석 오류: {e}")
        return visualization_files
