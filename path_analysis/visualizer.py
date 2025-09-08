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
