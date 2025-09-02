"""
semopy 내장 가시화 모듈

이 모듈은 semopy의 내장 가시화 기능을 활용하여
SEM 모델의 경로 다이어그램을 생성합니다.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import pandas as pd

# semopy 내장 가시화 임포트
try:
    import semopy
    from semopy import Model, semplot
    from semopy.plot import semplot as plot_semplot
    SEMOPY_AVAILABLE = True
except ImportError as e:
    logging.error(f"semopy를 찾을 수 없습니다: {e}")
    SEMOPY_AVAILABLE = False

# graphviz 확인 및 경로 설정
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True

    # Windows에서 Graphviz 경로 자동 설정
    if sys.platform.startswith('win'):
        import subprocess

        # 일반적인 Graphviz 설치 경로들
        possible_paths = [
            r'C:\Program Files\Graphviz\bin',
            r'C:\Program Files (x86)\Graphviz\bin',
            r'C:\Graphviz\bin'
        ]

        # PATH에 dot이 없는 경우 자동으로 추가
        try:
            subprocess.run(['dot', '-V'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # dot을 찾을 수 없으면 경로 추가 시도
            for path in possible_paths:
                dot_exe = os.path.join(path, 'dot.exe')
                if os.path.exists(dot_exe):
                    os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
                    print(f"Graphviz 경로를 PATH에 추가했습니다: {path}")
                    break
            else:
                print("Graphviz 실행 파일을 찾을 수 없습니다.")

except ImportError:
    GRAPHVIZ_AVAILABLE = False

from .config import FactorAnalysisConfig, get_default_config

logger = logging.getLogger(__name__)


class SemopyNativeVisualizer:
    """semopy 내장 가시화 기능을 사용하는 클래스"""
    
    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        """
        semopy Native Visualizer 초기화
        
        Args:
            config (Optional[FactorAnalysisConfig]): 설정 객체
        """
        self.config = config if config is not None else get_default_config()
        
        # 의존성 확인
        if not SEMOPY_AVAILABLE:
            raise ImportError("semopy가 설치되지 않았습니다. pip install semopy로 설치해주세요.")
        
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("graphviz가 설치되지 않았습니다. pip install graphviz로 설치해주세요.")
            logger.warning("일부 가시화 기능이 제한될 수 있습니다.")
        
        logger.info("semopy Native Visualizer 초기화 완료")
    
    def create_sem_diagram(self, model: Model,
                          filename: str,
                          plot_covs: bool = False,
                          plot_exos: bool = True,
                          plot_ests: bool = True,
                          std_ests: bool = True,
                          engine: str = 'dot',
                          latshape: str = 'circle',
                          show: bool = False,
                          dot_only: bool = False) -> Optional[str]:
        """
        semopy 내장 semplot을 사용하여 SEM 다이어그램 생성

        Args:
            model (Model): 적합된 semopy 모델
            filename (str): 저장할 파일명 (확장자 제외)
            plot_covs (bool): 공분산 표시 여부
            plot_exos (bool): 외생변수 표시 여부
            plot_ests (bool): 추정값 표시 여부
            std_ests (bool): 표준화 추정값 사용 여부
            engine (str): graphviz 엔진 ('dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo')
            latshape (str): 잠재변수 모양 ('circle', 'ellipse', 'box')
            show (bool): 즉시 표시 여부
            dot_only (bool): DOT 파일만 생성 (Graphviz 실행 파일 불필요)

        Returns:
            Optional[str]: 생성된 파일 경로
        """
        if not GRAPHVIZ_AVAILABLE and not dot_only:
            logger.warning("graphviz가 설치되지 않았습니다. DOT 파일만 생성합니다.")
            dot_only = True
        
        try:
            # 파일명에서 경로와 이름 분리
            filepath = Path(filename)
            output_dir = filepath.parent
            base_name = filepath.name

            # 출력 디렉토리로 이동하여 실행 (경로 문제 해결)
            original_cwd = os.getcwd()
            if output_dir != Path('.') and output_dir.exists():
                os.chdir(output_dir)
                actual_filename = base_name
            else:
                actual_filename = filename

            try:
                if dot_only:
                    # DOT 파일만 생성 (Graphviz 실행 파일 불필요)
                    try:
                        # 수동으로 DOT 다이어그램 생성
                        dot_content = self._create_manual_dot(
                            model=model,
                            plot_covs=plot_covs,
                            plot_ests=plot_ests,
                            std_ests=std_ests,
                            latshape=latshape
                        )

                        # DOT 파일 저장
                        dot_filename = f"{actual_filename}.dot"
                        with open(dot_filename, 'w', encoding='utf-8') as f:
                            f.write(dot_content)

                        logger.info(f"수동 DOT 파일 생성 완료: {dot_filename}")
                        return dot_filename

                    except Exception as dot_error:
                        logger.error(f"DOT 파일 생성 실패: {dot_error}")
                        return None
                else:
                    # 일반적인 PNG 파일 생성 (Graphviz 실행 파일 필요)
                    filename_with_ext = f"{actual_filename}.png"

                    graph = semplot(
                        mod=model,
                        filename=filename_with_ext,
                        plot_covs=plot_covs,
                        plot_exos=plot_exos,
                        plot_ests=plot_ests,
                        std_ests=std_ests,
                        engine=engine,
                        latshape=latshape,
                        show=show
                    )

                    logger.info(f"semplot 실행 완료: {filename_with_ext}")

            finally:
                # 원래 디렉토리로 복귀
                os.chdir(original_cwd)
            
            # 생성된 파일 확인
            if dot_only:
                expected_file = f"{filename}.dot"
                extension = ".dot"
            else:
                expected_file = f"{filename}.png"
                extension = ".png"

            # 파일 존재 확인
            if os.path.exists(expected_file):
                logger.info(f"SEM 다이어그램 생성 완료: {expected_file}")
                return expected_file

            # 출력 디렉토리에서도 확인
            if output_dir != Path('.') and output_dir.exists():
                alt_file = output_dir / f"{base_name}{extension}"
                if alt_file.exists():
                    logger.info(f"SEM 다이어그램 생성 완료: {alt_file}")
                    return str(alt_file)

            # 파일을 찾을 수 없는 경우
            logger.warning(f"생성된 파일을 찾을 수 없습니다: {expected_file}")
            return expected_file  # 예상 경로 반환
                
        except Exception as e:
            logger.error(f"SEM 다이어그램 생성 중 오류: {e}")
            return None
    
    def create_multiple_diagrams(self, model: Model,
                                base_filename: str,
                                output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Optional[str]]:
        """
        다양한 옵션으로 여러 SEM 다이어그램 생성
        
        Args:
            model (Model): 적합된 semopy 모델
            base_filename (str): 기본 파일명
            output_dir (Optional[Union[str, Path]]): 출력 디렉토리
            
        Returns:
            Dict[str, Optional[str]]: 생성된 다이어그램들의 파일 경로
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')
        
        diagrams = {}
        
        # 1. 기본 다이어그램 (표준화 추정값 포함)
        basic_file = output_path / f"{base_filename}_basic"
        diagrams['basic'] = self.create_sem_diagram(
            model=model,
            filename=str(basic_file),
            plot_ests=True,
            std_ests=True,
            plot_covs=False,
            engine='dot'
        )
        
        # 2. 상세 다이어그램 (공분산 포함)
        detailed_file = output_path / f"{base_filename}_detailed"
        diagrams['detailed'] = self.create_sem_diagram(
            model=model,
            filename=str(detailed_file),
            plot_ests=True,
            std_ests=True,
            plot_covs=True,
            engine='dot'
        )
        
        # 3. 간단한 다이어그램 (추정값 없음)
        simple_file = output_path / f"{base_filename}_simple"
        diagrams['simple'] = self.create_sem_diagram(
            model=model,
            filename=str(simple_file),
            plot_ests=False,
            std_ests=False,
            plot_covs=False,
            engine='dot'
        )
        
        # 4. 원형 레이아웃
        circular_file = output_path / f"{base_filename}_circular"
        diagrams['circular'] = self.create_sem_diagram(
            model=model,
            filename=str(circular_file),
            plot_ests=True,
            std_ests=True,
            plot_covs=False,
            engine='circo'
        )
        
        # 5. 비표준화 추정값
        unstd_file = output_path / f"{base_filename}_unstandardized"
        diagrams['unstandardized'] = self.create_sem_diagram(
            model=model,
            filename=str(unstd_file),
            plot_ests=True,
            std_ests=False,
            plot_covs=False,
            engine='dot'
        )
        
        return diagrams
    
    def visualize_analysis_results(self, analysis_results: Dict[str, Any],
                                 output_dir: Optional[Union[str, Path]] = None,
                                 base_filename: str = "sem_model") -> Dict[str, Any]:
        """
        분석 결과로부터 semopy 내장 가시화 수행
        
        Args:
            analysis_results (Dict[str, Any]): factor_analyzer의 분석 결과
            output_dir (Optional[Union[str, Path]]): 출력 디렉토리
            base_filename (str): 기본 파일명
            
        Returns:
            Dict[str, Any]: 가시화 결과 정보
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')
        
        visualization_results = {
            'diagrams_generated': {},
            'errors': [],
            'summary': {}
        }
        
        try:
            # semopy 모델 객체 추출 시도
            model = self._extract_model_from_results(analysis_results)
            
            if model is None:
                error_msg = "분석 결과에서 semopy 모델을 찾을 수 없습니다."
                logger.error(error_msg)
                visualization_results['errors'].append(error_msg)
                return visualization_results
            
            # 다양한 다이어그램 생성
            diagrams = self.create_multiple_diagrams(
                model=model,
                base_filename=base_filename,
                output_dir=output_path
            )
            
            visualization_results['diagrams_generated'] = diagrams
            
            # 성공적으로 생성된 다이어그램 수 계산
            successful_diagrams = [name for name, path in diagrams.items() if path is not None]
            visualization_results['summary'] = {
                'total_attempted': len(diagrams),
                'successful': len(successful_diagrams),
                'failed': len(diagrams) - len(successful_diagrams),
                'successful_types': successful_diagrams
            }
            
            logger.info(f"semopy 내장 가시화 완료: {len(successful_diagrams)}/{len(diagrams)}개 성공")
            
        except Exception as e:
            error_msg = f"semopy 내장 가시화 중 오류: {e}"
            logger.error(error_msg)
            visualization_results['errors'].append(error_msg)
        
        return visualization_results
    
    def _extract_model_from_results(self, analysis_results: Dict[str, Any]) -> Optional[Model]:
        """분석 결과에서 semopy 모델 객체 추출"""
        # 직접적인 모델 객체 확인
        if 'model' in analysis_results:
            return analysis_results['model']
        
        # 모델 정보로부터 재구성 시도
        if all(key in analysis_results for key in ['factor_loadings', 'model_info']):
            try:
                # 새로운 모델 생성 및 적합 (분석 결과 재현용)
                return self._recreate_model_from_results(analysis_results)
            except Exception as e:
                logger.warning(f"모델 재구성 실패: {e}")
        
        return None
    
    def _recreate_model_from_results(self, analysis_results: Dict[str, Any]) -> Optional[Model]:
        """분석 결과로부터 모델 재구성 (고급 기능)"""
        # 이 기능은 복잡하므로 현재는 None 반환
        # 실제 구현시에는 factor_loadings와 model_info를 사용하여
        # 모델 스펙을 재구성하고 데이터를 다시 적합해야 함
        logger.info("모델 재구성 기능은 현재 구현되지 않았습니다.")
        return None

    def _create_manual_dot(self, model: Model,
                          plot_covs: bool = False,
                          plot_ests: bool = True,
                          std_ests: bool = True,
                          latshape: str = 'circle') -> str:
        """
        수동으로 DOT 다이어그램 생성 (Graphviz 실행 파일 불필요)

        Args:
            model (Model): semopy 모델
            plot_covs (bool): 공분산 표시 여부
            plot_ests (bool): 추정값 표시 여부
            std_ests (bool): 표준화 추정값 사용 여부
            latshape (str): 잠재변수 모양

        Returns:
            str: DOT 다이어그램 소스 코드
        """
        # 모델 파라미터 추출
        if std_ests:
            params = model.inspect(std_est=True)
        else:
            params = model.inspect()

        loadings = params[params['op'] == '~']
        covariances = params[params['op'] == '~~'] if plot_covs else pd.DataFrame()

        # DOT 다이어그램 시작
        dot_lines = []
        dot_lines.append('digraph SEM {')
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [fontname="Arial"];')
        dot_lines.append('  edge [fontname="Arial"];')

        # 잠재변수들 추출
        factors = loadings['rval'].unique()

        # 잠재변수 노드 정의
        for factor in factors:
            if latshape == 'circle':
                shape_attr = 'shape=circle, style=filled, fillcolor=lightblue'
            elif latshape == 'ellipse':
                shape_attr = 'shape=ellipse, style=filled, fillcolor=lightblue'
            else:  # box
                shape_attr = 'shape=box, style=filled, fillcolor=lightblue'

            dot_lines.append(f'  "{factor}" [{shape_attr}];')

        # 관측변수들과 factor loading
        for _, row in loadings.iterrows():
            item = row['lval']
            factor = row['rval']

            if std_ests and 'Est. Std' in row:
                loading = row['Est. Std']
            else:
                loading = row['Estimate']

            # 관측변수 노드
            dot_lines.append(f'  "{item}" [shape=box];')

            # Factor loading 화살표
            if plot_ests:
                if pd.notna(loading):
                    label = f'label="{loading:.3f}"'
                else:
                    label = ''
            else:
                label = ''

            dot_lines.append(f'  "{factor}" -> "{item}" [{label}];')

        # 공분산 (요청된 경우)
        if plot_covs and len(covariances) > 0:
            for _, row in covariances.iterrows():
                var1 = row['lval']
                var2 = row['rval']

                if var1 != var2:  # 분산이 아닌 공분산만
                    if std_ests and 'Est. Std' in row:
                        cov_val = row['Est. Std']
                    else:
                        cov_val = row['Estimate']

                    if plot_ests and pd.notna(cov_val):
                        label = f'label="{cov_val:.3f}"'
                    else:
                        label = ''

                    # 양방향 화살표로 공분산 표시
                    dot_lines.append(f'  "{var1}" -> "{var2}" [dir=both, style=dashed, {label}];')

        dot_lines.append('}')

        return '\\n'.join(dot_lines)


class SemopyModelExtractor:
    """분석 결과에서 semopy 모델을 추출하는 유틸리티 클래스"""

    @staticmethod
    def extract_from_analyzer(analyzer_instance) -> Optional[Model]:
        """SemopyAnalyzer 인스턴스에서 모델 추출"""
        try:
            if hasattr(analyzer_instance, 'model') and analyzer_instance.model is not None:
                return analyzer_instance.model
        except Exception as e:
            logger.warning(f"Analyzer에서 모델 추출 실패: {e}")
        return None

    @staticmethod
    def create_fresh_model(factor_names: Union[str, List[str]],
                          data_dir: Optional[str] = None) -> Optional[Model]:
        """새로운 모델 생성 및 적합"""
        try:
            from .factor_analyzer import FactorAnalyzer
            from .data_loader import FactorDataLoader
            from .config import create_factor_model_spec

            # 데이터 로딩
            loader = FactorDataLoader(data_dir)
            if isinstance(factor_names, str):
                data = loader.load_single_factor(factor_names)
                spec = create_factor_model_spec(single_factor=factor_names)
            else:
                factor_data = loader.load_multiple_factors(factor_names)
                data = loader.merge_factors_for_analysis(factor_data)
                spec = create_factor_model_spec(factor_names=factor_names)

            # 모델 생성 및 적합
            clean_data = data.drop('no', axis=1).dropna()
            model = Model(spec)
            model.fit(clean_data)

            return model

        except Exception as e:
            logger.error(f"새 모델 생성 실패: {e}")
            return None


# 편의 함수들
def create_sem_diagram(model: Model,
                      filename: str,
                      **kwargs) -> Optional[str]:
    """
    SEM 다이어그램 생성 편의 함수

    Args:
        model (Model): semopy 모델
        filename (str): 파일명
        **kwargs: semplot 함수의 추가 인자들

    Returns:
        Optional[str]: 생성된 파일 경로
    """
    visualizer = SemopyNativeVisualizer()
    return visualizer.create_sem_diagram(model, filename, **kwargs)


def visualize_with_semopy(analysis_results: Dict[str, Any],
                         output_dir: Optional[Union[str, Path]] = None,
                         base_filename: str = "sem_model") -> Dict[str, Any]:
    """
    semopy 내장 가시화 편의 함수

    Args:
        analysis_results (Dict[str, Any]): 분석 결과
        output_dir (Optional[Union[str, Path]]): 출력 디렉토리
        base_filename (str): 기본 파일명

    Returns:
        Dict[str, Any]: 가시화 결과
    """
    visualizer = SemopyNativeVisualizer()
    return visualizer.visualize_analysis_results(analysis_results, output_dir, base_filename)


def create_diagrams_for_factors(factor_names: Union[str, List[str]],
                               output_dir: Optional[Union[str, Path]] = None,
                               data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    요인명으로부터 직접 SEM 다이어그램 생성

    Args:
        factor_names (Union[str, List[str]]): 요인명 또는 요인명 리스트
        output_dir (Optional[Union[str, Path]]): 출력 디렉토리
        data_dir (Optional[str]): 데이터 디렉토리

    Returns:
        Dict[str, Any]: 가시화 결과
    """
    try:
        # 새 모델 생성
        model = SemopyModelExtractor.create_fresh_model(factor_names, data_dir)

        if model is None:
            return {
                'diagrams_generated': {},
                'errors': ['모델 생성 실패'],
                'summary': {'total_attempted': 0, 'successful': 0, 'failed': 0}
            }

        # 가시화 수행
        visualizer = SemopyNativeVisualizer()

        if isinstance(factor_names, str):
            base_filename = f"sem_{factor_names}"
        else:
            base_filename = f"sem_{'_'.join(factor_names[:3])}"  # 최대 3개 요인명 사용

        return visualizer.create_multiple_diagrams(model, base_filename, output_dir)

    except Exception as e:
        logger.error(f"요인 기반 다이어그램 생성 실패: {e}")
        return {
            'diagrams_generated': {},
            'errors': [str(e)],
            'summary': {'total_attempted': 0, 'successful': 0, 'failed': 1}
        }


class IntegratedSemopyVisualizer:
    """semopy 내장 가시화와 커스텀 가시화를 통합하는 클래스"""

    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        """통합 가시화 클래스 초기화"""
        self.config = config if config is not None else get_default_config()
        self.native_visualizer = SemopyNativeVisualizer(config)

        # 커스텀 가시화 모듈 임포트 (순환 임포트 방지)
        try:
            from .visualizer import FactorAnalysisVisualizer
            self.custom_visualizer = FactorAnalysisVisualizer(config)
        except ImportError:
            self.custom_visualizer = None
            logger.warning("커스텀 가시화 모듈을 로드할 수 없습니다.")

    def create_comprehensive_visualization(self, analysis_results: Dict[str, Any],
                                         output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        semopy 내장 + 커스텀 가시화 종합 수행

        Args:
            analysis_results (Dict[str, Any]): 분석 결과
            output_dir (Optional[Union[str, Path]]): 출력 디렉토리

        Returns:
            Dict[str, Any]: 종합 가시화 결과
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')

        comprehensive_results = {
            'semopy_native': {},
            'custom_plots': {},
            'errors': [],
            'summary': {}
        }

        # 1. semopy 내장 가시화
        try:
            native_results = self.native_visualizer.visualize_analysis_results(
                analysis_results,
                output_path / 'semopy_native'
            )
            comprehensive_results['semopy_native'] = native_results
        except Exception as e:
            error_msg = f"semopy 내장 가시화 오류: {e}"
            logger.error(error_msg)
            comprehensive_results['errors'].append(error_msg)

        # 2. 커스텀 가시화
        if self.custom_visualizer:
            try:
                custom_results = self.custom_visualizer.visualize_analysis_results(
                    analysis_results,
                    output_path / 'custom_plots',
                    show_plots=False
                )
                comprehensive_results['custom_plots'] = custom_results
            except Exception as e:
                error_msg = f"커스텀 가시화 오류: {e}"
                logger.error(error_msg)
                comprehensive_results['errors'].append(error_msg)

        # 3. 결과 요약
        native_success = len(comprehensive_results['semopy_native'].get('diagrams_generated', {}))
        custom_success = len(comprehensive_results['custom_plots'].get('plots_generated', []))

        comprehensive_results['summary'] = {
            'semopy_diagrams': native_success,
            'custom_plots': custom_success,
            'total_visualizations': native_success + custom_success,
            'errors': len(comprehensive_results['errors'])
        }

        logger.info(f"종합 가시화 완료: semopy {native_success}개, 커스텀 {custom_success}개")

        return comprehensive_results
