"""
역문항 처리 전후 요인분석 결과 비교 분석 모듈

이 모듈은 역문항 처리 전후의 요인분석 결과를 비교하여:
- 요인부하량 변화 분석
- 신뢰도 지표 변화 분석
- 모델 적합도 변화 분석
- 시각적 비교 차트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import glob
import json
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class FactorAnalysisComparator:
    """역문항 처리 전후 요인분석 결과 비교 클래스"""
    
    def __init__(self, results_dir: str = "factor_analysis_results",
                 output_dir: str = "comparison_analysis_results"):
        """
        비교 분석기 초기화
        
        Args:
            results_dir (str): 요인분석 결과 디렉토리
            output_dir (str): 비교 분석 결과 출력 디렉토리
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"FactorAnalysisComparator 초기화 완료")
        logger.info(f"결과 디렉토리: {self.results_dir}")
        logger.info(f"출력 디렉토리: {self.output_dir}")
    
    def find_comparison_files(self) -> Dict[str, Dict[str, str]]:
        """
        비교할 파일들을 찾기
        
        Returns:
            Dict[str, Dict[str, str]]: 파일 유형별 pre/post 파일 경로
        """
        try:
            # 패턴별로 파일 찾기
            file_patterns = {
                'loadings': '*loadings.csv',
                'fit_indices': '*fit_indices.csv',
                'metadata': '*metadata.json'
            }
            
            comparison_files = {}
            
            for file_type, pattern in file_patterns.items():
                files = list(self.results_dir.glob(pattern))
                
                pre_files = [f for f in files if isinstance(f.name, str) and 'pre_reverse' in f.name]
                post_files = [f for f in files if isinstance(f.name, str) and 'post_reverse' in f.name]
                
                if pre_files and post_files:
                    # 가장 최신 파일 선택
                    latest_pre = max(pre_files, key=lambda x: x.stat().st_mtime)
                    latest_post = max(post_files, key=lambda x: x.stat().st_mtime)
                    
                    comparison_files[file_type] = {
                        'pre_reverse': str(latest_pre),
                        'post_reverse': str(latest_post)
                    }
                    
                    logger.info(f"{file_type} 비교 파일 찾음:")
                    logger.info(f"  - 처리 전: {latest_pre.name}")
                    logger.info(f"  - 처리 후: {latest_post.name}")
                else:
                    logger.warning(f"{file_type}: 비교할 파일이 부족합니다 (pre: {len(pre_files)}, post: {len(post_files)})")
            
            return comparison_files
            
        except Exception as e:
            logger.error(f"비교 파일 찾기 중 오류: {e}")
            return {}
    
    def compare_factor_loadings(self, comparison_files: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        요인부하량 비교 분석
        
        Args:
            comparison_files (Dict[str, Dict[str, str]]): 비교 파일 정보
            
        Returns:
            pd.DataFrame: 요인부하량 비교 결과
        """
        try:
            if 'loadings' not in comparison_files:
                logger.error("요인부하량 파일을 찾을 수 없습니다.")
                return pd.DataFrame()
            
            # 데이터 로드
            pre_loadings = pd.read_csv(comparison_files['loadings']['pre_reverse'])
            post_loadings = pd.read_csv(comparison_files['loadings']['post_reverse'])
            
            # 비교 테이블 생성
            comparison_data = []
            
            # 공통 문항들에 대해 비교
            common_items = set(pre_loadings['Item']) & set(post_loadings['Item'])
            
            for item in common_items:
                pre_row = pre_loadings[pre_loadings['Item'] == item].iloc[0]
                post_row = post_loadings[post_loadings['Item'] == item].iloc[0]
                
                pre_loading = pre_row['Loading']
                post_loading = post_row['Loading']
                
                comparison_data.append({
                    'Item': item,
                    'Factor': pre_row['Factor'],
                    'Pre_Reverse_Loading': pre_loading,
                    'Post_Reverse_Loading': post_loading,
                    'Loading_Change': post_loading - pre_loading,
                    'Abs_Loading_Change': abs(post_loading - pre_loading),
                    'Sign_Changed': (pre_loading * post_loading) < 0,
                    'Pre_Acceptable': abs(pre_loading) >= 0.5,
                    'Post_Acceptable': abs(post_loading) >= 0.5,
                    'Improvement': abs(post_loading) > abs(pre_loading)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # 결과 저장
            output_file = self.output_dir / "factor_loadings_comparison.csv"
            comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"요인부하량 비교 결과 저장: {output_file}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"요인부하량 비교 중 오류: {e}")
            return pd.DataFrame()
    
    def compare_fit_indices(self, comparison_files: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        모델 적합도 지수 비교
        
        Args:
            comparison_files (Dict[str, Dict[str, str]]): 비교 파일 정보
            
        Returns:
            pd.DataFrame: 적합도 지수 비교 결과
        """
        try:
            if 'fit_indices' not in comparison_files:
                logger.error("적합도 지수 파일을 찾을 수 없습니다.")
                return pd.DataFrame()
            
            # 데이터 로드
            pre_fit = pd.read_csv(comparison_files['fit_indices']['pre_reverse'])
            post_fit = pd.read_csv(comparison_files['fit_indices']['post_reverse'])
            
            # 비교 테이블 생성
            comparison_data = []
            
            # 공통 지수들에 대해 비교
            common_metrics = set(pre_fit['Metric']) & set(post_fit['Metric'])
            
            for metric in common_metrics:
                pre_value = pre_fit[pre_fit['Metric'] == metric]['Value'].iloc[0]
                post_value = post_fit[post_fit['Metric'] == metric]['Value'].iloc[0]
                
                comparison_data.append({
                    'Metric': metric,
                    'Pre_Reverse_Value': pre_value,
                    'Post_Reverse_Value': post_value,
                    'Change': post_value - pre_value,
                    'Percent_Change': ((post_value - pre_value) / abs(pre_value)) * 100 if pre_value != 0 else 0,
                    'Improvement': self._is_fit_improvement(metric, pre_value, post_value)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # 결과 저장
            output_file = self.output_dir / "fit_indices_comparison.csv"
            comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"적합도 지수 비교 결과 저장: {output_file}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"적합도 지수 비교 중 오류: {e}")
            return pd.DataFrame()
    
    def _is_fit_improvement(self, metric: str, pre_value: float, post_value: float) -> bool:
        """
        적합도 지수 개선 여부 판단
        
        Args:
            metric (str): 적합도 지수명
            pre_value (float): 처리 전 값
            post_value (float): 처리 후 값
            
        Returns:
            bool: 개선 여부
        """
        # 높을수록 좋은 지수들
        higher_is_better = ['CFI', 'TLI', 'GFI', 'AGFI']
        # 낮을수록 좋은 지수들  
        lower_is_better = ['RMSEA', 'SRMR', 'Chi-square', 'AIC', 'BIC']
        
        metric_upper = metric.upper()
        
        if any(good_metric in metric_upper for good_metric in higher_is_better):
            return post_value > pre_value
        elif any(bad_metric in metric_upper for bad_metric in lower_is_better):
            return post_value < pre_value
        else:
            # 알 수 없는 지수는 절댓값 기준으로 판단
            return abs(post_value) < abs(pre_value)
    
    def create_comparison_visualizations(self, loadings_comparison: pd.DataFrame,
                                       fit_comparison: pd.DataFrame) -> None:
        """
        비교 분석 시각화 생성
        
        Args:
            loadings_comparison (pd.DataFrame): 요인부하량 비교 결과
            fit_comparison (pd.DataFrame): 적합도 지수 비교 결과
        """
        try:
            # 1. 요인부하량 변화 시각화
            if not loadings_comparison.empty:
                self._plot_loading_changes(loadings_comparison)
            
            # 2. 적합도 지수 변화 시각화
            if not fit_comparison.empty:
                self._plot_fit_changes(fit_comparison)
            
            # 3. 종합 비교 대시보드
            if not loadings_comparison.empty and not fit_comparison.empty:
                self._create_comparison_dashboard(loadings_comparison, fit_comparison)
            
        except Exception as e:
            logger.error(f"비교 시각화 생성 중 오류: {e}")
    
    def _plot_loading_changes(self, loadings_comparison: pd.DataFrame) -> None:
        """요인부하량 변화 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('역문항 처리 전후 요인부하량 변화', fontsize=16, fontweight='bold')
            
            # 1. 처리 전후 산점도
            ax1 = axes[0, 0]
            ax1.scatter(loadings_comparison['Pre_Reverse_Loading'], 
                       loadings_comparison['Post_Reverse_Loading'],
                       alpha=0.6, s=50)
            ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='변화 없음')
            ax1.set_xlabel('처리 전 로딩값')
            ax1.set_ylabel('처리 후 로딩값')
            ax1.set_title('처리 전후 로딩값 비교')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 로딩값 변화량 히스토그램
            ax2 = axes[0, 1]
            ax2.hist(loadings_comparison['Loading_Change'], bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('로딩값 변화량')
            ax2.set_ylabel('빈도')
            ax2.set_title('로딩값 변화량 분포')
            ax2.grid(True, alpha=0.3)
            
            # 3. 요인별 개선 현황
            ax3 = axes[1, 0]
            factor_improvement = loadings_comparison.groupby('Factor').agg({
                'Improvement': 'sum',
                'Item': 'count'
            }).rename(columns={'Item': 'Total_Items'})
            factor_improvement['Improvement_Rate'] = factor_improvement['Improvement'] / factor_improvement['Total_Items']
            
            bars = ax3.bar(factor_improvement.index, factor_improvement['Improvement_Rate'])
            ax3.set_xlabel('요인')
            ax3.set_ylabel('개선 비율')
            ax3.set_title('요인별 로딩값 개선 비율')
            ax3.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, rate in zip(bars, factor_improvement['Improvement_Rate']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
            
            # 4. 부호 변화 문항들
            ax4 = axes[1, 1]
            sign_changed = loadings_comparison[loadings_comparison['Sign_Changed']]
            if not sign_changed.empty:
                y_pos = range(len(sign_changed))
                ax4.barh(y_pos, sign_changed['Pre_Reverse_Loading'], alpha=0.7, label='처리 전')
                ax4.barh(y_pos, sign_changed['Post_Reverse_Loading'], alpha=0.7, label='처리 후')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(sign_changed['Item'])
                ax4.set_xlabel('로딩값')
                ax4.set_title('부호가 변경된 문항들')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '부호가 변경된 문항이 없습니다', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('부호가 변경된 문항들')
            
            plt.tight_layout()
            
            # 저장
            output_file = self.output_dir / "factor_loadings_comparison_charts.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"요인부하량 변화 시각화 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"요인부하량 변화 시각화 중 오류: {e}")
    
    def _plot_fit_changes(self, fit_comparison: pd.DataFrame) -> None:
        """적합도 지수 변화 시각화"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('역문항 처리 전후 모델 적합도 변화', fontsize=16, fontweight='bold')
            
            # 1. 적합도 지수 변화량
            metrics = fit_comparison['Metric']
            changes = fit_comparison['Change']
            improvements = fit_comparison['Improvement']
            
            colors = ['green' if imp else 'red' for imp in improvements]
            bars = ax1.bar(range(len(metrics)), changes, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.set_ylabel('변화량')
            ax1.set_title('적합도 지수 변화량')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # 값 표시
            for bar, change in zip(bars, changes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, 
                        height + (0.01 if height >= 0 else -0.01),
                        f'{change:.4f}', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=8)
            
            # 2. 개선 현황 파이차트
            improvement_counts = fit_comparison['Improvement'].value_counts()
            labels = ['개선됨', '악화됨']
            sizes = [improvement_counts.get(True, 0), improvement_counts.get(False, 0)]
            colors_pie = ['lightgreen', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('적합도 지수 개선 현황')
            
            plt.tight_layout()
            
            # 저장
            output_file = self.output_dir / "fit_indices_comparison_charts.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"적합도 지수 변화 시각화 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"적합도 지수 변화 시각화 중 오류: {e}")

    def _create_comparison_dashboard(self, loadings_comparison: pd.DataFrame,
                                   fit_comparison: pd.DataFrame) -> None:
        """종합 비교 대시보드 생성"""
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('역문항 처리 전후 종합 비교 대시보드', fontsize=18, fontweight='bold')

            # 그리드 레이아웃 설정
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

            # 1. 전체 개선 요약 (좌상단)
            ax1 = fig.add_subplot(gs[0, :2])

            # 로딩값 개선 통계
            loading_improved = loadings_comparison['Improvement'].sum()
            loading_total = len(loadings_comparison)
            loading_improvement_rate = loading_improved / loading_total

            # 적합도 개선 통계
            fit_improved = fit_comparison['Improvement'].sum()
            fit_total = len(fit_comparison)
            fit_improvement_rate = fit_improved / fit_total

            categories = ['요인부하량', '적합도 지수']
            improvement_rates = [loading_improvement_rate, fit_improvement_rate]

            bars = ax1.bar(categories, improvement_rates, color=['skyblue', 'lightgreen'], alpha=0.8)
            ax1.set_ylabel('개선 비율')
            ax1.set_title('전체 개선 현황', fontweight='bold')
            ax1.set_ylim(0, 1)

            # 값 표시
            for bar, rate in zip(bars, improvement_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

            # 2. 문제 문항 개선 현황 (우상단)
            ax2 = fig.add_subplot(gs[0, 2:])

            # 처리 전 문제 문항들 (로딩값 < 0.5)
            pre_problematic = loadings_comparison[~loadings_comparison['Pre_Acceptable']]
            post_acceptable = pre_problematic['Post_Acceptable'].sum()

            if len(pre_problematic) > 0:
                problem_improvement_rate = post_acceptable / len(pre_problematic)

                labels = ['개선됨', '여전히 문제']
                sizes = [post_acceptable, len(pre_problematic) - post_acceptable]
                colors = ['lightgreen', 'lightcoral']

                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'문제 문항 개선 현황\n(총 {len(pre_problematic)}개 문항)', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, '처리 전 문제 문항이\n없었습니다',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('문제 문항 개선 현황', fontweight='bold')

            # 3. 요인별 상세 비교 (중간 행)
            ax3 = fig.add_subplot(gs[1, :])

            factors = loadings_comparison['Factor'].unique()
            x = np.arange(len(factors))
            width = 0.35

            # 요인별 평균 로딩값 (절댓값)
            factor_stats = loadings_comparison.groupby('Factor').agg({
                'Pre_Reverse_Loading': lambda x: np.mean(np.abs(x)),
                'Post_Reverse_Loading': lambda x: np.mean(np.abs(x)),
                'Improvement': 'mean'
            })

            bars1 = ax3.bar(x - width/2, factor_stats['Pre_Reverse_Loading'], width,
                           label='처리 전', alpha=0.8, color='lightcoral')
            bars2 = ax3.bar(x + width/2, factor_stats['Post_Reverse_Loading'], width,
                           label='처리 후', alpha=0.8, color='lightblue')

            ax3.set_xlabel('요인')
            ax3.set_ylabel('평균 로딩값 (절댓값)')
            ax3.set_title('요인별 평균 로딩값 비교', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(factors, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            # 4. 주요 개선 문항 (하단 좌측)
            ax4 = fig.add_subplot(gs[2, :2])

            # 가장 많이 개선된 문항들 (상위 10개)
            top_improvements = loadings_comparison.nlargest(10, 'Abs_Loading_Change')

            if not top_improvements.empty:
                y_pos = range(len(top_improvements))
                bars = ax4.barh(y_pos, top_improvements['Abs_Loading_Change'], alpha=0.7)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(top_improvements['Item'])
                ax4.set_xlabel('로딩값 변화량 (절댓값)')
                ax4.set_title('가장 많이 개선된 문항들 (상위 10개)', fontweight='bold')
                ax4.grid(True, alpha=0.3)

                # 값 표시
                for bar, change in zip(bars, top_improvements['Abs_Loading_Change']):
                    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{change:.3f}', ha='left', va='center', fontsize=8)

            # 5. 적합도 지수 상세 (하단 우측)
            ax5 = fig.add_subplot(gs[2, 2:])

            if not fit_comparison.empty:
                metrics = fit_comparison['Metric']
                pre_values = fit_comparison['Pre_Reverse_Value']
                post_values = fit_comparison['Post_Reverse_Value']

                x = np.arange(len(metrics))
                width = 0.35

                bars1 = ax5.bar(x - width/2, pre_values, width, label='처리 전', alpha=0.8)
                bars2 = ax5.bar(x + width/2, post_values, width, label='처리 후', alpha=0.8)

                ax5.set_xlabel('적합도 지수')
                ax5.set_ylabel('값')
                ax5.set_title('적합도 지수 상세 비교', fontweight='bold')
                ax5.set_xticks(x)
                ax5.set_xticklabels(metrics, rotation=45)
                ax5.legend()
                ax5.grid(True, alpha=0.3)

            plt.tight_layout()

            # 저장
            output_file = self.output_dir / "comprehensive_comparison_dashboard.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"종합 비교 대시보드 저장: {output_file}")

        except Exception as e:
            logger.error(f"종합 비교 대시보드 생성 중 오류: {e}")

    def generate_comparison_report(self, loadings_comparison: pd.DataFrame,
                                 fit_comparison: pd.DataFrame) -> str:
        """
        비교 분석 보고서 생성

        Args:
            loadings_comparison (pd.DataFrame): 요인부하량 비교 결과
            fit_comparison (pd.DataFrame): 적합도 지수 비교 결과

        Returns:
            str: 보고서 파일 경로
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("역문항 처리 전후 요인분석 결과 비교 보고서")
            report_lines.append("=" * 80)
            report_lines.append("")
            report_lines.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # 1. 전체 요약
            report_lines.append("1. 전체 요약")
            report_lines.append("-" * 40)

            if not loadings_comparison.empty:
                total_items = len(loadings_comparison)
                improved_items = loadings_comparison['Improvement'].sum()
                sign_changed_items = loadings_comparison['Sign_Changed'].sum()

                report_lines.append(f"분석 문항 수: {total_items}개")
                report_lines.append(f"로딩값 개선 문항: {improved_items}개 ({improved_items/total_items:.1%})")
                report_lines.append(f"부호 변경 문항: {sign_changed_items}개")

            if not fit_comparison.empty:
                total_indices = len(fit_comparison)
                improved_indices = fit_comparison['Improvement'].sum()

                report_lines.append(f"적합도 지수 수: {total_indices}개")
                report_lines.append(f"개선된 지수: {improved_indices}개 ({improved_indices/total_indices:.1%})")

            report_lines.append("")

            # 2. 요인별 상세 분석
            if not loadings_comparison.empty:
                report_lines.append("2. 요인별 상세 분석")
                report_lines.append("-" * 40)

                for factor in loadings_comparison['Factor'].unique():
                    factor_data = loadings_comparison[loadings_comparison['Factor'] == factor]

                    report_lines.append(f"\n[{factor}]")
                    report_lines.append(f"  문항 수: {len(factor_data)}개")
                    report_lines.append(f"  개선 문항: {factor_data['Improvement'].sum()}개")
                    report_lines.append(f"  평균 로딩값 변화: {factor_data['Loading_Change'].mean():.4f}")

                    # 문제 문항들
                    problematic = factor_data[~factor_data['Pre_Acceptable']]
                    if not problematic.empty:
                        improved_problematic = problematic['Post_Acceptable'].sum()
                        report_lines.append(f"  문제 문항 개선: {improved_problematic}/{len(problematic)}개")

            # 3. 주요 개선 사항
            report_lines.append("\n\n3. 주요 개선 사항")
            report_lines.append("-" * 40)

            if not loadings_comparison.empty:
                # 가장 많이 개선된 문항들
                top_improvements = loadings_comparison.nlargest(5, 'Abs_Loading_Change')
                report_lines.append("\n가장 많이 개선된 문항들:")
                for _, row in top_improvements.iterrows():
                    report_lines.append(f"  - {row['Item']}: {row['Pre_Reverse_Loading']:.4f} → {row['Post_Reverse_Loading']:.4f}")

                # 부호가 변경된 문항들
                sign_changed = loadings_comparison[loadings_comparison['Sign_Changed']]
                if not sign_changed.empty:
                    report_lines.append("\n부호가 변경된 문항들:")
                    for _, row in sign_changed.iterrows():
                        report_lines.append(f"  - {row['Item']}: {row['Pre_Reverse_Loading']:.4f} → {row['Post_Reverse_Loading']:.4f}")

            # 4. 적합도 지수 변화
            if not fit_comparison.empty:
                report_lines.append("\n\n4. 적합도 지수 변화")
                report_lines.append("-" * 40)

                for _, row in fit_comparison.iterrows():
                    status = "개선" if row['Improvement'] else "악화"
                    report_lines.append(f"  - {row['Metric']}: {row['Pre_Reverse_Value']:.4f} → {row['Post_Reverse_Value']:.4f} ({status})")

            # 5. 결론 및 권장사항
            report_lines.append("\n\n5. 결론 및 권장사항")
            report_lines.append("-" * 40)

            if not loadings_comparison.empty:
                improvement_rate = loadings_comparison['Improvement'].mean()
                if improvement_rate > 0.7:
                    report_lines.append("✅ 역문항 처리가 매우 효과적이었습니다.")
                elif improvement_rate > 0.5:
                    report_lines.append("✅ 역문항 처리가 효과적이었습니다.")
                else:
                    report_lines.append("⚠️ 역문항 처리 효과가 제한적입니다. 추가 검토가 필요합니다.")

                # 여전히 문제인 문항들
                still_problematic = loadings_comparison[
                    ~loadings_comparison['Pre_Acceptable'] &
                    ~loadings_comparison['Post_Acceptable']
                ]

                if not still_problematic.empty:
                    report_lines.append(f"\n⚠️ 여전히 문제인 문항들 ({len(still_problematic)}개):")
                    for _, row in still_problematic.iterrows():
                        report_lines.append(f"  - {row['Item']}: {row['Post_Reverse_Loading']:.4f}")
                    report_lines.append("  → 이 문항들에 대한 추가 검토가 필요합니다.")

            # 보고서 저장
            report_content = "\n".join(report_lines)
            report_file = self.output_dir / f"comparison_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"비교 분석 보고서 저장: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"비교 분석 보고서 생성 중 오류: {e}")
            return ""

    def run_complete_comparison(self) -> Dict[str, Any]:
        """
        완전한 비교 분석 실행

        Returns:
            Dict[str, Any]: 비교 분석 결과
        """
        try:
            logger.info("역문항 처리 전후 비교 분석 시작")

            # 1. 비교 파일 찾기
            comparison_files = self.find_comparison_files()
            if not comparison_files:
                return {'error': '비교할 파일을 찾을 수 없습니다.'}

            # 2. 요인부하량 비교
            loadings_comparison = self.compare_factor_loadings(comparison_files)

            # 3. 적합도 지수 비교
            fit_comparison = self.compare_fit_indices(comparison_files)

            # 4. 시각화 생성
            self.create_comparison_visualizations(loadings_comparison, fit_comparison)

            # 5. 보고서 생성
            report_file = self.generate_comparison_report(loadings_comparison, fit_comparison)

            logger.info("비교 분석 완료")

            return {
                'loadings_comparison': loadings_comparison,
                'fit_comparison': fit_comparison,
                'report_file': report_file,
                'output_dir': str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"완전한 비교 분석 중 오류: {e}")
            return {'error': str(e)}


def run_comparison_analysis(results_dir: str = "factor_analysis_results",
                          output_dir: str = "comparison_analysis_results") -> Dict[str, Any]:
    """
    비교 분석 실행 편의 함수

    Args:
        results_dir (str): 요인분석 결과 디렉토리
        output_dir (str): 출력 디렉토리

    Returns:
        Dict[str, Any]: 비교 분석 결과
    """
    comparator = FactorAnalysisComparator(results_dir, output_dir)
    return comparator.run_complete_comparison()
