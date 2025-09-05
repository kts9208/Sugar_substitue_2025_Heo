"""
신뢰도 분석 시각화 모듈

이 모듈은 신뢰도 분석 결과를 다양한 형태로 시각화합니다:
- 신뢰도 지표 요약 테이블
- 요인간 상관관계 히트맵
- 신뢰도 지표 비교 차트
- 판별타당도 검증 결과 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class ReliabilityVisualizer:
    """신뢰도 분석 결과 시각화 클래스"""
    
    def __init__(self, output_dir: str = "reliability_analysis_results"):
        """
        시각화 클래스 초기화
        
        Args:
            output_dir (str): 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Reliability Visualizer 초기화 완료: {self.output_dir}")
    
    def create_reliability_summary_table(self, reliability_results: Dict[str, Any]) -> pd.DataFrame:
        """
        신뢰도 요약 테이블 생성 및 저장
        
        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
            
        Returns:
            pd.DataFrame: 신뢰도 요약 테이블
        """
        try:
            reliability_stats = reliability_results['reliability_stats']
            
            # 테이블 데이터 준비
            table_data = []
            for factor_name, stats in reliability_stats.items():
                row = {
                    'Factor': factor_name,
                    'Items': ', '.join(stats.get('items', [])),
                    'N_Items': stats.get('n_items', 0),
                    'Cronbach_Alpha': stats.get('cronbach_alpha', np.nan),
                    'Composite_Reliability': stats.get('composite_reliability', np.nan),
                    'AVE': stats.get('ave', np.nan),
                    'Sqrt_AVE': stats.get('sqrt_ave', np.nan),
                    'Mean_Loading': stats.get('mean_loading', np.nan),
                    'Min_Loading': stats.get('min_loading', np.nan),
                    'Max_Loading': stats.get('max_loading', np.nan)
                }
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            
            # 수용성 기준 추가
            df['Alpha_Acceptable'] = df['Cronbach_Alpha'] >= 0.7
            df['CR_Acceptable'] = df['Composite_Reliability'] >= 0.7
            df['AVE_Acceptable'] = df['AVE'] >= 0.5
            
            # 파일 저장
            output_file = self.output_dir / "reliability_summary_table.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"신뢰도 요약 테이블 저장: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"신뢰도 요약 테이블 생성 중 오류: {e}")
            return pd.DataFrame()
    
    def plot_reliability_indicators(self, reliability_results: Dict[str, Any]) -> None:
        """
        신뢰도 지표 비교 차트 생성
        
        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        """
        try:
            reliability_stats = reliability_results['reliability_stats']
            
            # 데이터 준비
            factors = list(reliability_stats.keys())
            alpha_values = [reliability_stats[f].get('cronbach_alpha', np.nan) for f in factors]
            cr_values = [reliability_stats[f].get('composite_reliability', np.nan) for f in factors]
            ave_values = [reliability_stats[f].get('ave', np.nan) for f in factors]
            
            # 그래프 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('신뢰도 및 타당도 지표 비교', fontsize=16, fontweight='bold')
            
            # 1. Cronbach's Alpha
            ax1 = axes[0, 0]
            bars1 = ax1.bar(factors, alpha_values, color='skyblue', alpha=0.7)
            ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='기준값 (0.7)')
            ax1.set_title("Cronbach's Alpha", fontweight='bold')
            ax1.set_ylabel('Alpha 값')
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars1, alpha_values):
                if not np.isnan(value):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # 2. Composite Reliability
            ax2 = axes[0, 1]
            bars2 = ax2.bar(factors, cr_values, color='lightgreen', alpha=0.7)
            ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='기준값 (0.7)')
            ax2.set_title('Composite Reliability (CR)', fontweight='bold')
            ax2.set_ylabel('CR 값')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars2, cr_values):
                if not np.isnan(value):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # 3. AVE
            ax3 = axes[1, 0]
            bars3 = ax3.bar(factors, ave_values, color='lightcoral', alpha=0.7)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='기준값 (0.5)')
            ax3.set_title('Average Variance Extracted (AVE)', fontweight='bold')
            ax3.set_ylabel('AVE 값')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars3, ave_values):
                if not np.isnan(value):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # 4. 종합 비교
            ax4 = axes[1, 1]
            x = np.arange(len(factors))
            width = 0.25
            
            ax4.bar(x - width, alpha_values, width, label="Cronbach's Alpha", alpha=0.7)
            ax4.bar(x, cr_values, width, label='CR', alpha=0.7)
            ax4.bar(x + width, ave_values, width, label='AVE', alpha=0.7)
            
            ax4.set_title('신뢰도 지표 종합 비교', fontweight='bold')
            ax4.set_ylabel('값')
            ax4.set_xticks(x)
            ax4.set_xticklabels(factors, rotation=45)
            ax4.legend()
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # 저장
            output_file = self.output_dir / "reliability_indicators_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"신뢰도 지표 비교 차트 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"신뢰도 지표 차트 생성 중 오류: {e}")
    
    def plot_correlation_heatmap(self, reliability_results: Dict[str, Any]) -> None:
        """
        요인간 상관관계 히트맵 생성
        
        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        """
        try:
            correlations = reliability_results.get('correlations')
            if correlations is None or correlations.empty:
                logger.warning("상관관계 데이터가 없습니다.")
                return
            
            # 히트맵 생성
            plt.figure(figsize=(10, 8))
            
            # 마스크 생성 (상삼각형 숨기기)
            mask = np.triu(np.ones_like(correlations, dtype=bool))
            
            # 히트맵 그리기
            sns.heatmap(correlations, 
                       mask=mask,
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={'label': '상관계수'})
            
            plt.title('요인간 상관관계 매트릭스', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('요인', fontweight='bold')
            plt.ylabel('요인', fontweight='bold')
            
            # 저장
            output_file = self.output_dir / "factor_correlations_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"상관관계 히트맵 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"상관관계 히트맵 생성 중 오류: {e}")

    def plot_factor_loadings(self, reliability_results: Dict[str, Any]) -> None:
        """
        요인별 문항 로딩값 시각화

        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        """
        try:
            reliability_stats = reliability_results['reliability_stats']

            # 요인 수에 따라 subplot 구성 결정
            n_factors = len(reliability_stats)
            if n_factors <= 2:
                fig, axes = plt.subplots(1, n_factors, figsize=(6*n_factors, 6))
            elif n_factors <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            else:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            if n_factors == 1:
                axes = [axes]
            elif n_factors > 1:
                axes = axes.flatten()

            fig.suptitle('요인별 문항 로딩값', fontsize=16, fontweight='bold')

            for idx, (factor_name, stats) in enumerate(reliability_stats.items()):
                if idx >= len(axes):
                    break

                ax = axes[idx]

                items = stats.get('items', [])
                loadings = stats.get('loadings', [])

                if items and loadings and len(items) == len(loadings):
                    # 로딩값에 따라 색상 결정
                    colors = ['red' if abs(loading) < 0.5 else 'orange' if abs(loading) < 0.7 else 'green'
                             for loading in loadings]

                    bars = ax.bar(range(len(items)), loadings, color=colors, alpha=0.7)
                    ax.set_title(f'{factor_name}\n(문항 수: {len(items)})', fontweight='bold')
                    ax.set_xlabel('문항')
                    ax.set_ylabel('로딩값')
                    ax.set_xticks(range(len(items)))
                    ax.set_xticklabels(items, rotation=45)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='기준값 (0.5)')
                    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='권장값 (0.7)')

                    # 값 표시
                    for bar, loading in zip(bars, loadings):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                               f'{loading:.3f}', ha='center', va='bottom', fontsize=8)

                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'{factor_name}\n데이터 없음',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(factor_name, fontweight='bold')

            # 사용하지 않는 subplot 숨기기
            for idx in range(n_factors, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            # 저장
            output_file = self.output_dir / "factor_loadings_by_items.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"요인별 문항 로딩값 차트 저장: {output_file}")

        except Exception as e:
            logger.error(f"요인별 문항 로딩값 시각화 중 오류: {e}")

    def create_loadings_summary_table(self, reliability_results: Dict[str, Any]) -> pd.DataFrame:
        """
        문항별 로딩값 요약 테이블 생성

        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과

        Returns:
            pd.DataFrame: 문항별 로딩값 요약 테이블
        """
        try:
            reliability_stats = reliability_results['reliability_stats']

            # 모든 문항의 로딩값 정보 수집
            loadings_data = []
            for factor_name, stats in reliability_stats.items():
                items = stats.get('items', [])
                loadings = stats.get('loadings', [])

                for item, loading in zip(items, loadings):
                    loadings_data.append({
                        'Factor': factor_name,
                        'Item': item,
                        'Loading': loading,
                        'Abs_Loading': abs(loading),
                        'Loading_Level': 'High (≥0.7)' if abs(loading) >= 0.7 else
                                       'Medium (0.5-0.7)' if abs(loading) >= 0.5 else 'Low (<0.5)',
                        'Acceptable': abs(loading) >= 0.5
                    })

            df = pd.DataFrame(loadings_data)

            # 파일 저장
            output_file = self.output_dir / "factor_loadings_summary.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"문항별 로딩값 요약 테이블 저장: {output_file}")

            return df

        except Exception as e:
            logger.error(f"문항별 로딩값 요약 테이블 생성 중 오류: {e}")
            return pd.DataFrame()

    def plot_discriminant_validity(self, reliability_results: Dict[str, Any]) -> None:
        """
        판별타당도 검증 결과 시각화

        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        """
        try:
            discriminant_validity = reliability_results.get('discriminant_validity', {})
            correlations = reliability_results.get('correlations')
            reliability_stats = reliability_results.get('reliability_stats', {})

            if not discriminant_validity or correlations is None:
                logger.warning("판별타당도 데이터가 없습니다.")
                return

            factors = list(discriminant_validity.keys())
            n_factors = len(factors)

            # 판별타당도 매트릭스 생성
            validity_matrix = np.zeros((n_factors, n_factors))

            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i == j:
                        validity_matrix[i, j] = 1  # 대각선은 항상 유효
                    else:
                        validity_matrix[i, j] = discriminant_validity[factor1].get(factor2, 0)

            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 1. 판별타당도 결과 히트맵
            sns.heatmap(validity_matrix,
                       xticklabels=factors,
                       yticklabels=factors,
                       annot=True,
                       cmap='RdYlGn',
                       center=0.5,
                       cbar_kws={'label': '판별타당도 (1=유효, 0=무효)'},
                       ax=ax1)
            ax1.set_title('판별타당도 검증 결과', fontweight='bold')

            # 2. √AVE vs 상관계수 비교
            ave_sqrt_values = []
            correlation_values = []
            factor_pairs = []

            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i < j:  # 상삼각형만
                        sqrt_ave1 = reliability_stats[factor1].get('sqrt_ave', np.nan)
                        sqrt_ave2 = reliability_stats[factor2].get('sqrt_ave', np.nan)
                        corr = abs(correlations.loc[factor1, factor2])

                        if not (np.isnan(sqrt_ave1) or np.isnan(sqrt_ave2) or np.isnan(corr)):
                            ave_sqrt_values.append(min(sqrt_ave1, sqrt_ave2))
                            correlation_values.append(corr)
                            factor_pairs.append(f"{factor1}-{factor2}")

            if ave_sqrt_values:
                x = np.arange(len(factor_pairs))
                width = 0.35

                bars1 = ax2.bar(x - width/2, ave_sqrt_values, width,
                               label='√AVE (최소값)', alpha=0.7, color='skyblue')
                bars2 = ax2.bar(x + width/2, correlation_values, width,
                               label='상관계수 (절댓값)', alpha=0.7, color='lightcoral')

                ax2.set_xlabel('요인 쌍')
                ax2.set_ylabel('값')
                ax2.set_title('판별타당도: √AVE vs 상관계수', fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(factor_pairs, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 값 표시
                for bar, value in zip(bars1, ave_sqrt_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)

                for bar, value in zip(bars2, correlation_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            # 저장
            output_file = self.output_dir / "discriminant_validity_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"판별타당도 분석 차트 저장: {output_file}")

        except Exception as e:
            logger.error(f"판별타당도 시각화 중 오류: {e}")

    def create_comprehensive_report(self, reliability_results: Dict[str, Any]) -> None:
        """
        종합 신뢰도 분석 보고서 생성

        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        """
        try:
            # 1. 요약 테이블 생성
            summary_table = self.create_reliability_summary_table(reliability_results)

            # 2. 문항별 로딩값 요약 테이블 생성
            loadings_table = self.create_loadings_summary_table(reliability_results)

            # 3. 신뢰도 지표 차트
            self.plot_reliability_indicators(reliability_results)

            # 4. 요인별 문항 로딩값 차트
            self.plot_factor_loadings(reliability_results)

            # 5. 상관관계 히트맵
            self.plot_correlation_heatmap(reliability_results)

            # 6. 판별타당도 분석
            self.plot_discriminant_validity(reliability_results)

            # 7. 텍스트 보고서 생성
            self._generate_text_report(reliability_results, summary_table, loadings_table)

            logger.info("종합 신뢰도 분석 보고서 생성 완료")

        except Exception as e:
            logger.error(f"종합 보고서 생성 중 오류: {e}")

    def _generate_text_report(self, reliability_results: Dict[str, Any],
                             summary_table: pd.DataFrame,
                             loadings_table: pd.DataFrame = None) -> None:
        """
        텍스트 형태의 분석 보고서 생성

        Args:
            reliability_results (Dict[str, Any]): 신뢰도 분석 결과
            summary_table (pd.DataFrame): 요약 테이블
            loadings_table (pd.DataFrame): 문항별 로딩값 테이블
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("신뢰도 및 타당도 분석 보고서")
            report_lines.append("=" * 80)
            report_lines.append("")

            # 분석 정보
            metadata = reliability_results.get('metadata', {})
            report_lines.append(f"분석 일시: {reliability_results.get('analysis_timestamp', 'N/A')}")
            report_lines.append(f"표본 크기: {metadata.get('model_info', {}).get('n_observations', 'N/A')}")
            report_lines.append(f"요인 수: {metadata.get('n_factors', 'N/A')}")
            report_lines.append(f"문항 수: {metadata.get('n_items', 'N/A')}")
            report_lines.append("")

            # 신뢰도 요약
            report_lines.append("1. 신뢰도 지표 요약")
            report_lines.append("-" * 40)

            reliability_stats = reliability_results.get('reliability_stats', {})
            for factor_name, stats in reliability_stats.items():
                report_lines.append(f"\n[{factor_name}]")
                report_lines.append(f"  - 문항 수: {stats.get('n_items', 0)}")
                report_lines.append(f"  - 문항: {', '.join(stats.get('items', []))}")
                report_lines.append(f"  - Cronbach's Alpha: {stats.get('cronbach_alpha', np.nan):.4f}")
                report_lines.append(f"  - Composite Reliability: {stats.get('composite_reliability', np.nan):.4f}")
                report_lines.append(f"  - AVE: {stats.get('ave', np.nan):.4f}")
                report_lines.append(f"  - √AVE: {stats.get('sqrt_ave', np.nan):.4f}")

                # 문항별 로딩값 정보 추가
                items = stats.get('items', [])
                loadings = stats.get('loadings', [])
                if items and loadings and len(items) == len(loadings):
                    report_lines.append(f"  - 문항별 로딩값:")
                    for item, loading in zip(items, loadings):
                        level = "높음" if abs(loading) >= 0.7 else "보통" if abs(loading) >= 0.5 else "낮음"
                        report_lines.append(f"    * {item}: {loading:.4f} ({level})")

                # 수용성 판단
                alpha_ok = stats.get('cronbach_alpha', 0) >= 0.7
                cr_ok = stats.get('composite_reliability', 0) >= 0.7
                ave_ok = stats.get('ave', 0) >= 0.5

                report_lines.append(f"  - 신뢰도 수용성: Alpha({'O' if alpha_ok else 'X'}), CR({'O' if cr_ok else 'X'}), AVE({'O' if ave_ok else 'X'})")

            # 판별타당도 요약
            report_lines.append("\n\n2. 판별타당도 검증")
            report_lines.append("-" * 40)

            discriminant_validity = reliability_results.get('discriminant_validity', {})
            if discriminant_validity:
                valid_pairs = 0
                total_pairs = 0

                factors = list(discriminant_validity.keys())
                for i, factor1 in enumerate(factors):
                    for j, factor2 in enumerate(factors):
                        if i < j:
                            total_pairs += 1
                            if discriminant_validity[factor1].get(factor2, False):
                                valid_pairs += 1

                report_lines.append(f"판별타당도 통과 비율: {valid_pairs}/{total_pairs} ({valid_pairs/total_pairs*100:.1f}%)")

            # 문항별 로딩값 요약 (loadings_table이 있는 경우)
            if loadings_table is not None and not loadings_table.empty:
                report_lines.append("\n\n3. 문항별 로딩값 요약")
                report_lines.append("-" * 40)

                # 로딩값 수준별 통계
                high_loadings = len(loadings_table[loadings_table['Abs_Loading'] >= 0.7])
                medium_loadings = len(loadings_table[(loadings_table['Abs_Loading'] >= 0.5) & (loadings_table['Abs_Loading'] < 0.7)])
                low_loadings = len(loadings_table[loadings_table['Abs_Loading'] < 0.5])
                total_items = len(loadings_table)

                report_lines.append(f"전체 문항 수: {total_items}")
                report_lines.append(f"높은 로딩값 (≥0.7): {high_loadings}개 ({high_loadings/total_items*100:.1f}%)")
                report_lines.append(f"보통 로딩값 (0.5-0.7): {medium_loadings}개 ({medium_loadings/total_items*100:.1f}%)")
                report_lines.append(f"낮은 로딩값 (<0.5): {low_loadings}개 ({low_loadings/total_items*100:.1f}%)")

                # 수용 가능한 문항 비율
                acceptable_items = len(loadings_table[loadings_table['Acceptable'] == True])
                report_lines.append(f"수용 가능한 문항 (≥0.5): {acceptable_items}/{total_items} ({acceptable_items/total_items*100:.1f}%)")

            # 요인간 상관관계 정보 추가
            correlations = reliability_results.get('correlations')
            if correlations is not None and not correlations.empty:
                report_lines.append("\n\n4. 요인간 상관관계 정보")
                report_lines.append("-" * 40)
                report_lines.append("※ 주의: 현재 상관관계는 원본 설문 데이터의 요인 평균점수를 기반으로 별도 계산된 값입니다.")
                report_lines.append("※ 향후 semopy 모델에서 직접 추출한 요인간 상관계수로 대체 예정입니다.")

                # 상관계수 통계
                corr_values = []
                factors = list(correlations.index)
                for i in range(len(factors)):
                    for j in range(i+1, len(factors)):
                        val = correlations.iloc[i, j]
                        if not np.isnan(val):
                            corr_values.append(abs(val))

                if corr_values:
                    report_lines.append(f"평균 상관계수: {np.mean(corr_values):.3f}")
                    report_lines.append(f"최대 상관계수: {np.max(corr_values):.3f}")
                    report_lines.append(f"최소 상관계수: {np.min(corr_values):.3f}")

            # 파일 저장
            output_file = self.output_dir / "reliability_analysis_report.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            logger.info(f"텍스트 보고서 저장: {output_file}")

        except Exception as e:
            logger.error(f"텍스트 보고서 생성 중 오류: {e}")


def visualize_reliability_results(reliability_results: Dict[str, Any],
                                 output_dir: str = "reliability_analysis_results") -> None:
    """
    신뢰도 분석 결과 시각화 편의 함수

    Args:
        reliability_results (Dict[str, Any]): 신뢰도 분석 결과
        output_dir (str): 출력 디렉토리
    """
    visualizer = ReliabilityVisualizer(output_dir)
    visualizer.create_comprehensive_report(reliability_results)
