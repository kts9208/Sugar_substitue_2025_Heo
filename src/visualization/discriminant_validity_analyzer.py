"""
판별타당도 검증 모듈 (Discriminant Validity Analyzer)

이 모듈은 상관관계 분석 결과와 신뢰도 분석 결과를 불러와서
요인간 상관계수와 AVE의 제곱근을 비교하여 판별타당도를 검증합니다.

판별타당도 기준:
- Fornell-Larcker 기준: 각 요인의 AVE 제곱근이 다른 요인들과의 상관계수보다 커야 함
- HTMT (Heterotrait-Monotrait) 비율: 0.85 미만이어야 함 (보수적 기준: 0.90)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DiscriminantValidityAnalyzer:
    """판별타당도 검증 분석기"""
    
    def __init__(self, correlation_file=None, reliability_file=None, output_dir="discriminant_validity_results"):
        """
        초기화
        
        Args:
            correlation_file (str): 상관관계 분석 결과 파일 경로
            reliability_file (str): 신뢰도 분석 결과 파일 경로
            output_dir (str): 결과 저장 디렉토리
        """
        self.correlation_file = correlation_file
        self.reliability_file = reliability_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 데이터 저장 변수
        self.correlation_matrix = None
        self.reliability_data = None
        self.ave_sqrt_matrix = None
        self.discriminant_validity_results = {}
        
        # 자동으로 최신 파일 찾기
        if correlation_file is None:
            self.correlation_file = self._find_latest_correlation_file()
        if reliability_file is None:
            self.reliability_file = self._find_latest_reliability_file()
    
    def _find_latest_correlation_file(self):
        """최신 상관관계 분석 결과 파일 찾기"""
        correlation_dir = Path("factor_correlations_results")
        if correlation_dir.exists():
            csv_files = list(correlation_dir.glob("semopy_correlations_*.csv"))
            if csv_files:
                return str(max(csv_files, key=lambda x: x.stat().st_mtime))
        return None
    
    def _find_latest_reliability_file(self):
        """최신 신뢰도 분석 결과 파일 찾기"""
        # 통합 신뢰도 분석 결과 디렉토리들 찾기
        reliability_dirs = [d for d in Path(".").glob("integrated_reliability_results_*") if d.is_dir()]
        if reliability_dirs:
            latest_dir = max(reliability_dirs, key=lambda x: x.stat().st_mtime)
            reliability_file = latest_dir / "reliability_summary_table.csv"
            if reliability_file.exists():
                return str(reliability_file)
        
        # 기본 신뢰도 분석 결과 디렉토리 확인
        reliability_dir = Path("reliability_analysis_results")
        if reliability_dir.exists():
            reliability_file = reliability_dir / "reliability_summary_table.csv"
            if reliability_file.exists():
                return str(reliability_file)
        
        return None
    
    def load_data(self):
        """분석 결과 데이터 로드"""
        print("데이터 로딩 중...")
        
        # 상관관계 데이터 로드
        if self.correlation_file and Path(self.correlation_file).exists():
            print(f"상관관계 데이터 로드: {self.correlation_file}")
            self.correlation_matrix = pd.read_csv(self.correlation_file, index_col=0)
            print(f"상관관계 매트릭스 크기: {self.correlation_matrix.shape}")
        else:
            raise FileNotFoundError(f"상관관계 파일을 찾을 수 없습니다: {self.correlation_file}")
        
        # 신뢰도 데이터 로드
        if self.reliability_file and Path(self.reliability_file).exists():
            print(f"신뢰도 데이터 로드: {self.reliability_file}")
            self.reliability_data = pd.read_csv(self.reliability_file)
            print(f"신뢰도 데이터 크기: {self.reliability_data.shape}")
        else:
            raise FileNotFoundError(f"신뢰도 파일을 찾을 수 없습니다: {self.reliability_file}")
        
        print("데이터 로딩 완료!")
        return True
    
    def create_ave_sqrt_matrix(self):
        """AVE 제곱근 매트릭스 생성"""
        print("AVE 제곱근 매트릭스 생성 중...")
        
        # 요인 순서를 상관관계 매트릭스와 맞춤
        factors = self.correlation_matrix.index.tolist()
        
        # AVE 제곱근 매트릭스 초기화
        self.ave_sqrt_matrix = pd.DataFrame(
            np.zeros((len(factors), len(factors))),
            index=factors,
            columns=factors
        )
        
        # 대각선에 AVE 제곱근 값 설정
        for factor in factors:
            reliability_row = self.reliability_data[self.reliability_data['Factor'] == factor]
            if not reliability_row.empty:
                ave_sqrt = reliability_row['Sqrt_AVE'].iloc[0]
                self.ave_sqrt_matrix.loc[factor, factor] = ave_sqrt
                print(f"{factor}: AVE 제곱근 = {ave_sqrt:.4f}")
        
        print("AVE 제곱근 매트릭스 생성 완료!")
        return self.ave_sqrt_matrix
    
    def analyze_discriminant_validity(self):
        """판별타당도 분석 수행"""
        print("판별타당도 분석 수행 중...")
        
        factors = self.correlation_matrix.index.tolist()
        results = {
            'fornell_larcker_test': {},
            'violations': [],
            'summary': {}
        }
        
        # Fornell-Larcker 기준 검증
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                if i < j:  # 상삼각 매트릭스만 검사
                    correlation = abs(self.correlation_matrix.loc[factor1, factor2])
                    ave_sqrt1 = self.ave_sqrt_matrix.loc[factor1, factor1]
                    ave_sqrt2 = self.ave_sqrt_matrix.loc[factor2, factor2]
                    
                    # 판별타당도 검증: 상관계수 < min(AVE 제곱근)
                    min_ave_sqrt = min(ave_sqrt1, ave_sqrt2)
                    is_valid = correlation < min_ave_sqrt
                    
                    test_result = {
                        'correlation': correlation,
                        'ave_sqrt_1': ave_sqrt1,
                        'ave_sqrt_2': ave_sqrt2,
                        'min_ave_sqrt': min_ave_sqrt,
                        'is_valid': is_valid,
                        'difference': min_ave_sqrt - correlation
                    }
                    
                    results['fornell_larcker_test'][f"{factor1}_vs_{factor2}"] = test_result
                    
                    if not is_valid:
                        results['violations'].append({
                            'factor1': factor1,
                            'factor2': factor2,
                            'correlation': correlation,
                            'min_ave_sqrt': min_ave_sqrt,
                            'violation_magnitude': correlation - min_ave_sqrt
                        })
        
        # 요약 통계
        total_tests = len(results['fornell_larcker_test'])
        valid_tests = sum(1 for test in results['fornell_larcker_test'].values() if test['is_valid'])
        
        results['summary'] = {
            'total_factor_pairs': total_tests,
            'valid_pairs': valid_tests,
            'invalid_pairs': total_tests - valid_tests,
            'validity_rate': valid_tests / total_tests if total_tests > 0 else 0,
            'overall_discriminant_validity': len(results['violations']) == 0
        }
        
        self.discriminant_validity_results = results
        print(f"판별타당도 분석 완료! 전체 {total_tests}개 쌍 중 {valid_tests}개 유효")
        
        return results
    
    def create_comparison_matrix(self):
        """상관계수와 AVE 제곱근 비교 매트릭스 생성"""
        print("비교 매트릭스 생성 중...")
        
        factors = self.correlation_matrix.index.tolist()
        comparison_matrix = pd.DataFrame(
            index=factors,
            columns=factors,
            dtype=object
        )
        
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                if i == j:
                    # 대각선: AVE 제곱근
                    ave_sqrt = self.ave_sqrt_matrix.loc[factor1, factor1]
                    comparison_matrix.loc[factor1, factor2] = f"√AVE: {ave_sqrt:.3f}"
                else:
                    # 비대각선: 상관계수
                    correlation = self.correlation_matrix.loc[factor1, factor2]
                    comparison_matrix.loc[factor1, factor2] = f"r: {correlation:.3f}"
        
        return comparison_matrix

    def visualize_discriminant_validity(self):
        """Discriminant validity verification result visualization"""
        print("Creating discriminant validity visualizations...")

        # 1. Correlation vs AVE square root comparison heatmap
        self._create_comparison_heatmap()

        # 2. Discriminant validity verification result matrix
        self._create_validity_matrix()

        # 3. Violations visualization
        if self.discriminant_validity_results['violations']:
            self._create_violations_plot()

        # 4. Summary dashboard
        self._create_summary_dashboard()

        print("Visualization creation completed!")

    def _create_comparison_heatmap(self):
        """상관계수와 AVE 제곱근 비교 히트맵 생성"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 상관계수 히트맵
        mask_upper = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            mask=mask_upper,
            ax=ax1,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        ax1.set_title('Factor Correlations\n(Lower Triangle)', fontsize=14, fontweight='bold')

        # AVE 제곱근 히트맵 (대각선만)
        ave_diag = self.ave_sqrt_matrix.copy()
        mask_non_diag = ~np.eye(len(ave_diag), dtype=bool)
        sns.heatmap(
            ave_diag,
            annot=True,
            fmt='.3f',
            cmap='Greens',
            mask=mask_non_diag,
            ax=ax2,
            cbar_kws={'label': 'Square Root of AVE'}
        )
        ax2.set_title('Square Root of AVE\n(Diagonal)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_vs_ave_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_validity_matrix(self):
        """판별타당도 검증 결과 매트릭스 시각화"""
        factors = self.correlation_matrix.index.tolist()
        n_factors = len(factors)

        # 검증 결과 매트릭스 생성
        validity_matrix = np.ones((n_factors, n_factors))

        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                if i != j:
                    pair_key = f"{factor1}_vs_{factor2}" if i < j else f"{factor2}_vs_{factor1}"
                    if pair_key in self.discriminant_validity_results['fornell_larcker_test']:
                        is_valid = self.discriminant_validity_results['fornell_larcker_test'][pair_key]['is_valid']
                        validity_matrix[i, j] = 1 if is_valid else 0

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))

        # 커스텀 컬러맵 (빨강: 위반, 초록: 유효)
        colors = ['red', 'lightgreen']
        cmap = plt.matplotlib.colors.ListedColormap(colors)

        im = ax.imshow(validity_matrix, cmap=cmap, aspect='equal')

        # 축 설정
        ax.set_xticks(range(n_factors))
        ax.set_yticks(range(n_factors))
        ax.set_xticklabels(factors, rotation=45, ha='right')
        ax.set_yticklabels(factors)

        # 텍스트 추가
        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    text = "N/A"
                    color = 'black'
                else:
                    text = "Valid" if validity_matrix[i, j] == 1 else "Invalid"
                    color = 'white' if validity_matrix[i, j] == 0 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

        ax.set_title('Discriminant Validity Test Results\n(Fornell-Larcker Criterion)',
                    fontsize=14, fontweight='bold', pad=20)

        # 범례
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Valid (r < √AVE)'),
            Patch(facecolor='red', label='Invalid (r ≥ √AVE)')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'discriminant_validity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_violations_plot(self):
        """판별타당도 위반 사항 시각화"""
        violations = self.discriminant_validity_results['violations']

        if not violations:
            return

        # 위반 데이터 준비
        factor_pairs = [f"{v['factor1']}\nvs\n{v['factor2']}" for v in violations]
        correlations = [v['correlation'] for v in violations]
        min_ave_sqrts = [v['min_ave_sqrt'] for v in violations]
        violation_magnitudes = [v['violation_magnitude'] for v in violations]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 상관계수 vs AVE 제곱근 비교
        x = range(len(factor_pairs))
        width = 0.35

        ax1.bar([i - width/2 for i in x], correlations, width, label='Correlation', color='red', alpha=0.7)
        ax1.bar([i + width/2 for i in x], min_ave_sqrts, width, label='Min √AVE', color='green', alpha=0.7)

        ax1.set_xlabel('Factor Pairs')
        ax1.set_ylabel('Value')
        ax1.set_title('Discriminant Validity Violations\n(Correlation vs Minimum √AVE)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(factor_pairs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 위반 크기
        bars = ax2.bar(x, violation_magnitudes, color='red', alpha=0.7)
        ax2.set_xlabel('Factor Pairs')
        ax2.set_ylabel('Violation Magnitude')
        ax2.set_title('Magnitude of Discriminant Validity Violations', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(factor_pairs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 값 표시
        for i, (bar, mag) in enumerate(zip(bars, violation_magnitudes)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{mag:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'discriminant_validity_violations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_dashboard(self):
        """요약 대시보드 생성"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 전체 판별타당도 결과 파이차트
        summary = self.discriminant_validity_results['summary']
        valid_pairs = summary['valid_pairs']
        invalid_pairs = summary['invalid_pairs']

        labels = ['Valid Pairs', 'Invalid Pairs']
        sizes = [valid_pairs, invalid_pairs]
        colors = ['lightgreen', 'red']
        explode = (0.05, 0.05)

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               explode=explode, shadow=True, startangle=90)
        ax1.set_title('Overall Discriminant Validity Results', fontweight='bold')

        # 2. 요인별 AVE 제곱근 막대그래프
        factors = self.reliability_data['Factor'].tolist()
        ave_sqrts = self.reliability_data['Sqrt_AVE'].tolist()

        bars = ax2.bar(factors, ave_sqrts, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Factors')
        ax2.set_ylabel('Square Root of AVE')
        ax2.set_title('Square Root of AVE by Factor', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 값 표시
        for bar, ave_sqrt in zip(bars, ave_sqrts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ave_sqrt:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. 상관계수 분포 히스토그램
        correlations = []
        for i in range(len(self.correlation_matrix)):
            for j in range(i+1, len(self.correlation_matrix)):
                correlations.append(abs(self.correlation_matrix.iloc[i, j]))

        ax3.hist(correlations, bins=10, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Absolute Correlation Coefficient')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Inter-factor Correlations', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 요약 텍스트
        ax4.axis('off')
        summary_text = f"""
        Discriminant Validity Analysis Summary

        Total Factor Pairs Tested: {summary['total_factor_pairs']}
        Valid Pairs: {summary['valid_pairs']}
        Invalid Pairs: {summary['invalid_pairs']}
        Validity Rate: {summary['validity_rate']:.1%}

        Overall Discriminant Validity: {'ACHIEVED' if summary['overall_discriminant_validity'] else 'NOT ACHIEVED'}

        Fornell-Larcker Criterion:
        - Each factor's √AVE should exceed its correlations with other factors
        - This ensures that each factor shares more variance with its own
          indicators than with other factors

        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'discriminant_validity_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """판별타당도 분석 보고서 생성"""
        print("분석 보고서 생성 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'discriminant_validity_report_{timestamp}.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("판별타당도 분석 보고서 (Discriminant Validity Analysis Report)\n")
            f.write("=" * 80 + "\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"상관관계 데이터: {self.correlation_file}\n")
            f.write(f"신뢰도 데이터: {self.reliability_file}\n")
            f.write("\n")

            # 1. 분석 개요
            f.write("1. 분석 개요\n")
            f.write("-" * 40 + "\n")
            f.write("판별타당도(Discriminant Validity)는 서로 다른 구성개념들이 실제로 구별되는지를\n")
            f.write("검증하는 측정모형의 타당도 평가 기준입니다.\n")
            f.write("\n")
            f.write("검증 기준:\n")
            f.write("- Fornell-Larcker 기준: 각 요인의 AVE 제곱근 > 다른 요인들과의 상관계수\n")
            f.write("- 이는 각 요인이 자신의 측정항목들과 더 많은 분산을 공유함을 의미합니다.\n")
            f.write("\n")

            # 2. 요인별 AVE 제곱근
            f.write("2. 요인별 AVE 제곱근\n")
            f.write("-" * 40 + "\n")
            for _, row in self.reliability_data.iterrows():
                f.write(f"{row['Factor']}: {row['Sqrt_AVE']:.4f}\n")
            f.write("\n")

            # 3. 요인간 상관계수
            f.write("3. 요인간 상관계수 매트릭스\n")
            f.write("-" * 40 + "\n")
            f.write(self.correlation_matrix.to_string(float_format='%.4f'))
            f.write("\n\n")

            # 4. 판별타당도 검증 결과
            f.write("4. 판별타당도 검증 결과 (Fornell-Larcker 기준)\n")
            f.write("-" * 40 + "\n")

            results = self.discriminant_validity_results
            for pair, test_result in results['fornell_larcker_test'].items():
                factor1, factor2 = pair.split('_vs_')
                status = "✓ 유효" if test_result['is_valid'] else "✗ 위반"
                f.write(f"{factor1} vs {factor2}: {status}\n")
                f.write(f"  - 상관계수: {test_result['correlation']:.4f}\n")
                f.write(f"  - 최소 AVE 제곱근: {test_result['min_ave_sqrt']:.4f}\n")
                f.write(f"  - 차이: {test_result['difference']:.4f}\n")
                f.write("\n")

            # 5. 위반 사항 상세
            if results['violations']:
                f.write("5. 판별타당도 위반 사항\n")
                f.write("-" * 40 + "\n")
                for violation in results['violations']:
                    f.write(f"위반: {violation['factor1']} vs {violation['factor2']}\n")
                    f.write(f"  - 상관계수: {violation['correlation']:.4f}\n")
                    f.write(f"  - 최소 AVE 제곱근: {violation['min_ave_sqrt']:.4f}\n")
                    f.write(f"  - 위반 크기: {violation['violation_magnitude']:.4f}\n")
                    f.write("\n")
            else:
                f.write("5. 위반 사항: 없음 (모든 요인 쌍이 판별타당도 기준을 만족)\n")
                f.write("\n")

            # 6. 종합 결과
            f.write("6. 종합 결과\n")
            f.write("-" * 40 + "\n")
            summary = results['summary']
            f.write(f"전체 검증 쌍 수: {summary['total_factor_pairs']}\n")
            f.write(f"유효한 쌍 수: {summary['valid_pairs']}\n")
            f.write(f"위반 쌍 수: {summary['invalid_pairs']}\n")
            f.write(f"유효율: {summary['validity_rate']:.1%}\n")
            f.write(f"전체 판별타당도: {'달성' if summary['overall_discriminant_validity'] else '미달성'}\n")
            f.write("\n")

            # 7. 해석 및 권고사항
            f.write("7. 해석 및 권고사항\n")
            f.write("-" * 40 + "\n")
            if summary['overall_discriminant_validity']:
                f.write("✓ 모든 요인 쌍이 Fornell-Larcker 기준을 만족하여 판별타당도가 확보되었습니다.\n")
                f.write("✓ 각 요인이 다른 요인들과 충분히 구별되는 독립적인 구성개념임이 확인되었습니다.\n")
            else:
                f.write("✗ 일부 요인 쌍이 판별타당도 기준을 위반하였습니다.\n")
                f.write("✗ 해당 요인들 간의 개념적 구별성을 재검토할 필요가 있습니다.\n")
                f.write("\n권고사항:\n")
                f.write("- 위반된 요인들의 측정항목을 재검토하여 개념적 중복을 제거\n")
                f.write("- 요인 구조의 재정의 또는 측정모형의 수정 고려\n")
                f.write("- 추가적인 타당도 검증 (HTMT 비율 등) 실시\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("보고서 생성 완료\n")
            f.write("=" * 80 + "\n")

        print(f"보고서 저장: {report_file}")
        return report_file

    def save_results_to_csv(self):
        """분석 결과를 CSV 파일로 저장"""
        print("결과를 CSV 파일로 저장 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 판별타당도 검증 결과 테이블
        results_data = []
        for pair, test_result in self.discriminant_validity_results['fornell_larcker_test'].items():
            factor1, factor2 = pair.split('_vs_')
            results_data.append({
                'Factor1': factor1,
                'Factor2': factor2,
                'Correlation': test_result['correlation'],
                'AVE_Sqrt_Factor1': test_result['ave_sqrt_1'],
                'AVE_Sqrt_Factor2': test_result['ave_sqrt_2'],
                'Min_AVE_Sqrt': test_result['min_ave_sqrt'],
                'Is_Valid': test_result['is_valid'],
                'Difference': test_result['difference']
            })

        results_df = pd.DataFrame(results_data)
        results_file = self.output_dir / f'discriminant_validity_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')

        # 2. 비교 매트릭스 (상관계수 vs AVE 제곱근)
        comparison_matrix = self.create_comparison_matrix()
        comparison_file = self.output_dir / f'correlation_ave_comparison_matrix_{timestamp}.csv'
        comparison_matrix.to_csv(comparison_file, encoding='utf-8-sig')

        print(f"결과 파일 저장: {results_file}")
        print(f"비교 매트릭스 저장: {comparison_file}")

        return results_file, comparison_file

    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("=" * 60)
        print("판별타당도 분석 시작")
        print("=" * 60)

        try:
            # 1. 데이터 로드
            self.load_data()

            # 2. AVE 제곱근 매트릭스 생성
            self.create_ave_sqrt_matrix()

            # 3. 판별타당도 분석
            self.analyze_discriminant_validity()

            # 4. 시각화 생성
            self.visualize_discriminant_validity()

            # 5. 보고서 생성
            report_file = self.generate_report()

            # 6. CSV 결과 저장
            results_file, comparison_file = self.save_results_to_csv()

            print("=" * 60)
            print("판별타당도 분석 완료!")
            print("=" * 60)
            print(f"결과 디렉토리: {self.output_dir}")
            print(f"보고서: {report_file}")
            print(f"결과 파일: {results_file}")
            print(f"비교 매트릭스: {comparison_file}")

            # 요약 출력
            summary = self.discriminant_validity_results['summary']
            print(f"\n요약:")
            print(f"- 전체 검증 쌍: {summary['total_factor_pairs']}")
            print(f"- 유효한 쌍: {summary['valid_pairs']}")
            print(f"- 위반 쌍: {summary['invalid_pairs']}")
            print(f"- 유효율: {summary['validity_rate']:.1%}")
            print(f"- 전체 판별타당도: {'달성' if summary['overall_discriminant_validity'] else '미달성'}")

            return True

        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """메인 실행 함수"""
    print("판별타당도 분석기 실행")

    # 분석기 초기화 (자동으로 최신 파일 찾기)
    analyzer = DiscriminantValidityAnalyzer()

    # 전체 분석 실행
    success = analyzer.run_complete_analysis()

    if success:
        print("\n분석이 성공적으로 완료되었습니다!")
    else:
        print("\n분석 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()
