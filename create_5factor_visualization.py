#!/usr/bin/env python3
"""
5개 요인 경로분석 시각화 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from path_analysis import (
    PathAnalyzer,
    create_default_path_config,
    create_path_model
)

def create_comprehensive_visualization():
    """5개 요인 종합 시각화"""
    print("🎨 5개 요인 경로분석 시각화 생성")
    print("=" * 50)
    
    try:
        # 1. 모델 설정 및 분석
        variables = ['health_concern', 'perceived_benefit', 'perceived_price', 
                    'nutrition_knowledge', 'purchase_intention']
        
        paths = [
            ('health_concern', 'perceived_benefit'),
            ('health_concern', 'perceived_price'),
            ('health_concern', 'nutrition_knowledge'),
            ('nutrition_knowledge', 'perceived_benefit'),
            ('perceived_benefit', 'purchase_intention'),
            ('perceived_price', 'purchase_intention'),
            ('nutrition_knowledge', 'purchase_intention'),
            ('health_concern', 'purchase_intention')
        ]
        
        correlations = [
            ('perceived_benefit', 'perceived_price'),
            ('perceived_benefit', 'nutrition_knowledge')
        ]
        
        model_spec = create_path_model(
            model_type='custom',
            variables=variables,
            paths=paths,
            correlations=correlations
        )
        
        config = create_default_path_config(verbose=False)
        analyzer = PathAnalyzer(config)
        data = analyzer.load_data(variables)
        results = analyzer.fit_model(model_spec, data)
        
        print(f"모델 분석 완료: {results['model_info']['n_observations']}개 관측치")
        
        # 2. 출력 디렉토리 설정
        output_dir = Path("path_analysis_results/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 3. 적합도 지수 시각화
        create_fit_indices_chart(results, output_dir, timestamp)
        
        # 4. 구조적 경로계수 시각화
        create_structural_paths_chart(results, variables, output_dir, timestamp)
        
        # 5. 구매의도 영향요인 차트
        create_purchase_intention_effects_chart(results, output_dir, timestamp)
        
        # 6. 종합 대시보드
        create_comprehensive_dashboard(results, variables, output_dir, timestamp)
        
        print("✅ 모든 시각화 완료!")
        
    except Exception as e:
        print(f"❌ 시각화 생성 오류: {e}")
        import traceback
        traceback.print_exc()

def create_fit_indices_chart(results, output_dir, timestamp):
    """적합도 지수 차트"""
    try:
        if 'fit_indices' not in results or not results['fit_indices']:
            return
        
        # 적합도 지수 데이터 준비
        fit_data = []
        for index, value in results['fit_indices'].items():
            if hasattr(value, 'iloc'):
                numeric_value = value.iloc[0] if len(value) > 0 else np.nan
            else:
                numeric_value = value
            
            if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                fit_data.append({'Index': index.upper(), 'Value': numeric_value})
        
        if not fit_data:
            return
        
        df = pd.DataFrame(fit_data)
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['Index'], df['Value'], color='skyblue', alpha=0.7)
        
        # 기준선 추가
        benchmarks = {
            'CFI': 0.90, 'TLI': 0.90, 'RMSEA': 0.08
        }
        
        for i, (index, value) in enumerate(zip(df['Index'], df['Value'])):
            if index in benchmarks:
                ax.axhline(y=benchmarks[index], color='red', linestyle='--', alpha=0.5)
                ax.text(i, benchmarks[index], f'Benchmark: {benchmarks[index]}', 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Model Fit Indices', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Fit Index', fontsize=12)
        
        # 값 표시
        for bar, value in zip(bars, df['Value']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        chart_file = output_dir / f"5factor_fit_indices_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 적합도 지수 차트: {chart_file}")
        
    except Exception as e:
        print(f"❌ 적합도 차트 오류: {e}")

def create_structural_paths_chart(results, variables, output_dir, timestamp):
    """구조적 경로계수 차트"""
    try:
        if 'path_coefficients' not in results or not results['path_coefficients']:
            return
        
        path_coeffs = results['path_coefficients']
        if 'paths' not in path_coeffs or not path_coeffs['paths']:
            return
        
        # 구조적 경로만 추출
        structural_data = []
        for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
            if from_var in variables and to_var in variables:  # 잠재변수 간 경로만
                coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                structural_data.append({
                    'From': from_var.replace('_', ' ').title(),
                    'To': to_var.replace('_', ' ').title(),
                    'Path': f"{from_var.replace('_', ' ').title()}\n→ {to_var.replace('_', ' ').title()}",
                    'Coefficient': coeff
                })
        
        if not structural_data:
            return
        
        df = pd.DataFrame(structural_data)
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 색상 설정 (양수: 파랑, 음수: 빨강)
        colors = ['steelblue' if coeff >= 0 else 'indianred' for coeff in df['Coefficient']]
        
        bars = ax.barh(df['Path'], df['Coefficient'], color=colors, alpha=0.7)
        
        ax.set_title('Structural Path Coefficients', fontsize=16, fontweight='bold')
        ax.set_xlabel('Path Coefficient', fontsize=12)
        ax.set_ylabel('Structural Path', fontsize=12)
        
        # 0선 추가
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bar, coeff in zip(bars, df['Coefficient']):
            ax.text(coeff + (0.02 if coeff >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                   f'{coeff:.3f}', ha='left' if coeff >= 0 else 'right', va='center', fontsize=10)
        
        plt.tight_layout()
        chart_file = output_dir / f"5factor_structural_paths_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 구조적 경로 차트: {chart_file}")
        
    except Exception as e:
        print(f"❌ 구조적 경로 차트 오류: {e}")

def create_purchase_intention_effects_chart(results, output_dir, timestamp):
    """구매의도 영향요인 차트"""
    try:
        if 'path_coefficients' not in results or not results['path_coefficients']:
            return
        
        path_coeffs = results['path_coefficients']
        if 'paths' not in path_coeffs or not path_coeffs['paths']:
            return
        
        # 구매의도에 대한 직접효과만 추출
        effects_data = []
        for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
            if to_var == 'purchase_intention':
                coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                effects_data.append({
                    'Factor': from_var.replace('_', ' ').title(),
                    'Effect': coeff
                })
        
        if not effects_data:
            return
        
        df = pd.DataFrame(effects_data).sort_values('Effect', key=abs, ascending=True)
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['forestgreen' if eff >= 0 else 'crimson' for eff in df['Effect']]
        bars = ax.barh(df['Factor'], df['Effect'], color=colors, alpha=0.7)
        
        ax.set_title('Direct Effects on Purchase Intention', fontsize=16, fontweight='bold')
        ax.set_xlabel('Path Coefficient', fontsize=12)
        ax.set_ylabel('Factors', fontsize=12)
        
        # 0선 추가
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bar, effect in zip(bars, df['Effect']):
            ax.text(effect + (0.02 if effect >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                   f'{effect:.3f}', ha='left' if effect >= 0 else 'right', va='center', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        chart_file = output_dir / f"5factor_purchase_effects_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 구매의도 영향요인 차트: {chart_file}")
        
    except Exception as e:
        print(f"❌ 구매의도 차트 오류: {e}")

def create_comprehensive_dashboard(results, variables, output_dir, timestamp):
    """종합 대시보드"""
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # 2x2 서브플롯 생성
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 적합도 지수 (좌상)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'fit_indices' in results and results['fit_indices']:
            fit_data = []
            for index, value in results['fit_indices'].items():
                if hasattr(value, 'iloc'):
                    numeric_value = value.iloc[0] if len(value) > 0 else np.nan
                else:
                    numeric_value = value
                
                if isinstance(numeric_value, (int, float)) and not pd.isna(numeric_value):
                    fit_data.append({'Index': index.upper(), 'Value': numeric_value})
            
            if fit_data:
                df_fit = pd.DataFrame(fit_data)
                bars = ax1.bar(df_fit['Index'], df_fit['Value'], color='lightblue', alpha=0.7)
                ax1.set_title('Model Fit Indices', fontweight='bold')
                ax1.set_ylabel('Value')
                
                for bar, value in zip(bars, df_fit['Value']):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 구매의도 영향요인 (우상)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'path_coefficients' in results and results['path_coefficients']:
            path_coeffs = results['path_coefficients']
            effects_data = []
            for i, (from_var, to_var) in enumerate(path_coeffs['paths']):
                if to_var == 'purchase_intention':
                    coeff = path_coeffs.get('coefficients', {}).get(i, 0)
                    effects_data.append({
                        'Factor': from_var.replace('_', '\n').title(),
                        'Effect': coeff
                    })
            
            if effects_data:
                df_effects = pd.DataFrame(effects_data)
                colors = ['green' if eff >= 0 else 'red' for eff in df_effects['Effect']]
                bars = ax2.bar(df_effects['Factor'], df_effects['Effect'], color=colors, alpha=0.7)
                ax2.set_title('Effects on Purchase Intention', fontweight='bold')
                ax2.set_ylabel('Path Coefficient')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                for bar, effect in zip(bars, df_effects['Effect']):
                    ax2.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (0.02 if effect >= 0 else -0.02),
                           f'{effect:.3f}', ha='center', 
                           va='bottom' if effect >= 0 else 'top', fontsize=9)
        
        # 3. 모델 정보 (좌하)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        model_info = [
            f"Sample Size: {results['model_info']['n_observations']}",
            f"Variables: {results['model_info']['n_variables']}",
            f"Estimator: {results['model_info'].get('estimator', 'MLW')}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}"
        ]
        
        for i, info in enumerate(model_info):
            ax3.text(0.1, 0.8 - i*0.15, info, fontsize=12, transform=ax3.transAxes)
        
        ax3.set_title('Model Information', fontweight='bold')
        
        # 4. 변수 정보 (우하)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        var_info = [
            "Variables in Model:",
            "• Health Concern",
            "• Perceived Benefit", 
            "• Perceived Price",
            "• Nutrition Knowledge",
            "• Purchase Intention"
        ]
        
        for i, info in enumerate(var_info):
            weight = 'bold' if i == 0 else 'normal'
            ax4.text(0.1, 0.9 - i*0.12, info, fontsize=11, fontweight=weight, transform=ax4.transAxes)
        
        # 전체 제목
        fig.suptitle('5-Factor Path Analysis Dashboard', fontsize=18, fontweight='bold', y=0.95)
        
        # 저장
        dashboard_file = output_dir / f"5factor_dashboard_{timestamp}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 종합 대시보드: {dashboard_file}")
        
    except Exception as e:
        print(f"❌ 대시보드 생성 오류: {e}")

if __name__ == "__main__":
    create_comprehensive_visualization()
