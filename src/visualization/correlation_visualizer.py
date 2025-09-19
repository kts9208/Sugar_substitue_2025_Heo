#!/usr/bin/env python3
"""
semopy 상관계수 결과 시각화 모듈

이 모듈은 semopy_correlations.py에서 생성된 결과 파일들을 불러와서
다양한 방식으로 시각화합니다.

주요 기능:
1. 결과 파일 로드 (CSV, JSON)
2. 상관계수 히트맵 시각화
3. p값 시각화
4. 통합 시각화 (상관계수 + 유의성)
5. 네트워크 그래프 시각화

특징:
- 기존 모듈과 완전히 독립적
- 높은 재사용성과 확장성
- 간결하면서도 유지보수 용이

Author: Sugar Substitute Research Team
Date: 2025-01-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import warnings

# 영문 폰트 설정 (글꼴 문제 해결)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


class CorrelationResultLoader:
    """semopy 상관계수 결과 파일 로더"""
    
    def __init__(self, results_dir: str = "factor_correlations_results"):
        """
        초기화
        
        Args:
            results_dir: 결과 파일이 저장된 디렉토리
        """
        self.results_dir = Path(results_dir)
        
    def find_latest_files(self) -> Dict[str, Path]:
        """가장 최근 결과 파일들 찾기"""
        if not self.results_dir.exists():
            raise FileNotFoundError(f"결과 디렉토리를 찾을 수 없습니다: {self.results_dir}")
        
        # 패턴별 파일 찾기
        patterns = {
            'correlations': 'semopy_correlations_*.csv',
            'pvalues': 'semopy_pvalues_*.csv',
            'json': 'semopy_results_*.json'
        }
        
        latest_files = {}
        for key, pattern in patterns.items():
            files = list(self.results_dir.glob(pattern))
            if files:
                # 파일명의 타임스탬프로 정렬하여 가장 최근 파일 선택
                latest_file = sorted(files, key=lambda x: x.stem.split('_')[-1])[-1]
                latest_files[key] = latest_file
            else:
                print(f"Warning: {pattern} 패턴의 파일을 찾을 수 없습니다.")
        
        return latest_files
    
    def load_correlation_data(self, correlation_file: Optional[Path] = None, 
                            pvalue_file: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """
        상관계수와 p값 데이터 로드
        
        Args:
            correlation_file: 상관계수 파일 경로 (None이면 자동 탐지)
            pvalue_file: p값 파일 경로 (None이면 자동 탐지)
            
        Returns:
            Dict containing 'correlations' and 'pvalues' DataFrames
        """
        if correlation_file is None or pvalue_file is None:
            latest_files = self.find_latest_files()
            correlation_file = correlation_file or latest_files.get('correlations')
            pvalue_file = pvalue_file or latest_files.get('pvalues')
        
        if not correlation_file or not correlation_file.exists():
            raise FileNotFoundError(f"상관계수 파일을 찾을 수 없습니다: {correlation_file}")
        if not pvalue_file or not pvalue_file.exists():
            raise FileNotFoundError(f"p값 파일을 찾을 수 없습니다: {pvalue_file}")
        
        # 데이터 로드
        correlations = pd.read_csv(correlation_file, index_col=0)
        pvalues = pd.read_csv(pvalue_file, index_col=0)
        
        print(f"✅ 데이터 로드 완료:")
        print(f"  - 상관계수: {correlation_file.name}")
        print(f"  - p값: {pvalue_file.name}")
        print(f"  - 요인 수: {len(correlations)}")
        
        return {
            'correlations': correlations,
            'pvalues': pvalues,
            'correlation_file': correlation_file,
            'pvalue_file': pvalue_file
        }
    
    def load_json_results(self, json_file: Optional[Path] = None) -> Dict:
        """JSON 결과 파일 로드"""
        if json_file is None:
            latest_files = self.find_latest_files()
            json_file = latest_files.get('json')
        
        if not json_file or not json_file.exists():
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON 데이터 로드 완료: {json_file.name}")
        return data


class CorrelationVisualizer:
    """상관계수 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10), style: str = 'whitegrid'):
        """
        초기화
        
        Args:
            figsize: 기본 그래프 크기
            style: seaborn 스타일
        """
        self.figsize = figsize
        sns.set_style(style)
        
        # 영문 요인명 매핑 (글꼴 문제 해결)
        self.factor_labels = {
            'health_concern': 'Health\nConcern',
            'perceived_benefit': 'Perceived\nBenefit',
            'purchase_intention': 'Purchase\nIntention',
            'perceived_price': 'Perceived\nPrice',
            'nutrition_knowledge': 'Nutrition\nKnowledge'
        }
    
    def create_correlation_heatmap(self, correlations: pd.DataFrame, 
                                 pvalues: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None,
                                 show_values: bool = True,
                                 show_significance: bool = True) -> plt.Figure:
        """
        상관계수 히트맵 생성
        
        Args:
            correlations: 상관계수 매트릭스
            pvalues: p값 매트릭스 (유의성 표시용)
            save_path: 저장 경로
            show_values: 수치 표시 여부
            show_significance: 유의성 표시 여부
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # English labels applied
        corr_labeled = correlations.copy()
        corr_labeled.index = [self.factor_labels.get(idx, idx) for idx in correlations.index]
        corr_labeled.columns = [self.factor_labels.get(col, col) for col in correlations.columns]
        
        # 히트맵 생성
        mask = np.triu(np.ones_like(corr_labeled, dtype=bool), k=1)  # 상삼각 마스크

        # 상관계수와 p값을 함께 표시할 어노테이션 생성
        if show_values and pvalues is not None:
            annot_array = self._create_correlation_with_pvalue_annotations(correlations, pvalues)
            annot_labeled = pd.DataFrame(annot_array,
                                       index=[self.factor_labels.get(idx, idx) for idx in correlations.index],
                                       columns=[self.factor_labels.get(col, col) for col in correlations.columns])

            sns.heatmap(corr_labeled,
                       mask=mask,
                       annot=annot_labeled,
                       fmt='',
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       cbar_kws={'label': 'Correlation Coefficient'},
                       ax=ax)
        else:
            sns.heatmap(corr_labeled,
                       mask=mask,
                       annot=show_values,
                       fmt='.3f',
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       cbar_kws={'label': 'Correlation Coefficient'},
                       ax=ax)
        
        # 유의성 표시
        if show_significance and pvalues is not None:
            self._add_significance_markers(ax, correlations, pvalues)
        
        ax.set_title('Factor Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved: {save_path}")
        
        return fig
    
    def create_pvalue_heatmap(self, pvalues: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        p값을 유의성 수준에 따른 색상으로 구분한 히트맵 생성

        Args:
            pvalues: p값 매트릭스
            save_path: 저장 경로

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 영문 라벨 적용
        pval_labeled = pvalues.copy()
        pval_labeled.index = [self.factor_labels.get(idx, idx) for idx in pvalues.index]
        pval_labeled.columns = [self.factor_labels.get(col, col) for col in pvalues.columns]

        # 상삼각 마스크
        mask = np.triu(np.ones_like(pval_labeled, dtype=bool), k=1)

        # p값을 유의성 수준에 따라 카테고리화
        significance_matrix = self._create_significance_matrix(pvalues)
        significance_labeled = pd.DataFrame(significance_matrix,
                                          index=[self.factor_labels.get(idx, idx) for idx in pvalues.index],
                                          columns=[self.factor_labels.get(col, col) for col in pvalues.columns])

        # 유의성 수준별 어노테이션 생성
        annot_matrix = self._create_significance_annotations(pvalues)
        annot_labeled = pd.DataFrame(annot_matrix,
                                   index=[self.factor_labels.get(idx, idx) for idx in pvalues.index],
                                   columns=[self.factor_labels.get(col, col) for col in pvalues.columns])

        # 커스텀 컬러맵 생성 (유의성 수준별)
        from matplotlib.colors import ListedColormap
        colors = ['#f0f0f0', '#ffcccc', '#ff9999', '#ff6666', '#cc0000']  # 회색, 연한빨강 -> 진한빨강
        custom_cmap = ListedColormap(colors)

        sns.heatmap(significance_labeled,
                   mask=mask,
                   annot=annot_labeled,
                   fmt='',
                   cmap=custom_cmap,
                   square=True,
                   cbar_kws={'label': 'Significance Level'},
                   vmin=0, vmax=4,
                   ax=ax)

        ax.set_title('Factor Correlation Significance Levels', fontsize=16, fontweight='bold', pad=20)

        # 컬러바 라벨 수정
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
        cbar.set_ticklabels(['n.s.', 'p<0.05', 'p<0.01', 'p<0.001', 'Diagonal'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 유의성 히트맵 저장: {save_path}")

        return fig

    def _create_significance_matrix(self, pvalues: pd.DataFrame) -> np.ndarray:
        """p값을 유의성 수준에 따라 카테고리화한 매트릭스 생성"""
        significance_matrix = np.zeros(pvalues.shape)

        for i in range(len(pvalues)):
            for j in range(len(pvalues.columns)):
                p_val = pvalues.iloc[i, j]

                if i == j:
                    significance_matrix[i, j] = 4  # 대각선
                elif p_val < 0.001:
                    significance_matrix[i, j] = 3  # p < 0.001
                elif p_val < 0.01:
                    significance_matrix[i, j] = 2  # p < 0.01
                elif p_val < 0.05:
                    significance_matrix[i, j] = 1  # p < 0.05
                else:
                    significance_matrix[i, j] = 0  # not significant

        return significance_matrix

    def _create_significance_annotations(self, pvalues: pd.DataFrame) -> np.ndarray:
        """유의성 수준 어노테이션 생성"""
        annot_matrix = np.empty(pvalues.shape, dtype=object)

        for i in range(len(pvalues)):
            for j in range(len(pvalues.columns)):
                p_val = pvalues.iloc[i, j]

                if i == j:
                    annot_matrix[i, j] = '1.0'
                elif i > j:  # 하삼각만 표시
                    if p_val < 0.001:
                        annot_matrix[i, j] = '***'
                    elif p_val < 0.01:
                        annot_matrix[i, j] = '**'
                    elif p_val < 0.05:
                        annot_matrix[i, j] = '*'
                    else:
                        annot_matrix[i, j] = 'n.s.'
                else:
                    annot_matrix[i, j] = ''

        return annot_matrix
    
    def _add_significance_markers(self, ax, correlations: pd.DataFrame, 
                                pvalues: pd.DataFrame):
        """히트맵에 유의성 마커 추가"""
        for i in range(len(correlations)):
            for j in range(i+1, len(correlations)):
                p_val = pvalues.iloc[i, j]
                
                # 유의성 마커 결정
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    ax.text(j + 0.5, i + 0.7, marker,
                           ha='center', va='center',
                           fontsize=12, fontweight='bold', color='white')

    def _create_correlation_with_pvalue_annotations(self, correlations: pd.DataFrame,
                                                  pvalues: pd.DataFrame) -> np.ndarray:
        """상관계수와 p값을 함께 표시하는 어노테이션 배열 생성"""
        annot_array = np.empty(correlations.shape, dtype=object)

        for i in range(len(correlations)):
            for j in range(len(correlations.columns)):
                corr_val = correlations.iloc[i, j]
                p_val = pvalues.iloc[i, j]

                if i == j:
                    # 대각선 요소
                    annot_array[i, j] = '1.000'
                elif i > j:
                    # 하삼각 요소: 상관계수와 p값 함께 표시
                    if p_val < 0.001:
                        p_text = "p<0.001"
                    elif p_val < 0.01:
                        p_text = "p<0.01"
                    elif p_val < 0.05:
                        p_text = "p<0.05"
                    else:
                        p_text = f"p={p_val:.3f}"

                    annot_array[i, j] = f"{corr_val:.3f}\n{p_text}"
                else:
                    # 상삼각 요소: 마스크 처리됨
                    annot_array[i, j] = ""

        return annot_array

    def create_combined_correlation_plot(self, correlations: pd.DataFrame,
                                       pvalues: pd.DataFrame,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        상관계수와 p값을 동시에 보여주는 결합 플롯 생성

        Args:
            correlations: 상관계수 매트릭스
            pvalues: p값 매트릭스
            save_path: 저장 경로

        Returns:
            matplotlib Figure 객체
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 영문 라벨 적용
        corr_labeled = correlations.copy()
        pval_labeled = pvalues.copy()

        corr_labeled.index = [self.factor_labels.get(idx, idx) for idx in correlations.index]
        corr_labeled.columns = [self.factor_labels.get(col, col) for col in correlations.columns]
        pval_labeled.index = [self.factor_labels.get(idx, idx) for idx in pvalues.index]
        pval_labeled.columns = [self.factor_labels.get(col, col) for col in pvalues.columns]

        # 상삼각 마스크
        mask = np.triu(np.ones_like(corr_labeled, dtype=bool), k=1)

        # 왼쪽: 상관계수 히트맵 (p값 정보 포함)
        annot_array = self._create_correlation_with_pvalue_annotations(correlations, pvalues)
        annot_labeled = pd.DataFrame(annot_array,
                                   index=[self.factor_labels.get(idx, idx) for idx in correlations.index],
                                   columns=[self.factor_labels.get(col, col) for col in correlations.columns])

        sns.heatmap(corr_labeled,
                   mask=mask,
                   annot=annot_labeled,
                   fmt='',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax1)

        ax1.set_title('Correlation Coefficients with P-values', fontsize=14, fontweight='bold', pad=15)

        # 오른쪽: 유의성 수준 히트맵
        significance_matrix = self._create_significance_matrix(pvalues)
        significance_labeled = pd.DataFrame(significance_matrix,
                                          index=[self.factor_labels.get(idx, idx) for idx in pvalues.index],
                                          columns=[self.factor_labels.get(col, col) for col in pvalues.columns])

        annot_matrix = self._create_significance_annotations(pvalues)
        annot_labeled_sig = pd.DataFrame(annot_matrix,
                                       index=[self.factor_labels.get(idx, idx) for idx in pvalues.index],
                                       columns=[self.factor_labels.get(col, col) for col in pvalues.columns])

        from matplotlib.colors import ListedColormap
        colors = ['#f0f0f0', '#ffcccc', '#ff9999', '#ff6666', '#cc0000']
        custom_cmap = ListedColormap(colors)

        sns.heatmap(significance_labeled,
                   mask=mask,
                   annot=annot_labeled_sig,
                   fmt='',
                   cmap=custom_cmap,
                   square=True,
                   cbar_kws={'label': 'Significance Level'},
                   vmin=0, vmax=4,
                   ax=ax2)

        ax2.set_title('Statistical Significance Levels', fontsize=14, fontweight='bold', pad=15)

        # 오른쪽 컬러바 라벨 수정
        cbar2 = ax2.collections[0].colorbar
        cbar2.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
        cbar2.set_ticklabels(['n.s.', 'p<0.05', 'p<0.01', 'p<0.001', 'Diagonal'])

        # 전체 제목
        fig.suptitle('Factor Correlations: Coefficients and Significance',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 결합 플롯 저장: {save_path}")

        return fig

    def create_bubble_plot(self, correlations: pd.DataFrame,
                          pvalues: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        상관계수와 p값을 버블 플롯으로 시각화

        Args:
            correlations: 상관계수 매트릭스
            pvalues: p값 매트릭스
            save_path: 저장 경로

        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 데이터 준비
        factor_names = correlations.index.tolist()
        english_names = [self.factor_labels.get(name, name) for name in factor_names]

        x_coords = []
        y_coords = []
        corr_values = []
        p_values = []

        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                x_coords.append(i)
                y_coords.append(j)
                corr_values.append(correlations.iloc[i, j])
                p_values.append(pvalues.iloc[i, j])

        # 버블 크기 (상관계수 절댓값)
        bubble_sizes = [abs(corr) * 1000 for corr in corr_values]

        # 색상 (유의성 수준 기반)
        colors = []
        for p in p_values:
            if p < 0.001:
                colors.append(3)  # 가장 유의함
            elif p < 0.01:
                colors.append(2)
            elif p < 0.05:
                colors.append(1)
            else:
                colors.append(0)  # 유의하지 않음

        # 버블 플롯 생성
        scatter = ax.scatter(x_coords, y_coords,
                           s=bubble_sizes,
                           c=colors,
                           cmap='Reds',
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=1,
                           vmin=0, vmax=3)

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Significance Level', fontsize=12)
        cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(['n.s.', 'p<0.05', 'p<0.01', 'p<0.001'])

        # 축 설정
        ax.set_xticks(range(len(english_names)))
        ax.set_yticks(range(len(english_names)))
        ax.set_xticklabels(english_names, rotation=45, ha='right')
        ax.set_yticklabels(english_names)

        # 상관계수 값 표시
        for i, (x, y, corr) in enumerate(zip(x_coords, y_coords, corr_values)):
            ax.text(x, y, f'{corr:.3f}',
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='white' if abs(corr) > 0.5 else 'black')

        ax.set_title('Factor Correlations: Bubble Plot\n(Size = |Correlation|, Color = Significance)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🫧 버블 플롯 저장: {save_path}")

        return fig


class NetworkVisualizer:
    """네트워크 그래프 시각화 클래스"""

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """초기화"""
        self.figsize = figsize

        # 영문 요인명 매핑 (글꼴 문제 해결)
        self.factor_labels = {
            'health_concern': 'Health\nConcern',
            'perceived_benefit': 'Perceived\nBenefit',
            'purchase_intention': 'Purchase\nIntention',
            'perceived_price': 'Perceived\nPrice',
            'nutrition_knowledge': 'Nutrition\nKnowledge'
        }

    def create_network_graph(self, correlations: pd.DataFrame,
                           pvalues: pd.DataFrame,
                           threshold: float = 0.1,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        상관관계 네트워크 그래프 생성

        Args:
            correlations: 상관계수 매트릭스
            pvalues: p값 매트릭스
            threshold: 표시할 최소 상관계수 절댓값
            save_path: 저장 경로

        Returns:
            matplotlib Figure 객체
        """
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx가 설치되지 않아 네트워크 그래프를 생성할 수 없습니다.")
            print("설치 방법: pip install networkx")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # 네트워크 그래프 생성
        G = nx.Graph()

        # 노드 추가 (한국어 라벨)
        for factor in correlations.index:
            korean_label = self.factor_labels.get(factor, factor)
            G.add_node(korean_label)

        # 엣지 추가 (유의한 상관관계만)
        edges = []
        edge_weights = []
        edge_colors = []

        for i, factor1 in enumerate(correlations.index):
            for j, factor2 in enumerate(correlations.columns):
                if i < j:  # 상삼각만 처리
                    corr_val = correlations.iloc[i, j]
                    p_val = pvalues.iloc[i, j]

                    if abs(corr_val) >= threshold and p_val < 0.05:
                        korean_label1 = self.factor_labels.get(factor1, factor1)
                        korean_label2 = self.factor_labels.get(factor2, factor2)

                        G.add_edge(korean_label1, korean_label2, weight=abs(corr_val))
                        edges.append((korean_label1, korean_label2))
                        edge_weights.append(abs(corr_val) * 5)  # 시각화용 가중치

                        # 양의 상관관계는 빨간색, 음의 상관관계는 파란색
                        edge_colors.append('red' if corr_val > 0 else 'blue')

        # 레이아웃 설정
        pos = nx.spring_layout(G, k=2, iterations=50)

        # 노드 그리기
        nx.draw_networkx_nodes(G, pos,
                              node_color='lightblue',
                              node_size=3000,
                              alpha=0.7,
                              ax=ax)

        # 엣지 그리기
        if edges:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges,
                                  width=edge_weights,
                                  edge_color=edge_colors,
                                  alpha=0.6,
                                  ax=ax)

        # 라벨 그리기
        nx.draw_networkx_labels(G, pos,
                               font_size=10,
                               font_weight='bold',
                               ax=ax)

        ax.set_title('Factor Correlation Network\n(Significant relationships only, p<0.05)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # 범례 추가
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=3, label='Positive Correlation'),
            Line2D([0], [0], color='blue', lw=3, label='Negative Correlation')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🕸️ 네트워크 그래프 저장: {save_path}")

        return fig


class IntegratedVisualizer:
    """통합 시각화 클래스"""

    def __init__(self):
        """초기화"""
        self.loader = CorrelationResultLoader()
        self.visualizer = CorrelationVisualizer()
        self.network_visualizer = NetworkVisualizer()
    
    def create_comprehensive_report(self, output_dir: str = "correlation_visualization_results") -> Dict[str, str]:
        """
        종합 시각화 보고서 생성
        
        Args:
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 파일들의 경로 딕셔너리
        """
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 데이터 로드
            data = self.loader.load_correlation_data()
            correlations = data['correlations']
            pvalues = data['pvalues']
            
            generated_files = {}
            
            # 1. 상관계수 히트맵
            corr_heatmap_path = output_path / f"correlation_heatmap_{timestamp}.png"
            self.visualizer.create_correlation_heatmap(
                correlations, pvalues, 
                save_path=str(corr_heatmap_path),
                show_significance=True
            )
            generated_files['correlation_heatmap'] = str(corr_heatmap_path)
            
            # 2. p값 히트맵
            pval_heatmap_path = output_path / f"pvalue_heatmap_{timestamp}.png"
            self.visualizer.create_pvalue_heatmap(
                pvalues,
                save_path=str(pval_heatmap_path)
            )
            generated_files['pvalue_heatmap'] = str(pval_heatmap_path)

            # 3. 결합 플롯 (상관계수 + p값)
            combined_path = output_path / f"combined_plot_{timestamp}.png"
            self.visualizer.create_combined_correlation_plot(
                correlations, pvalues,
                save_path=str(combined_path)
            )
            generated_files['combined_plot'] = str(combined_path)

            # 4. 버블 플롯
            bubble_path = output_path / f"bubble_plot_{timestamp}.png"
            self.visualizer.create_bubble_plot(
                correlations, pvalues,
                save_path=str(bubble_path)
            )
            generated_files['bubble_plot'] = str(bubble_path)

            # 5. 네트워크 그래프 (networkx가 설치된 경우)
            network_path = output_path / f"network_graph_{timestamp}.png"
            network_fig = self.network_visualizer.create_network_graph(
                correlations, pvalues,
                threshold=0.1,
                save_path=str(network_path)
            )
            if network_fig is not None:
                generated_files['network_graph'] = str(network_path)
            
            print(f"\n🎨 종합 시각화 보고서 생성 완료!")
            print(f"📂 저장 위치: {output_path}")
            print(f"📊 생성된 파일: {len(generated_files)}개")
            
            return generated_files
            
        except Exception as e:
            print(f"❌ 시각화 생성 중 오류: {e}")
            raise
    
    def show_summary_statistics(self):
        """요약 통계 출력"""
        try:
            data = self.loader.load_correlation_data()
            correlations = data['correlations']
            pvalues = data['pvalues']
            
            print("\n" + "="*60)
            print("📊 상관계수 분석 요약")
            print("="*60)
            
            # 상관계수 통계
            upper_triangle = np.triu(correlations.values, k=1)
            upper_triangle = upper_triangle[upper_triangle != 0]
            
            print(f"\n📈 상관계수 통계:")
            print(f"  - 평균: {upper_triangle.mean():.4f}")
            print(f"  - 최대값: {upper_triangle.max():.4f}")
            print(f"  - 최소값: {upper_triangle.min():.4f}")
            print(f"  - 표준편차: {upper_triangle.std():.4f}")
            
            # 유의한 상관관계 개수
            upper_pvals = np.triu(pvalues.values, k=1)
            upper_pvals = upper_pvals[upper_pvals != 0]
            
            significant_count = (upper_pvals < 0.05).sum()
            total_count = len(upper_pvals)
            
            print(f"\n🎯 유의성 분석:")
            print(f"  - 전체 상관관계: {total_count}개")
            print(f"  - 유의한 관계 (p<0.05): {significant_count}개")
            print(f"  - 유의성 비율: {significant_count/total_count*100:.1f}%")
            
        except Exception as e:
            print(f"❌ 요약 통계 생성 중 오류: {e}")


def main():
    """메인 실행 함수"""
    print("🎨 semopy 상관계수 결과 시각화")
    print("="*50)
    
    try:
        # 통합 시각화 실행
        visualizer = IntegratedVisualizer()
        
        # 요약 통계 출력
        visualizer.show_summary_statistics()
        
        # 종합 보고서 생성
        generated_files = visualizer.create_comprehensive_report()
        
        print(f"\n📁 생성된 시각화 파일:")
        for key, path in generated_files.items():
            print(f"  - {key}: {Path(path).name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 시각화 실행 중 오류: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 상관계수 시각화가 완료되었습니다!")
    else:
        print("\n💥 시각화 실행 중 오류가 발생했습니다.")
