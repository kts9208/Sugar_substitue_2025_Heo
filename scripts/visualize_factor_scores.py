"""
잠재변수 점수 시각화

히스토그램과 분포 플롯을 생성합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 최신 파일 찾기
log_dir = Path('logs/factor_scores')
files = sorted(log_dir.glob('factor_scores_SEM_추출_직후_*.csv'))
if not files:
    print("파일을 찾을 수 없습니다!")
    exit(1)

latest_file = files[-1]
print(f"분석 파일: {latest_file.name}")
print()

# 데이터 로드
df = pd.read_csv(latest_file)

# 5개 잠재변수
latent_vars = df.columns.tolist()

# 1. 히스토그램 + 정규분포 곡선
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(latent_vars):
    ax = axes[i]
    values = df[var].values
    
    # 히스토그램
    ax.hist(values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 정규분포 곡선
    mu, sigma = np.mean(values), np.std(values)
    x = np.linspace(values.min(), values.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}²)')
    
    # 통계량 표시
    ax.axvline(mu, color='red', linestyle='--', linewidth=1, label=f'Mean: {mu:.4f}')
    ax.axvline(mu + sigma, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(mu - sigma, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_title(f'{var}\n(Var={np.var(values, ddof=1):.4f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Factor Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 마지막 subplot 제거
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('logs/factor_scores_histograms.png', dpi=300, bbox_inches='tight')
print("히스토그램 저장: logs/factor_scores_histograms.png")

# 2. Q-Q Plot (정규성 검정)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(latent_vars):
    ax = axes[i]
    values = df[var].values
    
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f'{var} Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('logs/factor_scores_qqplots.png', dpi=300, bbox_inches='tight')
print("Q-Q Plot 저장: logs/factor_scores_qqplots.png")

# 3. 박스플롯
fig, ax = plt.subplots(figsize=(12, 6))

df_melted = df.melt(var_name='Latent Variable', value_name='Factor Score')
sns.boxplot(data=df_melted, x='Latent Variable', y='Factor Score', ax=ax)
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title('잠재변수 점수 분포 비교', fontsize=14, fontweight='bold')
ax.set_xlabel('잠재변수', fontsize=12)
ax.set_ylabel('요인점수', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('logs/factor_scores_boxplot.png', dpi=300, bbox_inches='tight')
print("박스플롯 저장: logs/factor_scores_boxplot.png")

# 4. 상관관계 히트맵
fig, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('잠재변수 간 상관관계', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('logs/factor_scores_correlation.png', dpi=300, bbox_inches='tight')
print("상관관계 히트맵 저장: logs/factor_scores_correlation.png")

print()
print("모든 시각화 완료!")

