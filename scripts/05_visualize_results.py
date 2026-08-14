import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 設定論文風格繪圖樣式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 參數與路徑設定
# ==========================================
DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"

CLEAN_PARQUET = os.path.join(PROCESSED_DIR, f"sf_bay_clean_{DATE_STR}.parquet")
ANOMALY_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_anomalies_{DATE_STR}.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"[*] 讀取清洗後軌跡與異常結果資料...")
start_time = time.time()
df_pts = pd.read_parquet(CLEAN_PARQUET)
df_anom = pd.read_csv(ANOMALY_CSV)

# 將異常標籤與排名關聯回點位資料
df_pts = df_pts.merge(df_anom[['MMSI', 'is_anomaly', 'anomaly_rank', 'anomaly_score']], on='MMSI', how='left')

# ==========================================
# Figure 1: 研究區域整體 AIS 空間分佈圖
# ==========================================
print("[*] 正在繪製 Figure 1: AIS Traffic Distribution...")
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
# 抽樣繪製以加快渲染
sample_pts = df_pts.sample(n=min(50000, len(df_pts)), random_state=42)
sc = ax.scatter(sample_pts['LON'], sample_pts['LAT'], c=sample_pts['SOG'], 
                cmap='viridis', s=0.8, alpha=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Speed Over Ground (SOG, knots)', fontsize=11)
ax.set_title('Figure 1: AIS Vessel Traffic in San Francisco Bay (2023-01-01)', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig1_ais_traffic_distribution.png'))
plt.close(fig)

# ==========================================
# Figure 2: 正常船舶軌跡 (Normal Trajectories)
# ==========================================
print("[*] 正在繪製 Figure 2: Normal Trajectories...")
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
normal_mmsis = df_anom[~df_anom['is_anomaly']]['MMSI'].sample(n=min(25, (~df_anom['is_anomaly']).sum()), random_state=42)

for mmsi in normal_mmsis:
    vessel_data = df_pts[df_pts['MMSI'] == mmsi].sort_values('BaseDateTime')
    ax.plot(vessel_data['LON'], vessel_data['LAT'], alpha=0.6, linewidth=1.2, color='#2b5c8f')

ax.set_title('Figure 2: Representative Normal Vessel Trajectories (Lane Keeping)', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig2_normal_trajectories.png'))
plt.close(fig)

# ==========================================
# Figure 3: Top 5 異常船舶軌跡 (Anomalous Patterns)
# ==========================================
print("[*] 正在繪製 Figure 3: Top Anomalous Trajectories...")
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

# 先用淡灰色畫出背景整體航道作為對比
bg_sample = df_pts.sample(n=min(15000, len(df_pts)), random_state=42)
ax.scatter(bg_sample['LON'], bg_sample['LAT'], color='lightgray', s=0.3, alpha=0.3, label='Background Traffic')

colors = ['#d95f02', '#e7298a', '#7570b3', '#e6ab02', '#1b9e77']
top5_vessels = df_anom.head(5)

for idx, (_, row) in enumerate(top5_vessels.iterrows()):
    mmsi = int(row['MMSI'])
    rank = int(row['anomaly_rank'])
    v_pts = df_pts[df_pts['MMSI'] == mmsi].sort_values('BaseDateTime')
    ax.plot(v_pts['LON'], v_pts['LAT'], linewidth=2.0, color=colors[idx],
            label=f"Rank {rank} (MMSI: {mmsi}, Score: {row['anomaly_score']:.2f})")

ax.set_title('Figure 3: Top 5 Anomalous Vessel Navigation Trajectories', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
ax.legend(loc='upper right', frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig3_anomalous_trajectories.png'))
plt.close(fig)

# ==========================================
# Figure 4: 正常 vs 異常特徵分佈比較圖
# ==========================================
print("[*] 正在繪製 Figure 4: Feature Distributions Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

features_to_plot = [
    ('mean_sog', 'Mean SOG (knots)', axes[0, 0]),
    ('cog_std', 'COG Std Dev (degrees)', axes[0, 1]),
    ('stop_ratio', 'Stop Ratio (0~1)', axes[1, 0]),
    ('total_dist_nmi', 'Total Distance (nmi)', axes[1, 1])
]

for col, label, ax in features_to_plot:
    sns.boxplot(data=df_anom, x='is_anomaly', y=col, ax=ax, palette=['#4575b4', '#d73027'], width=0.4)
    ax.set_xticklabels(['Normal', 'Anomalous'], fontsize=11, fontweight='bold')
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel('')

fig.suptitle('Figure 4: Feature Distribution Comparison (Normal vs. Anomalous Vessels)', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig4_feature_comparison.png'))
plt.close(fig)

print(f"[+] 視覺化完成！耗時: {time.time() - start_time:.2f} 秒")
print("=" * 60)
print(f"所有圖表已存入: {FIGURES_DIR}/")
print("1. fig1_ais_traffic_distribution.png")
print("2. fig2_normal_trajectories.png")
print("3. fig3_anomalous_trajectories.png")
print("4. fig4_feature_comparison.png")
print("=" * 60)