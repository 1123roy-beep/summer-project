import os
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 參數設定
# ==========================================
DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
INPUT_PARQUET = os.path.join(PROCESSED_DIR, f"sf_bay_clean_{DATE_STR}.parquet")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_features_{DATE_STR}.csv")

print(f"[*] 讀取清洗後資料: {INPUT_PARQUET}")
start_time = time.time()
df = pd.read_parquet(INPUT_PARQUET)
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

# ==========================================
# 2. 輔助函式：計算兩點距離與圓形標準差
# ==========================================
def haversine_distance_nmi(lat1, lon1, lat2, lon2):
    """計算經緯度距離 (海浬)"""
    R = 3440.065  # 地球半徑 (海浬)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def circular_std_deg(cog_series):
    """計算航向角度的圓形標準差 (Circular Standard Deviation)"""
    rad = np.radians(cog_series.dropna())
    if len(rad) == 0:
        return 0.0
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    R = np.clip(R, 1e-6, 1.0)
    circ_std_rad = np.sqrt(-2.0 * np.log(R))
    return np.degrees(circ_std_rad)

# ==========================================
# 3. 逐船計算特徵向量 (Per-vessel Aggregation)
# ==========================================
print("[*] 正在計算每艘船舶的航行特徵向量...")

# 計算相鄰點距離
df['prev_lat'] = df.groupby('MMSI')['LAT'].shift(1)
df['prev_lon'] = df.groupby('MMSI')['LON'].shift(1)
df['segment_dist_nmi'] = haversine_distance_nmi(df['prev_lat'], df['prev_lon'], df['LAT'], df['LON']).fillna(0)

records = []
for mmsi, group in df.groupby('MMSI'):
    point_count = len(group)
    duration_hours = (group['BaseDateTime'].max() - group['BaseDateTime'].min()).total_seconds() / 3600.0
    
    mean_sog = group['SOG'].mean()
    max_sog = group['SOG'].max()
    sog_std = group['SOG'].std(ddof=0)
    cog_std = circular_std_deg(group['COG'])
    
    # 停滯時間比例 (速度 < 0.5 節視為靜止/錨泊/靠泊)
    stop_ratio = (group['SOG'] < 0.5).mean()
    
    # 累積總航程
    total_dist_nmi = group['segment_dist_nmi'].sum()
    
    records.append({
        'MMSI': mmsi,
        'point_count': point_count,
        'duration_hours': round(duration_hours, 2),
        'mean_sog': round(mean_sog, 2),
        'max_sog': round(max_sog, 2),
        'sog_std': round(sog_std, 2),
        'cog_std': round(cog_std, 2),
        'stop_ratio': round(stop_ratio, 2),
        'total_dist_nmi': round(total_dist_nmi, 2)
    })

feature_df = pd.DataFrame(records)

# ==========================================
# 4. 輸出特徵表與基本統計
# ==========================================
feature_df.to_csv(OUTPUT_CSV, index=False)

print(f"[+] 特徵工程完成！耗時: {time.time() - start_time:.2f} 秒")
print("=" * 45)
print(f"特徵表輸出路徑: {OUTPUT_CSV}")
print(f"提取船舶總數: {len(feature_df)} 艘")
print("\n特徵摘要預覽 (Top 5):")
print(feature_df[['MMSI', 'mean_sog', 'max_sog', 'sog_std', 'cog_std', 'stop_ratio', 'total_dist_nmi']].head())
print("=" * 45)