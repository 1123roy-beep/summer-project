import os
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 參數設定
# ==========================================
DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
INPUT_PARQUET = os.path.join(PROCESSED_DIR, f"sf_bay_ais_{DATE_STR}.parquet")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, f"sf_bay_clean_{DATE_STR}.parquet")

MIN_POINTS_PER_VESSEL = 20  # 至少需要 20 個點才視為有效軌跡
MAX_REALISTIC_SPEED = 50.0  # 節 (knots)，超過此速度視為 GPS 飄移/跳點

print(f"[*] 讀取資料: {INPUT_PARQUET}")
start_time = time.time()
df = pd.read_parquet(INPUT_PARQUET)
initial_rows = len(df)
initial_vessels = df['MMSI'].nunique()

# ==========================================
# 2. 時間轉換與排序 (Trajectory Ordering)
# ==========================================
print("[*] 正在進行時間排序與去重...")
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

# 依 MMSI 與時間嚴格排序
df = df.sort_values(by=['MMSI', 'BaseDateTime'])

# 移除同 MMSI、同時間的重複點
df = df.drop_duplicates(subset=['MMSI', 'BaseDateTime'])

# ==========================================
# 3. 軌跡跳點清洗 (Haversine 距離與跳點速度過濾)
# ==========================================
print("[*] 正在計算相鄰點隱含速度並剔除跳點...")

def haversine_distance_nmi(lat1, lon1, lat2, lon2):
    """計算經緯度距離並轉換為海浬 (Nautical Miles)"""
    R = 3440.065  # 地球半徑 (海浬)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 計算同一艘船的相鄰點時間差 (小時) 與距離差 (海浬)
df['prev_lat'] = df.groupby('MMSI')['LAT'].shift(1)
df['prev_lon'] = df.groupby('MMSI')['LON'].shift(1)
df['time_diff_hours'] = df.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds() / 3600.0

dist_nmi = haversine_distance_nmi(df['prev_lat'], df['prev_lon'], df['LAT'], df['LON'])
df['calculated_speed_knots'] = dist_nmi / df['time_diff_hours']

# 第一個點沒有前一點資訊，填 0
df['calculated_speed_knots'] = df['calculated_speed_knots'].fillna(0)

# 過濾掉瞬移點 (相鄰點隱含速度 > MAX_REALISTIC_SPEED 且時間差小於 1 小時)
valid_speed_mask = (df['time_diff_hours'].isna()) | (df['calculated_speed_knots'] <= MAX_REALISTIC_SPEED) | (df['time_diff_hours'] > 1.0)
df = df[valid_speed_mask].copy()

# 清理計算用的中介欄位
df = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]

# ==========================================
# 4. 過濾點數過少的船舶 (Fragmented Trajectories)
# ==========================================
vessel_counts = df['MMSI'].value_counts()
valid_mmsis = vessel_counts[vessel_counts >= MIN_POINTS_PER_VESSEL].index
df = df[df['MMSI'].isin(valid_mmsis)].copy()

# ==========================================
# 5. 儲存清洗後的資料
# ==========================================
df.to_parquet(OUTPUT_PARQUET, index=False, compression='zstd')
cleaned_rows = len(df)
cleaned_vessels = df['MMSI'].nunique()

print(f"[+] 清洗完成！耗時: {time.time() - start_time:.2f} 秒")
print("=" * 45)
print(f"清洗前: {initial_rows:,} 筆點位 | {initial_vessels} 艘船")
print(f"清洗後: {cleaned_rows:,} 筆點位 | {cleaned_vessels} 艘船")
print(f"剔除點位: {initial_rows - cleaned_rows:,} 筆 ({(initial_rows - cleaned_rows)/initial_rows*100:.1f}%)")
print(f"輸出檔案: {OUTPUT_PARQUET}")
print("=" * 45)