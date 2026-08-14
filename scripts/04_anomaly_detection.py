import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

# ==========================================
# 1. 參數設定
# ==========================================
DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
INPUT_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_features_{DATE_STR}.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_anomalies_{DATE_STR}.csv")

# 預估異常比例 (Contamination: 抓出前 5% 最特殊的航行型態)
CONTAMINATION = 0.05

print(f"[*] 讀取特徵表: {INPUT_CSV}")
start_time = time.time()
df = pd.read_csv(INPUT_CSV)

# ==========================================
# 2. 選取特徵與標準化 (Feature Scaling)
# ==========================================
feature_cols = [
    'mean_sog',
    'sog_std',
    'cog_std',
    'stop_ratio',
    'total_dist_nmi'
]

X = df[feature_cols].copy()

# 標準化: 使各特徵均值為 0，標準差為 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. 訓練 Isolation Forest 模型
# ==========================================
print("[*] 正在訓練 Isolation Forest 異常偵測模型...")

# random_state=42 確保每次執行的實驗結果完全一致 (可重現性)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION,
    random_state=42
)

# 預測: 1 為常態，-1 為異常
df['prediction'] = iso_forest.fit_predict(X_scaled)
df['is_anomaly'] = df['prediction'].apply(lambda x: True if x == -1 else False)

# 計算可解釋的異常評分 (轉換至 0 ~ 1，越接近 1 代表越偏離常態)
raw_scores = iso_forest.score_samples(X_scaled)
df['anomaly_score'] = np.round(1 - ((raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())), 4)

# 依異常分數由高至低排名
df = df.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
df['anomaly_rank'] = df.index + 1

# ==========================================
# 4. 儲存結果與印出 Top 5
# ==========================================
df.to_csv(OUTPUT_CSV, index=False)

print(f"[+] 異常偵測完成！耗時: {time.time() - start_time:.2f} 秒")
print("=" * 65)
print(f"分析船舶總數: {len(df)} 艘")
print(f"標記為異常船舶 (Top {int(CONTAMINATION*100)}%): {df['is_anomaly'].sum()} 艘")
print(f"輸出成果檔案: {OUTPUT_CSV}")
print("=" * 65)

print("\n【Top 5 異常航行型態船舶排行】:")
top5 = df[['anomaly_rank', 'MMSI', 'mean_sog', 'cog_std', 'stop_ratio', 'total_dist_nmi', 'anomaly_score']].head(5)
print(top5.to_string(index=False))
print("=" * 65)