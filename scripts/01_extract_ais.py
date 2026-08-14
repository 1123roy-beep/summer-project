import os
import urllib.request
import zipfile
import duckdb
import time
#%%
# ==========================================
# 1. 參數設定
# ==========================================
DATE_STR = "2023_01_01"
URL = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_{DATE_STR}.zip"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
ZIP_PATH = os.path.join(RAW_DIR, f"AIS_{DATE_STR}.zip")
CSV_PATH = os.path.join(RAW_DIR, f"AIS_{DATE_STR}.csv")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, f"sf_bay_ais_{DATE_STR}.parquet")

# 舊金山灣 (San Francisco Bay) Bounding Box
LAT_MIN, LAT_MAX = 37.45, 38.15
LON_MIN, LON_MAX = -122.65, -122.05

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
#%%
# ==========================================
# 2. 下載 (若本機已存在 ZIP 則自動跳過)
# ==========================================
if not os.path.exists(ZIP_PATH):
    print(f"[*] 正在下載 {ZIP_PATH}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
        f.write(resp.read())
    print("[+] 下載完成！")
else:
    print(f"[i] 原始 ZIP 檔案已存在: {ZIP_PATH}")

# ==========================================
# 3. 解壓縮 CSV
# ==========================================
if not os.path.exists(CSV_PATH):
    print(f"[*] 正在從 {ZIP_PATH} 解壓 CSV 檔案...")
    start_unzip = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("ZIP 內未發現 CSV 檔")
        zip_ref.extract(csv_files[0], RAW_DIR)
        extracted_file = os.path.join(RAW_DIR, csv_files[0])
        if extracted_file != CSV_PATH:
            os.rename(extracted_file, CSV_PATH)
    print(f"[+] 解壓完成！耗時: {time.time() - start_unzip:.1f} 秒")
else:
    print(f"[i] CSV 檔案已存在: {CSV_PATH}")

# ==========================================
# 4. DuckDB 空間過濾並轉存 Parquet
# ==========================================
print("[*] 正在透過 DuckDB 進行空間過濾 (San Francisco Bay)...")
start_query = time.time()

con = duckdb.connect()

query = f"""
COPY (
    SELECT 
        MMSI,
        BaseDateTime,
        LAT,
        LON,
        SOG,
        COG
    FROM read_csv_auto('{CSV_PATH}')
    WHERE 
        LAT BETWEEN {LAT_MIN} AND {LAT_MAX}
        AND LON BETWEEN {LON_MIN} AND {LON_MAX}
        AND MMSI IS NOT NULL
        AND LAT IS NOT NULL 
        AND LON IS NOT NULL
        AND SOG >= 0 AND SOG <= 60
        AND COG >= 0 AND COG <= 360
) TO '{OUTPUT_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""

con.execute(query)
print(f"[+] DuckDB 處理完成！耗時: {time.time() - start_query:.1f} 秒")

# ==========================================
# 5. 驗證產出結果
# ==========================================
result = con.execute(f"SELECT count(*), count(distinct MMSI) FROM '{OUTPUT_PARQUET}'").fetchall()
total_records, unique_vessels = result[0]
file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)

print("=" * 45)
print(f"產出檔案: {OUTPUT_PARQUET}")
print(f"檔案大小: {file_size_mb:.2f} MB")
print(f"灣區軌跡點數: {total_records:,} 筆")
print(f"灣區船舶總數: {unique_vessels:,} 艘 (Unique MMSI)")
print("=" * 45)
# %%
