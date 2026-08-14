import os
import json
import urllib.request
import pandas as pd
import time

# ==========================================
# 1. 參數與路徑設定
# ==========================================
DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"
ANOMALY_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_anomalies_{DATE_STR}.csv")
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "agent_analysis_summary.md")

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
TOP_N = 3  # 預設自動分析異常排名前 3 名的船舶

os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================================
# 2. 讀取數據與計算母體基準 (Population Baseline)
# ==========================================
print(f"[*] 讀取異常結果表: {ANOMALY_CSV}")
df = pd.read_csv(ANOMALY_CSV)

# 計算全體船舶的統計平均值作為對照基準
baseline = {
    'mean_sog_avg': df['mean_sog'].mean(),
    'cog_std_avg': df['cog_std'].mean(),
    'stop_ratio_avg': df['stop_ratio'].mean(),
    'dist_avg': df['total_dist_nmi'].mean()
}

# ==========================================
# 3. 呼叫本地 Ollama 生成專業分析
# ==========================================
def query_ollama(prompt):
    """透過 HTTP REST API 呼叫本地運行的 Ollama 模型"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2  # 低溫保持分析客觀與邏輯嚴謹
        }
    }
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('response', '')
    except Exception as e:
        return f"連線 Ollama 失敗: {e}。請確認終端機是否可執行 ollama。"

# ==========================================
# 4. 逐船生成結構化研究報告
# ==========================================
print(f"[*] 啟動 AI Agent 分析 Top {TOP_N} 異常船舶...")
start_time = time.time()
report_lines = [
    "# AIS 船舶異常航行型態分析報告 (AI Agent 生成)",
    f"- **分析日期**: {DATE_STR}",
    f"- **分析模型**: {MODEL_NAME}",
    f"- **全體平均基準**: 平均航速 {baseline['mean_sog_avg']:.1f} kts | 平均航向變異 {baseline['cog_std_avg']:.1f}° | 平均停滯率 {baseline['stop_ratio_avg']:.2f} | 平均航程 {baseline['dist_avg']:.1f} nmi\n",
    "---"
]

top_vessels = df.head(TOP_N)

for _, row in top_vessels.iterrows():
    mmsi = int(row['MMSI'])
    rank = int(row['anomaly_rank'])
    score = row['anomaly_score']
    
    prompt = f"""
你是一名資深海事交通與 AIS 軌跡分析專家。請根據以下單一船舶的航行特徵與全體平均基準，提供 3~4 點簡要、客觀且具專業海事物理意義的航行型態解釋。

【船舶航行數據】
- MMSI: {mmsi} (異常排名: Rank {rank}, 異常分數: {score:.3f})
- 平均速度 (Mean SOG): {row['mean_sog']:.2f} 節 (全體平均: {baseline['mean_sog_avg']:.2f} 節)
- 航向標準差 (COG Std): {row['cog_std']:.2f}° (全體平均: {baseline['cog_std_avg']:.2f}°)
- 停滯/慢速比例 (Stop Ratio): {row['stop_ratio']:.2f} (全體平均: {baseline['stop_ratio_avg']:.2f})
- 累積總航程 (Total Distance): {row['total_dist_nmi']:.2f} 海浬 (全體平均: {baseline['dist_avg']:.2f} 海浬)

【分析要求】
1. 指出該船哪項特徵最顯著偏離常態。
2. 根據運動學特徵推測其可能的海事作業型態（例如：固定航線渡輪折返、引水船待命接駁、拖船輔助作業、港內慢速盤旋、或錨泊漂移等）。
3. 保持客觀學術語氣，說明這屬於特定作業型態之「特殊航行模式 (Anomalous Pattern)」，非直接宣稱違法或危險。
4. 請以繁體中文回答。
"""
    print(f"[*] 正在分析 Rank {rank} (MMSI: {mmsi})...")
    analysis_text = query_ollama(prompt)
    
    section = f"""
## Rank {rank} | MMSI: {mmsi} (異常分數: {score:.3f})
- **主要特徵**: 平均 SOG `{row['mean_sog']} kts` | 航向變異 `{row['cog_std']}°` | 停滯率 `{row['stop_ratio']}` | 總航程 `{row['total_dist_nmi']} nmi`
- **專家代理分析**:
{analysis_text.strip()}

---"""
    report_lines.append(section)
    print(f"\n--- [Rank {rank} | MMSI: {mmsi}] ---")
    print(analysis_text.strip())
    print("-" * 50)

# ==========================================
# 5. 儲存分析結果為 Markdown
# ==========================================
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[+] AI Agent 分析完成！耗時: {time.time() - start_time:.2f} 秒")
print(f"[+] 完整分析報告已儲存至: {OUTPUT_REPORT}") 