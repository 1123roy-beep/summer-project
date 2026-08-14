import os
import json
import urllib.request
import pandas as pd
import time

DATE_STR = "2023_01_01"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"
ANOMALY_CSV = os.path.join(PROCESSED_DIR, f"sf_bay_anomalies_{DATE_STR}.csv")
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "agent_analysis_summary.md")

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
TOP_N = 3

os.makedirs(REPORTS_DIR, exist_ok=True)
df = pd.read_csv(ANOMALY_CSV)

# 計算母體基準
avg_sog = df['mean_sog'].mean()
avg_cog_std = df['cog_std'].mean()
avg_stop = df['stop_ratio'].mean()
avg_dist = df['total_dist_nmi'].mean()

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode('utf-8')).get('response', '')
    except Exception as e:
        return f"連線失敗: {e}"

# ============================================================
# Python 工具層：預先計算客觀海事物理事實 (Deterministic Facts)
# ============================================================
def generate_vessel_facts(row):
    facts = []
    # 速度對比
    sog_ratio = row['mean_sog'] / (avg_sog + 1e-5)
    if sog_ratio >= 3.0:
        facts.append(f"- 【航速特徵】：平均 SOG ({row['mean_sog']} 節) 高達全體平均 ({avg_sog:.1f} 節) 的 {sog_ratio:.1f} 倍，屬於持續高航速運作。")
    elif row['mean_sog'] < 1.0:
        facts.append(f"- 【航速特徵】：平均 SOG ({row['mean_sog']} 節) 低於 1 節，長時間處於極低速或靜止狀態。")
    else:
        facts.append(f"- 【航速特徵】：平均 SOG ({row['mean_sog']} 節) 接近常規巡航速度。")

    # 航向變異分析
    if row['cog_std'] >= 140:
        facts.append(f"- 【轉向行為】：航向標準差高達 {row['cog_std']}° (全體平均僅 {avg_cog_std:.1f}°)，呈現近乎 180° 對蹠反向折返特性。")
    elif row['cog_std'] >= 70:
        facts.append(f"- 【轉向行為】：航向標準差為 {row['cog_std']}°，轉向頻繁，非直線航道保持。")
    else:
        facts.append(f"- 【轉向行為】：航向標準差 {row['cog_std']}°，航線筆直穩定。")

    # 停滯時間
    if row['stop_ratio'] >= 0.6:
        facts.append(f"- 【作業節奏】：有 {int(row['stop_ratio']*100)}% 時間停滯或慢速，顯示具間歇性起停或靠泊作業。")
    elif row['stop_ratio'] <= 0.2:
        facts.append(f"- 【作業節奏】：停滯時間僅 {int(row['stop_ratio']*100)}%，全程幾乎無間斷航行。")

    # 總航程
    dist_ratio = row['total_dist_nmi'] / (avg_dist + 1e-5)
    facts.append(f"- 【累積活動量】：總航程 {row['total_dist_nmi']} 海浬 (全體平均 {avg_dist:.1f} 海浬，為平均的 {dist_ratio:.1f} 倍)。")
    return "\n".join(facts)

print(f"[*] 啟動 Tool-Augmented AI Agent 分析 Top {TOP_N} 船舶...")
report_lines = [
    "# AIS 船舶異常航行型態分析報告 (精確修正版)",
    f"- **全體平均基準**: 平均航速 {avg_sog:.1f} kts | 航向變異 {avg_cog_std:.1f}° | 停滯率 {avg_stop:.2f} | 航程 {avg_dist:.1f} nmi\n",
    "---"
]

for _, row in df.head(TOP_N).iterrows():
    mmsi = int(row['MMSI'])
    rank = int(row['anomaly_rank'])
    score = row['anomaly_score']
    
    verified_facts = generate_vessel_facts(row)
    
    prompt = f"""
你是一名海事交通與 AIS 軌跡分析專家。請嚴格根據以下由 Python 系統預先驗證的事實數據，為該船舶產出一份精確、專業、無數值矛盾的航行型態診斷。

【系統驗證數據與事實】
- MMSI: {mmsi} (異常排名: Rank {rank}, 異常分數: {score:.3f})
{verified_facts}

【撰寫指引】
1. 請以 3 個簡短重點歸納其運動學特徵。
2. 根據事實（如高航速+180度折返、或長航程+高停滯），推測其具體海事作業型態（如雙向穿梭渡輪、引水接駁、拖船維護）。
3. 嚴格遵守上述已驗證的數值事實，切勿自行扭曲高低關係。
4. 請以繁體中文回答。
"""
    print(f"[*] 正在分析 Rank {rank} (MMSI: {mmsi})...")
    analysis_text = query_ollama(prompt)
    
    section = f"""
## Rank {rank} | MMSI: {mmsi} (異常分數: {score:.3f})
**客觀物理事實**:
{verified_facts}

**專家代理判讀**:
{analysis_text.strip()}

---"""
    report_lines.append(section)
    print(f"\n--- [Rank {rank} | MMSI: {mmsi}] ---")
    print(analysis_text.strip())
    print("-" * 50)

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\n[+] 精確版報告已更新至: {OUTPUT_REPORT}")