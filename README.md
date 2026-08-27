# 基於開放 AIS 資料與輕量化機器學習之船舶軌跡異常偵測系統

## 一、專案簡介

本專案建立了一套端到端輕量化海事船舶交通分析與無監督異常偵測管線。針對美國國家海洋暨大氣總署（NOAA）公開的海量 AIS 船舶廣播大數據，在一般個人筆記型電腦上實現了高吞吐量的空間串流過濾、時空軌跡重建、海事圓形統計特徵萃取、孤立森林（Isolation Forest）機器學習異常偵測，並結合本地離線大型語言模型（Local LLM AI Agent）進行無幻覺的海事物理與航行作業判讀。

## 二、理論基礎與學術參考

本專案之資料清洗、時空軌跡重建與動態／靜止船舶劃分方法，參考了以下海事領域開放資料研究文獻：

**文獻：** Moritz Hütten (2025). *Maritime Activities Observed Through Open-Access Positioning Data: Moving and Stationary Vessels in the Baltic Sea.*

**論文連結：** https://arxiv.org/abs/2511.23016

### 文獻核心啟發與本專案實作對應

1. 開放 AIS 數據清洗與軌跡重建：參考該文獻針對開放 AIS 接收涵蓋率不均與雜訊點的處理邏輯，本專案在 `02_clean_and_reconstruct.py` 中實現了時序去重、時間戳嚴格排序與基於隱含航速的 GPS 跳點濾除演算法。

2. 航行中與靜止活動解構：依據文獻對船舶作業型態之劃分標準，本專案導入航速臨界閥值（SOG < 0.5 節）計算「停滯時間比例」，成功將一般航道巡航、港內穿梭渡輪與定點／間歇性泊靠作業進行量化區隔。

## 三、AI 輔助機制

本專案在系統架構開發與海事結果判讀中全面導入了現代 AI 輔助：

### 1. AI 輔助開發與效能最佳化

* 藉由 AI 輔助重構資料工程管線，導入 DuckDB Out-of-Core 串流架構，使得數百萬筆的原始全美 AIS CSV 壓縮檔能在 1 秒內完成空間過濾，記憶體峰值控制在 500 MB 以內。
* 導入海事角度數學修正，利用圓形統計學解決 0 度／360 度航向跳躍的邊界計算問題。

### 2. 本地工具增強型 AI Agent

* 確定性事實預計算層：由 Python 預先計算精確倍數（如航速為全體 6.9 倍）、幾何特徵（180 度對蹠折返標籤）與停滯時間，完全消除邊緣語言模型的數值計算顛倒幻覺。
* 本地推論層：利用本地離線運行的 Meta Llama 3.1 8B 或 Qwen 模型，根據確定性事實生成結構嚴謹、客觀的海事作業假說（例如通勤渡輪、引水接駁船、港灣多點維護船）。
* 隱私與離線安全性：推論完全在本地硬體進行，無需任何雲端 API 金鑰，無資料外洩風險。

## 四、系統架構與資料流

```text
[ NOAA 原始 AIS 壓縮檔 (AIS_YYYY_MM_DD.zip, 約 300MB) ]
|
+-> 步驟 01 (01_extract_ais.py): DuckDB 空間串流過濾
[ 舊金山灣區點位 Parquet (sf_bay_clean_*.parquet, 16.5萬筆 / 381艘船) ]
|
+-> 步驟 02 (02_clean_and_reconstruct.py): 時序去重、GPS 跳點濾除
[ 清洗後乾淨軌跡資料集 ]
|
+-> 步驟 03 (03_feature_engineering.py): 圓形統計與 5 維運動學聚合
[ 船舶運動特徵矩陣 (sf_bay_features_*.csv) ]
|
+-> 步驟 04 (04_anomaly_detection.py): StandardScaler + Isolation Forest
[ 異常評分排行榜 (sf_bay_anomalies_*.csv) ]
|
+-> 步驟 05 (05_visualize_results.py): 產生 4 張高解析度學術圖表
+-> 步驟 06 (06_ai_agent_analysis.py): 本地 Ollama 產出確定性海事假說
+-> 步驟 07 (07_generate_final_report.py): 產出完整研究報告與口頭簡報大綱
```

## 五、核心海事運動學特徵

為每艘船舶萃取 5 維核心特徵向量：

1. 平均對地航速（Mean SOG）：單位為節（knots）。
2. 航速變異標準差（SOG std）。
3. 航向圓形標準差（COG Circ Std）：使用單位向量分解法避免 0 度／360 度邊界誤差。
4. 停滯時間比例（Stop Ratio）：SOG < 0.5 節的觀測點比例。
5. 大圓總航程（Total Distance）：透過 Haversine 公式累積相鄰點位之球面距離（海浬）。

## 六、研究視覺化成果

* 圖表 1（`fig1_ais_traffic_distribution.png`）：舊金山灣區 AIS 航跡密度與速度分佈。
* 圖表 2（`fig2_normal_trajectories.png`）：常態大型商船遵循分道通航制（TSS）之航道保持軌跡。
* 圖表 3（`fig3_anomalous_trajectories.png`）：Top 5 異常船舶（渡輪折返、局部作業）軌跡空間疊加圖。
* 圖表 4（`fig4_feature_comparison.png`）：正常船舶 vs 異常船舶特徵分佈箱型圖。

## 七、專案目錄結構

```text
summer-project/

- data/
  - raw/ (NOAA 原始 AIS 壓縮檔，已由 .gitignore 排除)
  - processed/ (空間過濾 Parquet 與特徵 CSV，已由 .gitignore 排除)
- reports/
  - figures/ (4 張高解析度學術圖表)
  - agent_analysis_summary.md (AI Agent 逐船客觀判讀報告)
  - FINAL_RESEARCH_REPORT.md (完整學術研究論文草稿)
  - PRESENTATION_DECK.md (10 頁口頭報告簡報架構)
- scripts/
  - 01_extract_ais.py (DuckDB 串流空間過濾)
  - 02_clean_and_reconstruct.py (軌跡時序排序與 GPS 跳點清洗)
  - 03_feature_engineering.py (圓形統計與 5 維特徵聚合)
  - 04_anomaly_detection.py (Isolation Forest 無監督異常偵測)
  - 05_visualize_results.py (圖表繪製模組)
  - 06_ai_agent_analysis.py (工具增強型本地 Ollama AI Agent)
  - 07_generate_final_report.py (綜合研究報告自動產出器)
- .gitignore (排除大型資料檔與虛擬環境)
- README.md (專案說明文件)
- requirements.txt (專案相依套件清單)
```

## 八、快速啟動指南

### 1. 環境建置

```bash
git clone https://github.com/1123roy-beep/summer-project.git
cd summer-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 啟動本地 Ollama 語言模型

```bash
ollama pull llama3.1:8b
```

### 3. 依序執行完整管線

```bash
python scripts/01_extract_ais.py
python scripts/02_clean_and_reconstruct.py
python scripts/03_feature_engineering.py
python scripts/04_anomaly_detection.py
python scripts/05_visualize_results.py
python scripts/06_ai_agent_analysis.py
python scripts/07_generate_final_report.py
```

## 九、引用與致謝

資料來源：美國 NOAA Office for Coastal Management & BOEM
https://hub.marinecadastre.gov/pages/vesseltraffic

學術參考：Hutten, M. (2025). *Maritime Activities Observed Through Open-Access Positioning Data*. arXiv:2511.23016
https://arxiv.org/abs/2511.23016

本地推論：基於 Ollama 與 Meta Llama / Qwen 開源社群。
