# 📈 A股智能分析系统

基于 **Gemini AI + Tushare + 技术指标引擎** 的 A 股单只股票深度分析工具。

输入股票代码 → 获取 K线图表 + 技术指标可视化 + AI 深度研报。

## 功能

- **技术指标图表**: K线 / MA / MACD / RSI / 布林带，Plotly 交互式可视化
- **技术信号识别**: 金叉死叉 / 超买超卖 / 均线排列 / 量价背离
- **支撑压力位**: 基于 Pivot Point 自动计算
- **实时消息面**: Gemini + Google Search Grounding 联网搜索最新政策公告
- **AI 深度分析**: 多空评分 / 趋势判断 / 交易策略 / 风险提示

## 部署到 Streamlit Cloud

### 1. 创建 GitHub 仓库

```bash
git init
git add .
git commit -m "init: stock analyzer"
git remote add origin https://github.com/你的用户名/stock-analyzer.git
git push -u origin main
```

### 2. 部署

1. 打开 [share.streamlit.io](https://share.streamlit.io)
2. 点击 **New app** → 选择你的 GitHub 仓库
3. Main file path 填 `app.py`
4. 点击 **Advanced settings** → 在 **Secrets** 中粘贴:

```toml
GEMINI_API_KEY = "你的Gemini API Key"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
TUSHARE_TOKEN = "你的Tushare Token"
TUSHARE_PROXY_URL = "你的Tushare代理地址(可选)"
```

5. 点击 **Deploy**，等待部署完成

### 3. 获取 API Key

- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) 免费申请
- **Tushare**: [tushare.pro](https://tushare.pro) 注册获取 token

## 本地运行

```bash
pip install -r requirements.txt

# 创建 secrets 文件
cp secrets.toml.example .streamlit/secrets.toml
# 编辑 .streamlit/secrets.toml 填入你的 key

streamlit run app.py
```

## ⚠️ 免责声明

本工具仅供学习研究，分析结果不构成任何投资建议。投资有风险，入市需谨慎。
