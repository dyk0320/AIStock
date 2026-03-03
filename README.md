# 📈 A股智能分析系统 v4.0

基于 **Gemini AI + Tushare + AKShare + 多Agent辩论** 的 A 股深度分析工具。

输入股票代码 → 大盘+板块+技术+风险+情绪+资金全维度分析 → 5位AI交易员辩论 → 大盘优先裁决。

## 功能

- **技术指标图表**: K线 / MA / MACD / RSI / 布林带，Plotly 交互式可视化
- **技术信号识别**: 金叉死叉 / 超买超卖 / 均线排列 / 量价背离
- **支撑压力位**: 基于 Pivot Point 自动计算
- **风险量化仪表盘**: VaR(95%/99%) / CVaR / 最大回撤 / Sharpe / 波动率
- **大盘环境分析**: 上证/深证/创业板走势、4级预警(红/橙/黄/正常)、系统性风险自动覆盖
- **板块联动分析**: 申万行业识别 + 板块指数对比 + 个股vs大盘/板块强弱
- **恐贪指数**: 6维度综合 (RSI + 波动率 + 价格位置 + 量能 + 涨跌停比 + 均线)
- **市场状态识别**: 温和上涨 / 剧烈上涨 / 阴跌 / 恐慌 / 窄幅整理 / 宽幅震荡
- **资金流向 (AKShare)**: 个股主力净流入 / 北向资金 / 融资融券 / 涨跌停统计
- **实时消息面**: Gemini + Google Search Grounding 联网搜索
- **5 Agent 辩论 + CoT 推理**:
  - 🐂 趋势猎手 (趋势跟踪)
  - 🛡️ 价值守卫 (基本面)
  - 😈 魔鬼代言人 (专业唱反调)
  - ⚡ 短线游侠 (波段交易)
  - 🧠 情绪猎手 (资金+行为金融)
  - 🎯 首席策略官 (大盘优先裁决 + 系统性风险覆盖 + 风险矩阵)

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
