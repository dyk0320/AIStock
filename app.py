# coding=utf-8
"""
A股智能分析系统 - Streamlit Web版
基于 Gemini + Tushare + 技术指标引擎
"""

import datetime
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tushare as ts
from google import genai
from google.genai import types

warnings.filterwarnings("ignore")

# =====================================================================
# 页面配置
# =====================================================================
st.set_page_config(
    page_title="A股智能分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 样式
# =====================================================================
st.markdown("""
<style>
    /* ---- 基础 ---- */
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* ---- 指标网格 (自适应) ---- */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem 0.6rem;
        text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 0.78rem; margin-bottom: 0.2rem; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; }
    .metric-card .up { color: #ef4444; }
    .metric-card .down { color: #22c55e; }
    .metric-card .neutral { color: #e2e8f0; }

    /* ---- 关键价位网格 ---- */
    .levels-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.4rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .level-item .level-label { color: #8892b0; font-size: 0.72rem; }
    .level-item .level-val { font-weight: 700; font-size: 1.05rem; }
    .level-item .lv-r { color: #ef4444; }
    .level-item .lv-s { color: #22c55e; }
    .level-item .lv-p { color: #94a3b8; }

    /* ---- 信号标签 ---- */
    .signal-wrap { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
    .signal-tag {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .signal-bullish { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .signal-bearish { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .signal-neutral { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }

    /* ---- 侧边栏 ---- */
    .sidebar-title {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        padding-bottom: 0.5rem; border-bottom: 2px solid #3b82f6; margin-bottom: 1rem;
    }

    /* ---- 隐藏默认元素 ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---- 移动端首页卡片 ---- */
    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }
    .landing-card {
        background: #1e293b; border-radius: 12px; padding: 1.2rem;
        text-align: center;
    }
    .landing-card h4 { margin: 0.4rem 0; font-size: 1rem; }
    .landing-card p { color: #94a3b8; font-size: 0.85rem; margin: 0; }

    /* ---- 辩论 Agent 卡片 ---- */
    .agent-header {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.8rem 1rem; border-radius: 10px 10px 0 0;
        font-weight: 700; font-size: 0.95rem;
    }
    .agent-header .agent-emoji { font-size: 1.4rem; }
    .agent-header .agent-stance {
        margin-left: auto; padding: 0.15rem 0.6rem; border-radius: 12px;
        font-size: 0.72rem; font-weight: 600;
    }
    .stance-bullish { background: rgba(239,68,68,0.2); color: #ef4444; }
    .stance-bearish { background: rgba(34,197,94,0.2); color: #22c55e; }
    .stance-neutral { background: rgba(234,179,8,0.2); color: #eab308; }

    .agent-trend   .agent-header { background: linear-gradient(135deg, #1a1a2e, #1e293b); border-left: 3px solid #f59e0b; }
    .agent-value   .agent-header { background: linear-gradient(135deg, #1a1a2e, #1e293b); border-left: 3px solid #3b82f6; }
    .agent-contra  .agent-header { background: linear-gradient(135deg, #1a1a2e, #1e293b); border-left: 3px solid #a855f7; }
    .agent-swing   .agent-header { background: linear-gradient(135deg, #1a1a2e, #1e293b); border-left: 3px solid #ef4444; }

    .verdict-box {
        background: linear-gradient(135deg, #0c1524 0%, #1a1a2e 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
    }
    .verdict-title {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 1.1rem; font-weight: 700; color: #f59e0b;
        margin-bottom: 0.8rem;
    }

    /* ============ 移动端适配 (≤768px) ============ */
    @media (max-width: 768px) {
        .block-container { padding: 0.8rem 0.6rem !important; }

        /* 指标: 3+2 布局 */
        .metrics-grid { grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
        .metric-card { padding: 0.7rem 0.3rem; border-radius: 8px; }
        .metric-card .label { font-size: 0.68rem; }
        .metric-card .value { font-size: 1.1rem; }

        /* 价位: 3+2 */
        .levels-grid { grid-template-columns: repeat(3, 1fr); gap: 0.3rem; }
        .level-item .level-label { font-size: 0.65rem; }
        .level-item .level-val { font-size: 0.92rem; }

        /* 信号标签 */
        .signal-tag { font-size: 0.7rem; padding: 0.2rem 0.55rem; }

        /* 首页卡片竖排 */
        .landing-grid { grid-template-columns: 1fr; gap: 0.6rem; }
        .landing-card { padding: 0.8rem; }

        /* 标题缩小 */
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* 隐藏 Plotly 模式栏 */
        .modebar { display: none !important; }

        /* Tab 字号 */
        .stTabs [data-baseweb="tab"] { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }

        /* 辩论卡片 */
        .agent-header { padding: 0.6rem 0.8rem; font-size: 0.85rem; }
        .agent-header .agent-emoji { font-size: 1.2rem; }
        .verdict-box { padding: 0.8rem 1rem; }
        .verdict-title { font-size: 0.95rem; }
    }

    /* ============ 超小屏 (≤480px) ============ */
    @media (max-width: 480px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        .metric-card .value { font-size: 1rem; }
        .levels-grid { grid-template-columns: repeat(2, 1fr); }
    }

    /* ---- 输入行对齐 ---- */
    [data-testid="column"] { display: flex; align-items: end; }
</style>

<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)


# =====================================================================
# 配置 & 初始化
# =====================================================================
def get_config():
    """从 st.secrets 读取密钥"""
    return {
        "gemini_key": st.secrets["GEMINI_API_KEY"],
        "tushare_token": st.secrets["TUSHARE_TOKEN"],
        "tushare_proxy": st.secrets.get("TUSHARE_PROXY_URL", ""),
        "gemini_model": st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17"),
    }


@st.cache_resource
def init_clients():
    """初始化 API 客户端 (缓存)"""
    cfg = get_config()
    gemini_client = genai.Client(api_key=cfg["gemini_key"])
    ts.set_token(cfg["tushare_token"])
    pro = ts.pro_api()
    if cfg["tushare_proxy"]:
        pro._DataApi__http_url = cfg["tushare_proxy"]
    return gemini_client, pro, cfg


# =====================================================================
# 数据获取
# =====================================================================
def build_ts_code(code: str) -> str:
    code = code.strip()
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


@st.cache_data(ttl=300)
def fetch_kline(_pro, ts_code: str, days: int = 60):
    end = datetime.datetime.now().strftime("%Y%m%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=days + 15)).strftime("%Y%m%d")
    df = _pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df.empty:
        return df
    return df.sort_values("trade_date").reset_index(drop=True)


@st.cache_data(ttl=600)
def fetch_financial(_pro, ts_code: str):
    try:
        df = _pro.fina_indicator(ts_code=ts_code, limit=1)
        if df.empty:
            return {}
        f = df.iloc[0]
        return {
            "roe": f.get("roe"),
            "grossprofit_margin": f.get("grossprofit_margin"),
            "netprofit_yoy": f.get("netprofit_yoy"),
            "revenue_yoy": f.get("revenue_yoy"),
            "debt_to_assets": f.get("debt_to_assets"),
            "eps": f.get("eps"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_stock_name(_pro, ts_code: str):
    try:
        df = _pro.stock_basic(ts_code=ts_code, fields="ts_code,name")
        if not df.empty:
            return df.iloc[0]["name"]
    except Exception:
        pass
    return ts_code


@st.cache_data(ttl=3600)
def fetch_company(_pro, ts_code: str):
    try:
        df = _pro.stock_company(ts_code=ts_code, fields="business_scope,main_business")
        if not df.empty:
            return df.iloc[0].get("main_business", "未知")
    except Exception:
        pass
    return "未知"


# =====================================================================
# 技术指标引擎
# =====================================================================
class TechEngine:

    @staticmethod
    def calc_all(df, ma_periods=(5, 10, 20), rsi_period=14, boll_period=20, boll_std=2):
        # MA
        for p in ma_periods:
            df[f"MA{p}"] = df["close"].rolling(window=p).mean()

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = 2 * (df["DIF"] - df["DEA"])

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
        loss = (-delta).clip(lower=0).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BOLL_MID"] = df["close"].rolling(window=boll_period).mean()
        rolling_std = df["close"].rolling(window=boll_period).std()
        df["BOLL_UP"] = df["BOLL_MID"] + boll_std * rolling_std
        df["BOLL_DN"] = df["BOLL_MID"] - boll_std * rolling_std

        # Volume ratio
        df["VOL_MA5"] = df["vol"].rolling(window=5).mean()
        df["VOL_RATIO"] = df["vol"] / df["VOL_MA5"].replace(0, np.nan)

        return df

    @staticmethod
    def get_signals(df):
        if len(df) < 3:
            return []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []

        # MACD cross
        if latest["DIF"] > latest["DEA"] and prev["DIF"] <= prev["DEA"]:
            signals.append(("MACD 金叉", "bullish"))
        elif latest["DIF"] < latest["DEA"] and prev["DIF"] >= prev["DEA"]:
            signals.append(("MACD 死叉", "bearish"))
        elif latest["DIF"] > latest["DEA"]:
            signals.append(("MACD 多头", "bullish"))
        else:
            signals.append(("MACD 空头", "bearish"))

        # RSI
        rsi = latest.get("RSI", 50)
        if pd.notna(rsi):
            if rsi > 80:
                signals.append((f"RSI {rsi:.0f} 超买", "bearish"))
            elif rsi > 70:
                signals.append((f"RSI {rsi:.0f} 偏强", "neutral"))
            elif rsi < 20:
                signals.append((f"RSI {rsi:.0f} 超卖", "bullish"))
            elif rsi < 30:
                signals.append((f"RSI {rsi:.0f} 偏弱", "neutral"))

        # Bollinger
        boll_up = latest.get("BOLL_UP")
        boll_dn = latest.get("BOLL_DN")
        if pd.notna(boll_up) and latest["close"] > boll_up:
            signals.append(("突破布林上轨", "bearish"))
        elif pd.notna(boll_dn) and latest["close"] < boll_dn:
            signals.append(("跌破布林下轨", "bullish"))

        # MA alignment
        ma_vals = [latest.get(f"MA{p}") for p in (5, 10, 20)]
        ma_vals = [v for v in ma_vals if pd.notna(v)]
        if len(ma_vals) >= 3:
            if ma_vals[0] > ma_vals[1] > ma_vals[2]:
                signals.append(("均线多头排列", "bullish"))
            elif ma_vals[0] < ma_vals[1] < ma_vals[2]:
                signals.append(("均线空头排列", "bearish"))

        # Volume-price divergence
        if len(df) >= 10:
            recent = df.tail(10)
            p_trend = recent["close"].iloc[-1] - recent["close"].iloc[0]
            v_trend = recent["vol"].iloc[-1] - recent["vol"].iloc[0]
            if p_trend > 0 and v_trend < 0:
                signals.append(("顶背离:价涨量缩", "bearish"))
            elif p_trend < 0 and v_trend > 0:
                signals.append(("底背离:价跌量增", "bullish"))

        return signals

    @staticmethod
    def support_resistance(df):
        recent = df.tail(20)
        if recent.empty:
            return {}
        c = recent["close"].iloc[-1]
        h = recent["high"].max()
        l = recent["low"].min()
        pivot = (h + l + c) / 3
        return {
            "强压力": round(pivot + (h - l), 2),
            "弱压力": round(2 * pivot - l, 2),
            "枢轴位": round(pivot, 2),
            "弱支撑": round(2 * pivot - h, 2),
            "强支撑": round(pivot - (h - l), 2),
        }


# =====================================================================
# 可视化图表
# =====================================================================
def build_main_chart(df):
    """K线 + 均线 + 布林带 + 成交量"""
    display_df = df.tail(30).copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=display_df["trade_date"],
        open=display_df["open"], high=display_df["high"],
        low=display_df["low"], close=display_df["close"],
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#22c55e", decreasing_fillcolor="#22c55e",
        name="K线",
    ), row=1, col=1)

    # MA lines
    colors = {"MA5": "#f59e0b", "MA10": "#3b82f6", "MA20": "#a855f7"}
    for ma, color in colors.items():
        if ma in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df["trade_date"], y=display_df[ma],
                line=dict(width=1.2, color=color),
                name=ma, opacity=0.9,
            ), row=1, col=1)

    # Bollinger Bands
    if "BOLL_UP" in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df["trade_date"], y=display_df["BOLL_UP"],
            line=dict(width=0.8, color="rgba(100,100,100,0.5)", dash="dot"),
            name="布林上轨", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=display_df["trade_date"], y=display_df["BOLL_DN"],
            line=dict(width=0.8, color="rgba(100,100,100,0.5)", dash="dot"),
            fill="tonexty", fillcolor="rgba(100,100,100,0.05)",
            name="布林下轨", showlegend=False,
        ), row=1, col=1)

    # Volume
    vol_colors = ["#ef4444" if c >= o else "#22c55e"
                  for c, o in zip(display_df["close"], display_df["open"])]
    fig.add_trace(go.Bar(
        x=display_df["trade_date"], y=display_df["vol"],
        marker_color=vol_colors, name="成交量", opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        margin=dict(l=5, r=5, t=30, b=5),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor="#1e293b", title=""),
        yaxis2=dict(gridcolor="#1e293b", title=""),
        xaxis=dict(gridcolor="#1e293b"),
        xaxis2=dict(gridcolor="#1e293b"),
        dragmode="pan",
    )

    return fig


def build_macd_chart(df):
    display_df = df.tail(30).copy()
    fig = go.Figure()

    hist_colors = ["#ef4444" if v >= 0 else "#22c55e" for v in display_df["MACD_HIST"]]
    fig.add_trace(go.Bar(
        x=display_df["trade_date"], y=display_df["MACD_HIST"],
        marker_color=hist_colors, name="MACD柱", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=display_df["trade_date"], y=display_df["DIF"],
        line=dict(width=1.5, color="#f59e0b"), name="DIF",
    ))
    fig.add_trace(go.Scatter(
        x=display_df["trade_date"], y=display_df["DEA"],
        line=dict(width=1.5, color="#3b82f6"), name="DEA",
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=5, r=5, t=10, b=5),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center", font_size=10),
        yaxis=dict(gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        dragmode="pan",
    )
    return fig


def build_rsi_chart(df):
    display_df = df.tail(30).copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=display_df["trade_date"], y=display_df["RSI"],
        line=dict(width=2, color="#a855f7"), name="RSI(14)", fill="tozeroy",
        fillcolor="rgba(168,85,247,0.08)",
    ))

    # Overbought/oversold lines
    for level, color, label in [(80, "#ef4444", "超买"), (20, "#22c55e", "超卖")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.5,
                      annotation_text=label, annotation_font_color=color)
    fig.add_hline(y=50, line_dash="dot", line_color="#475569", opacity=0.3)

    fig.update_layout(
        height=200,
        margin=dict(l=5, r=5, t=10, b=5),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        yaxis=dict(gridcolor="#1e293b", range=[0, 100]),
        xaxis=dict(gridcolor="#1e293b"),
        showlegend=False,
        dragmode="pan",
    )
    return fig


# =====================================================================
# 多 Agent 交易员定义
# =====================================================================
TRADER_AGENTS = [
    {
        "id": "trend",
        "name": "趋势猎手",
        "emoji": "🐂",
        "style": "agent-trend",
        "tab_label": "🐂 趋势猎手",
        "desc": "趋势跟踪 · 只做主升浪",
        "system_prompt": """你是"趋势猎手", 一位激进的趋势跟踪交易员, 15年A股实战经验。
你的核心信仰: 趋势是你唯一的朋友, 均线多头排列就是印钞机。

你的分析框架 (必须严格遵循):
- 重点关注: MA5/MA10/MA20 排列状态、MACD 方向与力度、成交量趋势
- 你最看重: 趋势强度和量价配合
- 你最不在乎: 估值高低 (趋势面前不言顶)

输出要求:
## 立场: [看多/看空/中性] (用一句话概括理由)
## 关键论据 (3条, 每条必须引用具体技术数据)
## 操作建议: 方向 + 关键价位 + 时间维度
## 信心指数: X/10

语气要求: 果断、自信、进攻性强。用交易员的口吻, 不要学术化。200-350字。""",
    },
    {
        "id": "value",
        "name": "价值守卫",
        "emoji": "🛡️",
        "style": "agent-value",
        "tab_label": "🛡️ 价值守卫",
        "desc": "基本面为王 · 安全边际",
        "system_prompt": """你是"价值守卫", 一位保守的价值投资者, 崇尚格雷厄姆和巴菲特。
你的核心信仰: 好公司+好价格=好投资。短期涨跌是噪音, ROE才是真相。

你的分析框架 (必须严格遵循):
- 重点关注: ROE、毛利率、净利润增速、营收增速、资产负债率
- 你最看重: 业务质量和财务健康度
- 你会质疑: 纯炒作题材股、业绩不匹配的高估值

输出要求:
## 立场: [看多/看空/中性] (用一句话概括理由)
## 关键论据 (3条, 每条必须引用具体财务数据)
## 操作建议: 方向 + 安全边际价位 + 持有周期
## 信心指数: X/10

语气要求: 沉稳、严谨、有时会泼冷水。经常提醒风险。200-350字。""",
    },
    {
        "id": "contra",
        "name": "逆向思维",
        "emoji": "🔄",
        "style": "agent-contra",
        "tab_label": "🔄 逆向思维",
        "desc": "人弃我取 · 别人恐惧我贪婪",
        "system_prompt": """你是"逆向思维", 一位逆向投资专家, 专门在市场共识的反面寻找机会。
你的核心信仰: 当所有人看多时风险最大, 当所有人恐慌时机会最好。

你的分析框架 (必须严格遵循):
- 重点关注: RSI 超买超卖、布林带极端位置、量价背离、市场情绪
- 你最擅长: 找出市场可能忽视的反面逻辑
- 你会主动唱反调: 如果技术面看多, 你找看空的理由; 反之亦然

输出要求:
## 立场: [看多/看空/中性] (必须给出与表面趋势相反或不同的视角)
## 关键论据 (3条, 重点指出市场可能忽视的风险或机会)
## 操作建议: 方向 + 逆向布局的价位 + 触发条件
## 信心指数: X/10

语气要求: 冷静、犀利、喜欢挑战共识。"大家都看好? 那我告诉你为什么该小心。" 200-350字。""",
    },
    {
        "id": "swing",
        "name": "短线游侠",
        "emoji": "⚡",
        "style": "agent-swing",
        "tab_label": "⚡ 短线游侠",
        "desc": "快进快出 · 波段为王",
        "system_prompt": """你是"短线游侠", 一位日内/波段交易高手, 擅长捕捉1-5天的短期波动。
你的核心信仰: 不预测方向, 只捕捉波动。关键是风险收益比和执行纪律。

你的分析框架 (必须严格遵循):
- 重点关注: 日K线形态、量比异动、支撑压力位精确计算、RSI/布林带位置
- 你最看重: 精确的买卖点位和止损位
- 你最不在乎: 长期基本面 (那是别人的事)

输出要求:
## 立场: [做多/做空/观望] (用一句话概括短线判断)
## 关键论据 (3条, 必须精确到价格和百分比)
## 操作建议: 入场价 / 止损价 / 止盈价 + 仓位比例 + 持有天数
## 信心指数: X/10

语气要求: 精准、干脆、数字说话。每句话都要有具体价位。200-350字。""",
    },
]

JUDGE_SYSTEM_PROMPT = """你是"首席策略官", 负责主持这场交易员辩论会并做出最终裁决。

你刚刚听取了4位不同风格交易员的分析:
- 趋势猎手: 趋势跟踪派, 重技术面动量
- 价值守卫: 价值投资派, 重基本面
- 逆向思维: 逆向投资派, 专找市场盲点
- 短线游侠: 波段交易派, 重短期买卖点

你的任务:

## 一、观点交锋总结
用1-2段话总结4位交易员的核心分歧在哪, 谁和谁意见一致, 谁和谁针锋相对。

## 二、综合评分
从以下维度给出最终评分 (每项1-10分):
| 维度 | 评分 | 依据 |
|------|------|------|
| 趋势强度 | | |
| 量价配合 | | |
| 基本面质量 | | |
| 消息面催化 | | |
| 风险收益比 | | |
| **综合** | | |

## 三、最终裁决
1. **操作方向**: 强烈买入 / 买入 / 持有 / 减仓 / 强烈卖出
2. **目标仓位**: 建议占总资金百分比
3. **关键价位**: 止损 / 目标 / 加仓位
4. **执行节奏**: 具体分批方案
5. **时间维度**: 短线 / 波段 / 中线

## 四、最大风险警示
综合所有人意见, 最需要警惕的1-2个风险。

你的裁决要有主见, 不是简单平均, 而是基于逻辑判断哪位交易员的论据更有说服力。
如果多数人观点一致, 要特别重视逆向思维的警告。
"""


def build_prompt(stock_name, ts_code, df, fina, business, news_text):
    latest = df.iloc[-1]
    kline_lines = []
    for _, r in df.tail(15).iloc[::-1].iterrows():
        chg = r["pct_chg"]
        kline_lines.append(
            f"{r['trade_date']} O:{r['open']:.2f} H:{r['high']:.2f} "
            f"L:{r['low']:.2f} C:{r['close']:.2f} {chg:+.2f}%"
        )

    fina_parts = []
    labels = {"roe": "ROE", "grossprofit_margin": "毛利率", "netprofit_yoy": "净利增长",
              "revenue_yoy": "营收增长", "debt_to_assets": "负债率", "eps": "EPS"}
    for k, label in labels.items():
        v = fina.get(k)
        fina_parts.append(f"{label}:{v:.2f}" if v is not None and str(v) != "nan" else f"{label}:N/A")

    # Tech summary
    tech_lines = []
    for p in (5, 10, 20):
        v = latest.get(f"MA{p}")
        if pd.notna(v):
            pos = "上方" if latest["close"] > v else "下方"
            tech_lines.append(f"MA{p}={v:.2f}(价在{pos})")

    rsi_val = latest.get("RSI", 50)
    if pd.isna(rsi_val):
        rsi_val = 50

    levels = TechEngine.support_resistance(df)
    level_text = " | ".join(f"{k}:{v}" for k, v in levels.items()) if levels else "N/A"

    return f"""
标的: {stock_name} ({ts_code})
现价: {latest['close']:.2f} | 今日涨跌: {latest['pct_chg']:+.2f}%

【公司概况】业务: {business}
【财务】{' | '.join(fina_parts)}

【K线(近15日)】
{chr(10).join(kline_lines)}

【技术指标】
均线: {' | '.join(tech_lines)}
MACD: DIF={latest.get('DIF', 0):.3f} DEA={latest.get('DEA', 0):.3f}
RSI(14): {rsi_val:.1f}
关键价位: {level_text}

{news_text}

请按照系统指令框架输出完整分析。
"""


# =====================================================================
# Agent 调用
# =====================================================================
def run_single_agent(client, agent, data_prompt, model):
    """调用单个交易员 Agent, 返回文本结果"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=data_prompt,
            config=types.GenerateContentConfig(
                system_instruction=agent["system_prompt"],
                temperature=0.5,
                max_output_tokens=1500,
            ),
        )
        return response.text if response.text else "(该交易员未给出意见)"
    except Exception as e:
        return f"(调用失败: {e})"


def run_judge(client, stock_name, data_prompt, agent_opinions, model):
    """裁判综合所有意见给出最终裁决"""
    opinions_text = ""
    for agent, opinion in agent_opinions:
        opinions_text += f"\n{'='*40}\n"
        opinions_text += f"【{agent['emoji']} {agent['name']}】({agent['desc']})\n"
        opinions_text += f"{opinion}\n"

    judge_prompt = f"""
以下是关于 {stock_name} 的数据:
{data_prompt}

{'='*50}
以下是4位交易员的分析意见:
{opinions_text}
{'='*50}

请作为首席策略官, 综合以上所有信息和观点, 给出你的最终裁决。
"""
    try:
        response = client.models.generate_content_stream(
            model=model,
            contents=judge_prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=3000,
            ),
        )
        return response
    except Exception as e:
        return None


def extract_stance(text):
    """从 Agent 输出中提取立场标签"""
    text_lower = text[:200]
    if any(k in text_lower for k in ["看多", "做多", "买入", "强烈买入"]):
        return "看多", "bullish"
    elif any(k in text_lower for k in ["看空", "做空", "卖出", "减仓", "强烈卖出"]):
        return "看空", "bearish"
    return "中性", "neutral"


# =====================================================================
# 联网搜索
# =====================================================================
def search_news(client, stock_name, business, model):
    prompt = f"""搜索A股标的 "{stock_name}" (主营: {business}) 最新信息:
1. 最新行业政策/监管
2. 近期公司公告(业绩/定增/回购)
3. 行业趋势变化
4. 机构评级/北向资金
5. 风险事件
每条2-3句, 标注来源和时间, 搜索不到标注"暂无"。"""

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        text = resp.text if resp.text else ""
        sources = []
        try:
            meta = resp.candidates[0].grounding_metadata
            if meta and meta.grounding_chunks:
                for ch in meta.grounding_chunks[:5]:
                    if hasattr(ch, "web") and ch.web:
                        sources.append(f"- {ch.web.title}")
        except Exception:
            pass
        src = "\n".join(sources) if sources else "(无明确来源)"
        return f"【实时消息面 (联网搜索)】\n{text}\n\n来源:\n{src}"
    except Exception as e:
        return f"(联网搜索异常: {e})"


# =====================================================================
# 主界面
# =====================================================================
def main():
    # Sidebar (secondary settings only)
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ 分析设置</div>', unsafe_allow_html=True)
        kline_days = st.selectbox("K线回溯天数", [30, 60, 90, 120], index=1)
        enable_search = st.toggle("联网搜索", value=True, help="使用Google Search获取最新消息")
        st.divider()
        st.caption("⚠️ 分析仅供参考，不构成投资建议")
        st.caption(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Main area - input always visible
    st.markdown("## 📈 A股智能分析系统")
    st.caption("Gemini AI + 技术指标引擎 + 实时联网搜索")

    input_col1, input_col2 = st.columns([3, 1])
    with input_col1:
        stock_code = st.text_input(
            "股票代码",
            placeholder="输入6位代码，如 600519",
            max_chars=6,
            label_visibility="collapsed",
            help="支持沪深A股，输入纯数字代码",
        )
    with input_col2:
        analyze_btn = st.button("🚀 分析", type="primary", use_container_width=True)

    if not analyze_btn or not stock_code:
        st.markdown("""
        <div class="landing-grid">
            <div class="landing-card">
                <h4>🔍 技术面</h4>
                <p>MA / MACD / RSI / 布林带<br>支撑压力 · 量价背离</p>
            </div>
            <div class="landing-card">
                <h4>🌐 消息面</h4>
                <p>Gemini + Google Search<br>政策 · 公告 · 行业</p>
            </div>
            <div class="landing-card">
                <h4>🤖 AI研报</h4>
                <p>多空评分 · 趋势判断<br>策略 · 风险提示</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Validate
    if len(stock_code) != 6 or not stock_code.isdigit():
        st.error("请输入有效的6位股票代码")
        return

    # Init
    try:
        gemini_client, pro, cfg = init_clients()
    except Exception as e:
        st.error(f"初始化失败，请检查 secrets 配置: {e}")
        return

    ts_code = build_ts_code(stock_code)

    # ---- Fetch data ----
    with st.status("正在获取数据...", expanded=True) as status:
        st.write("📥 拉取K线数据...")
        df = fetch_kline(pro, ts_code, kline_days)
        if df.empty:
            st.error(f"未找到 {ts_code} 的K线数据，请检查代码是否正确")
            return

        st.write("📊 拉取财务数据...")
        fina = fetch_financial(pro, ts_code)

        st.write("🏢 获取公司信息...")
        stock_name = fetch_stock_name(pro, ts_code)
        business = fetch_company(pro, ts_code)

        st.write("⚙️ 计算技术指标...")
        df = TechEngine.calc_all(df)
        signals = TechEngine.get_signals(df)
        levels = TechEngine.support_resistance(df)

        news_text = ""
        if enable_search:
            st.write("🌐 联网搜索最新消息...")
            news_text = search_news(gemini_client, stock_name, business, cfg["gemini_model"])

        status.update(label="数据准备完毕 ✅", state="complete", expanded=False)

    # ---- Header ----
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
    change_pct = latest["pct_chg"]
    change_color = "up" if change_pct >= 0 else "down"

    st.markdown(f"## {stock_name}  `{ts_code}`")

    # Metric cards (CSS grid, auto-responsive)
    rsi_val = latest.get("RSI", 50)
    rsi_class = "up" if pd.notna(rsi_val) and rsi_val > 70 else ("down" if pd.notna(rsi_val) and rsi_val < 30 else "neutral")
    vol_ratio = latest.get("VOL_RATIO", 1.0)
    vr_class = "up" if pd.notna(vol_ratio) and vol_ratio > 1.5 else ("down" if pd.notna(vol_ratio) and vol_ratio < 0.5 else "neutral")

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">最新价</div>
            <div class="value {change_color}">{latest['close']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">涨跌幅</div>
            <div class="value {change_color}">{change_pct:+.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">RSI(14)</div>
            <div class="value {rsi_class}">{rsi_val:.1f}</div>
        </div>
        <div class="metric-card">
            <div class="label">量比</div>
            <div class="value {vr_class}">{vol_ratio:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">成交额</div>
            <div class="value neutral">{latest['amount']/10000:.0f}万</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Signal tags
    if signals:
        tags_html = '<div class="signal-wrap"><strong style="margin-right:0.4rem">技术信号</strong>'
        for text, kind in signals:
            tags_html += f'<span class="signal-tag signal-{kind}">{text}</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()

    # ---- Charts ----
    tab_main, tab_macd, tab_rsi = st.tabs(["📈 K线 + 均线 + 布林带", "📊 MACD", "📉 RSI"])

    _chart_config = {"displayModeBar": False, "scrollZoom": True}

    with tab_main:
        st.plotly_chart(build_main_chart(df), use_container_width=True, config=_chart_config)
    with tab_macd:
        st.plotly_chart(build_macd_chart(df), use_container_width=True, config=_chart_config)
    with tab_rsi:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True, config=_chart_config)

    # Support/Resistance
    if levels:
        items_html = ""
        for label, val in levels.items():
            if "压力" in label:
                css = "lv-r"
            elif "支撑" in label:
                css = "lv-s"
            else:
                css = "lv-p"
            items_html += f'<div class="level-item"><div class="level-label">{label}</div><div class="level-val {css}">{val}</div></div>'
        st.markdown(f'<div style="margin-top:0.2rem"><strong>关键价位</strong></div><div class="levels-grid">{items_html}</div>', unsafe_allow_html=True)

    st.divider()

    # ---- Multi-Agent Debate ----
    st.markdown("### 🎭 交易员辩论会")
    st.caption("4 位不同风格的交易员各抒己见，首席策略官综合裁决")

    data_prompt = build_prompt(stock_name, ts_code, df, fina, business, news_text)

    # Phase 1: 各交易员发言
    agent_opinions = []

    with st.status("交易员辩论进行中...", expanded=True) as debate_status:
        for agent in TRADER_AGENTS:
            st.write(f"{agent['emoji']} {agent['name']}正在分析...")
            opinion = run_single_agent(gemini_client, agent, data_prompt, cfg["gemini_model"])
            agent_opinions.append((agent, opinion))
            time.sleep(0.5)

        st.write("🎯 首席策略官正在综合裁决...")
        judge_stream = run_judge(gemini_client, stock_name, data_prompt, agent_opinions, cfg["gemini_model"])
        debate_status.update(label="辩论完毕 ✅", state="complete", expanded=False)

    # Display agent opinions in tabs
    agent_tabs = st.tabs([a["tab_label"] for a in TRADER_AGENTS])

    for i, (agent, opinion) in enumerate(agent_opinions):
        with agent_tabs[i]:
            stance_text, stance_class = extract_stance(opinion)
            st.markdown(f"""
            <div class="{agent['style']}">
                <div class="agent-header">
                    <span class="agent-emoji">{agent['emoji']}</span>
                    <span>{agent['name']}<br><small style="font-weight:400;color:#8892b0">{agent['desc']}</small></span>
                    <span class="agent-stance stance-{stance_class}">{stance_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(opinion)

    # Phase 2: Judge verdict
    st.divider()
    st.markdown("""
    <div class="verdict-title">🎯 首席策略官 · 最终裁决</div>
    """, unsafe_allow_html=True)

    if judge_stream:
        try:
            st.write_stream(
                (chunk.text for chunk in judge_stream if chunk.text)
            )
        except Exception as e:
            st.error(f"裁决生成失败: {e}")
    else:
        st.error("首席策略官调用失败")

    # ---- Financial summary ----
    with st.expander("📋 财务指标详情"):
        if fina:
            labels = {"roe": "ROE (%)", "grossprofit_margin": "毛利率 (%)",
                      "netprofit_yoy": "净利润同比 (%)", "revenue_yoy": "营收同比 (%)",
                      "debt_to_assets": "资产负债率 (%)", "eps": "每股收益 (元)"}
            cols = st.columns(2)
            for i, (k, label) in enumerate(labels.items()):
                v = fina.get(k)
                with cols[i % 2]:
                    val_str = f"{v:.2f}" if v is not None and str(v) != "nan" else "N/A"
                    st.metric(label, val_str)
        else:
            st.info("暂无财务数据")

    if news_text and enable_search:
        with st.expander("🌐 联网搜索结果"):
            st.markdown(news_text)

    # Debate raw data
    with st.expander("📝 查看完整辩论记录"):
        for agent, opinion in agent_opinions:
            st.markdown(f"**{agent['emoji']} {agent['name']}**")
            st.markdown(opinion)
            st.divider()


if __name__ == "__main__":
    main()
