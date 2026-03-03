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
    initial_sidebar_state="expanded",
)

# =====================================================================
# 样式
# =====================================================================
st.markdown("""
<style>
    /* 主容器 */
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .label { color: #8892b0; font-size: 0.8rem; margin-bottom: 0.3rem; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; }
    .metric-card .up { color: #ef4444; }
    .metric-card .down { color: #22c55e; }
    .metric-card .neutral { color: #e2e8f0; }

    /* 信号标签 */
    .signal-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .signal-bullish { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .signal-bearish { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .signal-neutral { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }

    /* 侧边栏标题 */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 1rem;
    }

    /* 分析报告区域 */
    .report-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.8;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
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
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor="#1e293b", title=""),
        yaxis2=dict(gridcolor="#1e293b", title=""),
        xaxis=dict(gridcolor="#1e293b"),
        xaxis2=dict(gridcolor="#1e293b"),
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
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center", font_size=10),
        yaxis=dict(gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
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
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        yaxis=dict(gridcolor="#1e293b", range=[0, 100]),
        xaxis=dict(gridcolor="#1e293b"),
        showlegend=False,
    )
    return fig


# =====================================================================
# 构建 Gemini 提示词
# =====================================================================
SYSTEM_PROMPT = """
你是一位在A股拥有20年实战经验的顶级交易员, 同时具备机构研究员的基本面分析能力。
投资哲学: "价值投机"——只做有真实业务支撑的主线标的, 同时利用技术面择时。

严格按以下框架输出:

## 一、多空力量评估 (打分制)
从5个维度各打1-10分:
- 趋势强度 | 量价配合 | 技术位置 | 基本面质量 | 消息面催化
汇总综合得分并给出偏多/偏空/中性判断。

## 二、趋势阶段判断
判断: 底部蓄势 / 突破启动 / 主升浪 / 高位震荡 / 顶部派发 / 下跌趋势 / 超跌反弹

## 三、交易策略
1. 操作方向: 买入/持有/减仓/观望
2. 关键价位: 止损位 / 目标位 / 加仓位
3. 执行方案: 分批操作建议
4. 时间维度: 短线/波段/中线

## 四、最大风险点
1-2个具体风险, 引用数据。

不要废话, 直接给结论, 每个判断必须引用具体数据。
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
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📈 A股智能分析系统</div>', unsafe_allow_html=True)
        st.caption("Gemini AI + 技术指标引擎 + 实时联网搜索")

        st.divider()

        stock_code = st.text_input(
            "股票代码",
            placeholder="输入6位代码，如 600519",
            max_chars=6,
            help="支持沪深A股，输入纯数字代码",
        )

        col1, col2 = st.columns(2)
        with col1:
            kline_days = st.selectbox("回溯天数", [30, 60, 90, 120], index=1)
        with col2:
            enable_search = st.toggle("联网搜索", value=True, help="使用Google Search获取最新消息")

        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

        st.divider()
        st.caption("⚠️ 分析仅供参考，不构成投资建议")
        st.caption(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Main area
    if not analyze_btn or not stock_code:
        # Landing page
        st.markdown("## 📊 A股智能分析系统")
        st.markdown("在左侧输入股票代码，点击 **开始分析** 获取 AI 深度研报。")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 技术面分析")
            st.markdown("MA / MACD / RSI / 布林带\n\n支撑压力位 · 量价背离检测")
        with col2:
            st.markdown("#### 🌐 实时消息面")
            st.markdown("Gemini + Google Search\n\n政策 · 公告 · 行业 · 机构动向")
        with col3:
            st.markdown("#### 🤖 AI 深度推理")
            st.markdown("多空评分 · 趋势判断\n\n交易策略 · 风险提示")
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

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">最新价</div>
            <div class="value {change_color}">{latest['close']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">涨跌幅</div>
            <div class="value {change_color}">{change_pct:+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        rsi_val = latest.get("RSI", 50)
        rsi_class = "up" if pd.notna(rsi_val) and rsi_val > 70 else ("down" if pd.notna(rsi_val) and rsi_val < 30 else "neutral")
        st.markdown(f"""<div class="metric-card">
            <div class="label">RSI(14)</div>
            <div class="value {rsi_class}">{rsi_val:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        vol_ratio = latest.get("VOL_RATIO", 1.0)
        vr_class = "up" if pd.notna(vol_ratio) and vol_ratio > 1.5 else ("down" if pd.notna(vol_ratio) and vol_ratio < 0.5 else "neutral")
        st.markdown(f"""<div class="metric-card">
            <div class="label">量比</div>
            <div class="value {vr_class}">{vol_ratio:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="label">成交额</div>
            <div class="value neutral">{latest['amount']/10000:.0f}万</div>
        </div>""", unsafe_allow_html=True)

    # Signal tags
    if signals:
        tags_html = ""
        for text, kind in signals:
            css = f"signal-{kind}"
            tags_html += f'<span class="signal-tag {css}">{text}</span>'
        st.markdown(f"**技术信号** {tags_html}", unsafe_allow_html=True)

    st.divider()

    # ---- Charts ----
    tab_main, tab_macd, tab_rsi = st.tabs(["📈 K线 + 均线 + 布林带", "📊 MACD", "📉 RSI"])

    with tab_main:
        st.plotly_chart(build_main_chart(df), use_container_width=True, config={"displayModeBar": False})
    with tab_macd:
        st.plotly_chart(build_macd_chart(df), use_container_width=True, config={"displayModeBar": False})
    with tab_rsi:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True, config={"displayModeBar": False})

    # Support/Resistance
    if levels:
        st.markdown("**关键价位**")
        lc = st.columns(len(levels))
        for i, (label, val) in enumerate(levels.items()):
            with lc[i]:
                color = "#ef4444" if "压力" in label else ("#22c55e" if "支撑" in label else "#94a3b8")
                st.markdown(f"<div style='text-align:center'><span style='color:#8892b0;font-size:0.75rem'>{label}</span><br>"
                            f"<span style='color:{color};font-weight:700;font-size:1.1rem'>{val}</span></div>",
                            unsafe_allow_html=True)

    st.divider()

    # ---- AI Analysis ----
    st.markdown("### 🤖 AI 深度分析报告")

    prompt = build_prompt(stock_name, ts_code, df, fina, business, news_text)

    try:
        response = gemini_client.models.generate_content_stream(
            model=cfg["gemini_model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=4096,
            ),
        )
        report = st.write_stream(
            (chunk.text for chunk in response if chunk.text)
        )
    except Exception as e:
        st.error(f"AI 分析失败: {e}")

    # ---- Financial summary ----
    with st.expander("📋 财务指标详情"):
        if fina:
            labels = {"roe": "ROE (%)", "grossprofit_margin": "毛利率 (%)",
                      "netprofit_yoy": "净利润同比 (%)", "revenue_yoy": "营收同比 (%)",
                      "debt_to_assets": "资产负债率 (%)", "eps": "每股收益 (元)"}
            cols = st.columns(3)
            for i, (k, label) in enumerate(labels.items()):
                v = fina.get(k)
                with cols[i % 3]:
                    val_str = f"{v:.2f}" if v is not None and str(v) != "nan" else "N/A"
                    st.metric(label, val_str)
        else:
            st.info("暂无财务数据")

    if news_text and enable_search:
        with st.expander("🌐 联网搜索结果"):
            st.markdown(news_text)


if __name__ == "__main__":
    main()
