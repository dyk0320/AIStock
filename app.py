# coding=utf-8
"""
A股智能分析系统 v4.2 - 多模型Fallback + 多Agent辩论
DeepSeek/Qwen/Gemini 多模型 + Tushare + AKShare + 大盘板块 + 风险 + 情绪
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
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from model_hub import get_config, ModelHub, init_clients

warnings.filterwarnings("ignore")

# =====================================================================
# 页面配置
# =====================================================================
st.set_page_config(
    page_title="A股智能分析系统 v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 样式
# =====================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .metrics-grid {
        display: grid; grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem; margin-bottom: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a; border-radius: 12px;
        padding: 1rem 0.6rem; text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 0.78rem; margin-bottom: 0.2rem; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; }
    .metric-card .up { color: #ef4444; }
    .metric-card .down { color: #22c55e; }
    .metric-card .neutral { color: #e2e8f0; }

    .levels-grid {
        display: grid; grid-template-columns: repeat(5, 1fr);
        gap: 0.4rem; text-align: center; margin: 0.5rem 0;
    }
    .level-item .level-label { color: #8892b0; font-size: 0.72rem; }
    .level-item .level-val { font-weight: 700; font-size: 1.05rem; }
    .level-item .lv-r { color: #ef4444; }
    .level-item .lv-s { color: #22c55e; }
    .level-item .lv-p { color: #94a3b8; }

    .signal-wrap { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
    .signal-tag {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; white-space: nowrap;
    }
    .signal-bullish { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .signal-bearish { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .signal-neutral { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }

    .sidebar-title {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        padding-bottom: 0.5rem; border-bottom: 2px solid #3b82f6; margin-bottom: 1rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .landing-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    .landing-card { background: #1e293b; border-radius: 12px; padding: 1.2rem; text-align: center; }
    .landing-card h4 { margin: 0.4rem 0; font-size: 1rem; }
    .landing-card p { color: #94a3b8; font-size: 0.85rem; margin: 0; }

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
    .agent-senti   .agent-header { background: linear-gradient(135deg, #1a1a2e, #1e293b); border-left: 3px solid #06b6d4; }

    .verdict-title {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 1.1rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.8rem;
    }

    /* 恐贪仪表盘 */
    .fg-gauge {
        display: flex; align-items: center; gap: 1rem;
        background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px;
        padding: 1rem 1.5rem; margin: 0.5rem 0;
    }
    .fg-score { font-size: 2.2rem; font-weight: 800; }
    .fg-label { font-size: 0.85rem; color: #8892b0; }
    .fg-bar { flex: 1; height: 12px; background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); border-radius: 6px; position: relative; }
    .fg-marker { position: absolute; top: -4px; width: 4px; height: 20px; background: white; border-radius: 2px; }
    .fg-components { display: flex; flex-wrap: wrap; gap: 0.3rem 0.8rem; margin-top: 0.4rem; }

    @media (max-width: 768px) {
        .block-container { padding: 0.8rem 0.6rem !important; }
        .metrics-grid { grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
        .metric-card { padding: 0.7rem 0.3rem; border-radius: 8px; }
        .metric-card .label { font-size: 0.68rem; }
        .metric-card .value { font-size: 1.1rem; }
        .levels-grid { grid-template-columns: repeat(3, 1fr); gap: 0.3rem; }
        .signal-tag { font-size: 0.7rem; padding: 0.2rem 0.55rem; }
        .landing-grid { grid-template-columns: 1fr; gap: 0.6rem; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        .modebar { display: none !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }
        .agent-header { padding: 0.6rem 0.8rem; font-size: 0.85rem; }
        .fg-gauge { flex-direction: column; text-align: center; padding: 0.8rem; }
        .fg-score { font-size: 1.8rem; }
    }
    @media (max-width: 480px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        .levels-grid { grid-template-columns: repeat(2, 1fr); }
    }
    [data-testid="column"] { display: flex; align-items: end; }
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# Floating sidebar toggle button - inject into parent document (mobile only)
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    var pd = window.parent.document;
    if (pd.getElementById('sidebar-fab')) return;
    var btn = pd.createElement('div');
    btn.id = 'sidebar-fab';
    btn.innerHTML = '⚙️';
    btn.style.cssText = 'position:fixed;bottom:24px;left:16px;z-index:999999;'
        + 'width:48px;height:48px;border-radius:50%;'
        + 'background:linear-gradient(135deg,#3b82f6,#6366f1);'
        + 'border:2px solid rgba(255,255,255,0.2);'
        + 'color:white;font-size:1.3rem;cursor:pointer;'
        + 'display:flex;align-items:center;justify-content:center;'
        + 'box-shadow:0 4px 15px rgba(59,130,246,0.4);'
        + 'transition:transform 0.2s;user-select:none;';
    btn.onmousedown = function(){ btn.style.transform='scale(0.9)'; };
    btn.onmouseup = function(){ btn.style.transform='scale(1)'; };
    btn.onclick = function() {
        var c = pd.querySelector('[data-testid="collapsedControl"]');
        if (c) { c.click(); return; }
        var x = pd.querySelector('[data-testid="stSidebarCollapseButton"] button');
        if (x) { x.click(); return; }
    };
    pd.body.appendChild(btn);
    // Auto-hide on desktop
    var mq = window.parent.matchMedia('(min-width: 769px)');
    function check(e) { btn.style.display = e.matches ? 'none' : 'flex'; }
    check(mq); mq.addEventListener('change', check);
})();
</script>
""", height=0)


# =====================================================================
# 配置 & 初始化
# =====================================================================

# =====================================================================
# 模块1: Tushare 数据获取
# =====================================================================
def build_ts_code(code: str) -> str:
    code = code.strip()
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


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
        return {k: f.get(k) for k in
                ["roe", "grossprofit_margin", "netprofit_yoy", "revenue_yoy", "debt_to_assets", "eps"]}
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
# 模块2: AKShare 增强数据 (资金流向 / 北向 / 融资融券 / 涨跌停)
# =====================================================================
@st.cache_data(ttl=600)
def fetch_akshare_data(stock_code: str):
    """一次性获取所有 AKShare 数据, 任一失败不影响主流程"""
    result = {"fund_flow": None, "north_flow": None, "margin": None,
              "limit_up_count": None, "limit_down_count": None}
    try:
        import akshare as ak
    except ImportError:
        return result

    # 个股资金流向
    try:
        mkt = "sh" if stock_code.startswith("6") else "sz"
        ff = ak.stock_individual_fund_flow(stock=stock_code, market=mkt)
        if ff is not None and not ff.empty:
            result["fund_flow"] = ff.tail(10)
    except Exception:
        pass

    # 北向资金
    try:
        nf = ak.stock_hsgt_hist_em(symbol="北向资金")
        if nf is not None and not nf.empty:
            result["north_flow"] = nf.tail(10)
    except Exception:
        pass

    # 融资融券
    try:
        end = datetime.datetime.now().strftime("%Y%m%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y%m%d")
        mg = ak.stock_margin_sse(start_date=start, end_date=end)
        if mg is not None and not mg.empty:
            result["margin"] = mg.tail(10)
    except Exception:
        pass

    # 涨停/跌停
    today = datetime.datetime.now().strftime("%Y%m%d")
    try:
        zt = ak.stock_zt_pool_em(date=today)
        result["limit_up_count"] = len(zt) if zt is not None else None
    except Exception:
        pass
    try:
        dt = ak.stock_zt_pool_dtgc_em(date=today)
        result["limit_down_count"] = len(dt) if dt is not None else None
    except Exception:
        pass

    return result


def _safe_col(df, keywords):
    """从 DataFrame 列名中模糊匹配含指定关键字的列"""
    for c in df.columns:
        if all(k in str(c) for k in keywords):
            return c
    return None


def format_akshare_text(ak_data: dict) -> str:
    """AKShare 数据 → Agent 可读文本"""
    lines = []

    # ---- 个股资金流向 ----
    ff = ak_data.get("fund_flow")
    if ff is not None and not ff.empty:
        lines.append("【个股资金流向(近5日)】")
        date_c = _safe_col(ff, ["日期"]) or ff.columns[0]
        main_c = _safe_col(ff, ["主力", "净"])
        for _, row in ff.tail(5).iterrows():
            d = str(row[date_c])[:10]
            m = f"{float(row[main_c]) / 10000:.1f}万" if main_c else "N/A"
            lines.append(f"  {d} 主力净流入:{m}")
    else:
        lines.append("【个股资金流向】暂无数据")

    # ---- 北向资金 ----
    nf = ak_data.get("north_flow")
    if nf is not None and not nf.empty:
        lines.append("\n【北向资金(近5日)】")
        date_c = _safe_col(nf, ["日期"]) or _safe_col(nf, ["date"]) or nf.columns[0]
        val_c = _safe_col(nf, ["当日", "流入"]) or _safe_col(nf, ["净", "买"])
        for _, row in nf.tail(5).iterrows():
            d = str(row[date_c])[:10]
            v = f"{float(row[val_c]):.2f}亿" if val_c else "N/A"
            lines.append(f"  {d} 北向净买入:{v}")
    else:
        lines.append("\n【北向资金】暂无数据")

    # ---- 融资融券 ----
    mg = ak_data.get("margin")
    if mg is not None and not mg.empty:
        lines.append("\n【融资融券(近期)】")
        rz_c = _safe_col(mg, ["融资", "余额"])
        try:
            latest_mg = mg.iloc[-1]
            if rz_c:
                rz_val = float(latest_mg[rz_c])
                lines.append(f"  融资余额: {rz_val / 1e8:.0f}亿")
                if len(mg) >= 5:
                    prev_mg = mg.iloc[-5]
                    delta = (rz_val - float(prev_mg[rz_c])) / 1e8
                    lines.append(f"  5日变化: {delta:+.1f}亿 ({'加杠杆' if delta > 0 else '去杠杆'})")
        except Exception:
            pass
    else:
        lines.append("\n【融资融券】暂无数据")

    # ---- 涨跌停 ----
    lu = ak_data.get("limit_up_count")
    ld = ak_data.get("limit_down_count")
    if lu is not None or ld is not None:
        lu_s, ld_s = str(lu) if lu is not None else "N/A", str(ld) if ld is not None else "N/A"
        lines.append(f"\n【今日涨跌停】涨停:{lu_s}只 | 跌停:{ld_s}只")
        if lu is not None and ld is not None and (lu + ld) > 0:
            ratio = lu / (lu + ld) * 100
            mood = ("极度狂热" if ratio > 90 else "偏多" if ratio > 65
                    else "均衡" if ratio > 35 else "偏空" if ratio > 10 else "极度恐慌")
            lines.append(f"  涨停占比:{ratio:.0f}% → 市场情绪:{mood}")

    return "\n".join(lines)


# =====================================================================
# 模块2a+: 实时分时数据 (三湖量化API)
# =====================================================================
import requests

@st.cache_data(ttl=60)
def fetch_realtime(stock_code: str, token: str) -> dict:
    """获取当日实时分时数据"""
    if not token:
        return {"error": "REALTIME_TOKEN not set"}
    url = (f"http://www.sanhulianghua.com:2008/v1/hsa_fenshi"
           f"?token={token}&code={stock_code}&all=1&simple=0")
    try:
        resp = requests.get(url, timeout=10)
        raw = resp.json()
        if raw.get("ret") != 200:
            return {"error": f"API ret={raw.get('ret')}: {raw.get('msg', '')}"}

        data_list = raw.get("data", [])
        if not data_list:
            return {"error": "data list empty (盘前无数据?)"}

        # Take the latest tick
        tick = data_list[-1]

        # Unit conversions: prices in 0.1分(0.001元), pcts in 0.001%
        def p(v):
            return round(v / 1000, 3) if v else 0.0
        def pct(v):
            return round(v / 1000, 3) if v is not None else 0.0

        result = {
            "name": raw.get("name", ""),
            "date": raw.get("date", ""),
            "time": tick.get("ShiJian", ""),
            "price": p(tick.get("JiaGe", 0)),
            "open": p(tick.get("KaiPan", 0)),
            "high": p(tick.get("ZuiGao", 0)),
            "low": p(tick.get("ZuiDi", 0)),
            "prev_close": p(tick.get("ZuoShou", 0)),
            "avg_price": p(tick.get("JunJia", 0)),
            "change_pct": pct(tick.get("ZhangFu", 0)),
            "change_speed": pct(tick.get("ZhangSu", 0)),
            "amplitude": pct(tick.get("ZhenFu", 0)),
            "volume": tick.get("ZongLiang", 0),        # 手
            "amount": tick.get("JinE", 0),              # 元
            "turnover": pct(tick.get("HuanShou", 0)),
            "vol_ratio": pct(tick.get("LiangBi", 0)),
            "wei_bi": pct(tick.get("WeiBi", 0)),
            "inner_vol": tick.get("NeiPan", 0),         # 手
            "outer_vol": tick.get("WaiPan", 0),         # 手
            "inner_outer_ratio": pct(tick.get("NeiWaiBi", 0)),
            "pe": pct(tick.get("ShiYingLv", 0)),
            "pb": pct(tick.get("ShiJingLv", 0)),
            "market_cap": tick.get("ShiZhi", 0),        # 万元
            "streak_days": tick.get("LianZhangTian", 0),
            "chg_3d": pct(tick.get("03RiZhangFu", 0)),
            "chg_5d": pct(tick.get("05RiZhangFu", 0)),
            "chg_10d": pct(tick.get("10RiZhangFu", 0)),
            "chg_20d": pct(tick.get("20RiZhangFu", 0)),
            "chg_60d": pct(tick.get("60RiZhangFu", 0)),
            "chg_1y": pct(tick.get("NianZhangFu", 0)),
            "ticks_count": len(data_list),
        }
        return result
    except requests.exceptions.Timeout:
        return {"error": "API timeout (10s)"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def format_realtime_text(rt: dict) -> str:
    """Format realtime data for LLM prompt"""
    if "error" in rt:
        return f"(实时数据不可用: {rt['error']})"

    amt_yi = rt['amount'] / 1e8 if rt['amount'] else 0
    cap_yi = rt['market_cap'] / 10000 if rt['market_cap'] else 0

    # Streak description
    streak = rt.get('streak_days', 0)
    if streak > 0:
        streak_str = f"连涨{streak}天"
    elif streak < 0:
        streak_str = f"连跌{abs(streak)}天"
    else:
        streak_str = "无连续涨跌"

    # Inner/outer pressure
    inner = rt.get('inner_vol', 0)
    outer = rt.get('outer_vol', 0)
    total = inner + outer
    if total > 0:
        buy_pct = outer / total * 100
        pressure = f"外盘{buy_pct:.0f}%({'多方主导' if buy_pct > 55 else '空方主导' if buy_pct < 45 else '多空均衡'})"
    else:
        pressure = "N/A"

    return f"""【实时盘口数据】({rt['date']} {rt['time']}更新, 共{rt['ticks_count']}个tick)
现价: {rt['price']:.3f} | 涨幅: {rt['change_pct']:+.3f}% | 涨速: {rt['change_speed']:+.3f}%
开盘: {rt['open']:.3f} | 最高: {rt['high']:.3f} | 最低: {rt['low']:.3f} | 昨收: {rt['prev_close']:.3f}
均价: {rt['avg_price']:.3f} | 振幅: {rt['amplitude']:.3f}%
成交量: {rt['volume']}手 | 成交额: {amt_yi:.2f}亿 | 换手率: {rt['turnover']:.3f}%
量比: {rt['vol_ratio']:.3f} | 委比: {rt['wei_bi']:+.3f}% | {pressure}
PE: {rt['pe']:.2f} | PB: {rt['pb']:.2f} | 总市值: {cap_yi:.1f}亿
{streak_str} | 3日:{rt['chg_3d']:+.3f}% | 5日:{rt['chg_5d']:+.3f}% | 10日:{rt['chg_10d']:+.3f}% | 20日:{rt['chg_20d']:+.3f}% | 60日:{rt['chg_60d']:+.3f}% | 年:{rt['chg_1y']:+.3f}%"""


# =====================================================================
# 模块2b: 大盘 & 板块上下文 (Tushare + AKShare)
# =====================================================================
INDEX_MAP = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
}


@st.cache_data(ttl=300)
def fetch_index_kline(_pro, index_code: str, days: int = 30):
    """获取指数K线"""
    end = datetime.datetime.now().strftime("%Y%m%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=days + 10)).strftime("%Y%m%d")
    try:
        df = _pro.index_daily(ts_code=index_code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            return df.sort_values("trade_date").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_stock_sector(_pro, ts_code: str):
    """获取个股所属行业 (申万行业)"""
    try:
        df = _pro.stock_basic(ts_code=ts_code, fields="ts_code,industry")
        if not df.empty:
            return df.iloc[0].get("industry", "未知")
    except Exception:
        pass
    return "未知"


@st.cache_data(ttl=600)
def fetch_sector_data(sector_name: str):
    """获取板块指数数据 (AKShare 东方财富板块)"""
    try:
        import akshare as ak
        df = ak.stock_board_industry_hist_em(
            symbol=sector_name,
            period="日k",
            start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y%m%d"),
            end_date=datetime.datetime.now().strftime("%Y%m%d"),
            adjust=""
        )
        if df is not None and not df.empty:
            return df.sort_values("日期").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_sector_ranking():
    """获取今日板块涨幅排行 (AKShare)"""
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            chg_col = _safe_col(df, ["涨跌幅"]) or _safe_col(df, ["涨幅"])
            name_col = _safe_col(df, ["板块名称"]) or df.columns[0]
            if chg_col:
                df[chg_col] = pd.to_numeric(df[chg_col], errors="coerce")
                df = df.sort_values(chg_col, ascending=False)
                top5 = df.head(5)[[name_col, chg_col]].values.tolist()
                bot5 = df.tail(5)[[name_col, chg_col]].values.tolist()
                return {"top5": top5, "bottom5": bot5}
    except Exception:
        pass
    return None


class MarketContext:
    """大盘环境分析: 判断系统性风险, 生成市场全景"""

    @staticmethod
    def analyze_indices(pro) -> dict:
        """分析三大指数, 返回结构化结果"""
        result = {"indices": {}, "systemic_risk": False, "risk_level": "正常",
                  "crash_alert": False, "summary": ""}

        crash_count = 0
        weak_count = 0

        for code, name in INDEX_MAP.items():
            idx_df = fetch_index_kline(pro, code, 30)
            if idx_df.empty:
                result["indices"][name] = {"status": "数据缺失"}
                continue

            latest = idx_df.iloc[-1]
            pct = latest.get("pct_chg", 0) or 0

            # 计算5日/10日/20日涨跌幅
            chg_5 = chg_10 = chg_20 = 0
            close_now = latest["close"]
            if len(idx_df) >= 6:
                chg_5 = (close_now / idx_df.iloc[-6]["close"] - 1) * 100
            if len(idx_df) >= 11:
                chg_10 = (close_now / idx_df.iloc[-11]["close"] - 1) * 100
            if len(idx_df) >= 21:
                chg_20 = (close_now / idx_df.iloc[-21]["close"] - 1) * 100

            # 20日波动率
            if len(idx_df) >= 20:
                vol_20 = (idx_df["pct_chg"].tail(20) / 100).std() * np.sqrt(252) * 100
            else:
                vol_20 = 0

            # 判断大跌
            is_crash_today = pct <= -2.5
            is_crash_week = chg_5 <= -5
            is_bear_trend = chg_20 <= -10

            if is_crash_today:
                crash_count += 1
            if chg_5 < -3:
                weak_count += 1

            result["indices"][name] = {
                "close": round(close_now, 2),
                "today_chg": round(pct, 2),
                "chg_5d": round(chg_5, 2),
                "chg_10d": round(chg_10, 2),
                "chg_20d": round(chg_20, 2),
                "vol_20d": round(vol_20, 1),
                "crash_today": is_crash_today,
                "crash_week": is_crash_week,
                "bear_trend": is_bear_trend,
            }

        # 系统性风险判断
        if crash_count >= 2:
            result["systemic_risk"] = True
            result["crash_alert"] = True
            result["risk_level"] = "⚠️ 大盘暴跌"
            result["summary"] = ("三大指数多数暴跌! 系统性风险爆发, "
                                 "个股分析需大幅下调预期, 无论基本面多好都可能被拖累。"
                                 "优先考虑防守: 降低仓位、设紧止损、回避高beta标的。")
        elif weak_count >= 2:
            result["systemic_risk"] = True
            result["risk_level"] = "⚠️ 大盘走弱"
            result["summary"] = ("大盘近期明显走弱, 赚钱效应差。"
                                 "个股操作需更加谨慎, 优选逆势强势股或防御板块。")
        elif any(v.get("bear_trend") for v in result["indices"].values() if isinstance(v, dict)):
            result["systemic_risk"] = True
            result["risk_level"] = "🔴 中期下跌趋势"
            result["summary"] = "指数处于中期下跌趋势(20日跌幅>10%), 反弹做空思路, 控制仓位。"
        else:
            all_positive = all(v.get("chg_5d", 0) > 0
                              for v in result["indices"].values() if isinstance(v, dict) and "chg_5d" in v)
            if all_positive:
                result["risk_level"] = "🟢 大盘健康"
                result["summary"] = "三大指数近期走势健康, 市场环境支持个股操作。"
            else:
                result["risk_level"] = "🟡 大盘分化"
                result["summary"] = "指数间走势分化, 需关注个股所在板块是否受青睐。"

        return result

    @staticmethod
    def analyze_sector(pro, ts_code: str) -> dict:
        """分析个股所属板块及板块排名"""
        sector_name = fetch_stock_sector(pro, ts_code)
        result = {"name": sector_name, "rank_info": None, "sector_chg": None, "sector_trend": ""}

        # 板块排行
        ranking = fetch_sector_ranking()
        if ranking:
            result["rank_info"] = ranking
            # 判断个股板块在排行中的位置
            all_items = ranking["top5"] + ranking["bottom5"]
            for item in all_items:
                if sector_name and sector_name in str(item[0]):
                    result["sector_chg"] = item[1]
                    break

        # 板块历史数据
        if sector_name and sector_name != "未知":
            sdf = fetch_sector_data(sector_name)
            if not sdf.empty:
                close_col = _safe_col(sdf, ["收盘"]) or _safe_col(sdf, ["close"])
                if close_col and len(sdf) >= 5:
                    try:
                        chg_5 = (float(sdf.iloc[-1][close_col]) / float(sdf.iloc[-5][close_col]) - 1) * 100
                        result["sector_trend"] = f"{sector_name}板块5日涨跌:{chg_5:+.2f}%"
                        if chg_5 > 3:
                            result["sector_trend"] += " (板块强势, 有资金关注)"
                        elif chg_5 < -3:
                            result["sector_trend"] += " (板块走弱, 可能拖累个股)"
                    except Exception:
                        pass

        return result

    @staticmethod
    def format_text(mkt: dict, sector: dict) -> str:
        """生成大盘+板块的文本摘要"""
        lines = [f"{'='*50}", f"【大盘环境】 {mkt['risk_level']}", mkt["summary"], ""]

        for name, data in mkt["indices"].items():
            if not isinstance(data, dict) or "close" not in data:
                lines.append(f"  {name}: 数据缺失")
                continue
            alert = " 🚨暴跌!" if data.get("crash_today") else ""
            lines.append(f"  {name}: {data['close']}  今日:{data['today_chg']:+.2f}%{alert}  "
                         f"5日:{data['chg_5d']:+.2f}%  20日:{data['chg_20d']:+.2f}%  波动:{data['vol_20d']}%")

        lines.append(f"\n【所属板块】{sector['name']}")
        if sector["sector_trend"]:
            lines.append(f"  {sector['sector_trend']}")

        if sector["rank_info"]:
            top5 = sector["rank_info"]["top5"]
            bot5 = sector["rank_info"]["bottom5"]
            lines.append("  今日涨幅前5板块: " + " | ".join(
                f"{n}({float(c):+.2f}%)" for n, c in top5))
            lines.append("  今日跌幅前5板块: " + " | ".join(
                f"{n}({float(c):+.2f}%)" for n, c in bot5))

        if mkt["systemic_risk"]:
            lines.append(f"\n⚠️ 系统性风险警告: {mkt['summary']}")
            lines.append("所有交易员必须将大盘环境作为分析的首要考量! "
                         "大盘暴跌时, 个股分析结论需相应调整:")
            lines.append("  - 看多信心应大幅下调")
            lines.append("  - 止损位应收紧")
            lines.append("  - 仓位建议应更保守")
            lines.append("  - 优先考虑是否应该空仓观望")

        lines.append(f"{'='*50}")
        return "\n".join(lines)


# =====================================================================
# 模块3: 风险指标引擎  (VaR / CVaR / MDD / Sharpe / 波动率)
# =====================================================================
class RiskEngine:

    @staticmethod
    def calc_all(df):
        if len(df) < 10:
            return {}
        returns = df["pct_chg"].dropna() / 100
        r = {}

        # VaR
        r["var_95"] = round(np.percentile(returns, 5) * 100, 2)
        r["var_99"] = round(np.percentile(returns, 1) * 100, 2)

        # CVaR (Expected Shortfall)
        var95 = np.percentile(returns, 5)
        tail = returns[returns <= var95]
        r["cvar_95"] = round(tail.mean() * 100, 2) if len(tail) > 0 else r["var_95"]

        # 最大回撤
        cum = (1 + returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        r["max_drawdown"] = round(dd.min() * 100, 2)

        # 年化波动率
        r["volatility"] = round(returns.std() * np.sqrt(252) * 100, 2)

        # Sharpe (RF=2%)
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        r["sharpe"] = round((ann_ret - 0.02) / ann_vol, 2) if ann_vol > 0 else 0

        # 短/长波动比
        if len(returns) >= 20:
            v5 = returns.tail(5).std() * np.sqrt(252) * 100
            v20 = returns.tail(20).std() * np.sqrt(252) * 100
            r["vol_ratio"] = round(v5 / v20, 2) if v20 > 0 else 1.0
        else:
            r["vol_ratio"] = 1.0

        return r

    @staticmethod
    def format_text(risk: dict) -> str:
        if not risk:
            return "【风险指标】数据不足"
        vr = risk["vol_ratio"]
        vr_label = "波动放大⚠" if vr > 1.3 else "波动收敛" if vr < 0.7 else "正常"
        return (
            f"【风险指标】\n"
            f"  VaR(95%): {risk['var_95']}% | VaR(99%): {risk['var_99']}%\n"
            f"  CVaR(95%): {risk['cvar_95']}% | 最大回撤: {risk['max_drawdown']}%\n"
            f"  年化波动率: {risk['volatility']}% | Sharpe: {risk['sharpe']}\n"
            f"  短期/长期波动比: {vr} ({vr_label})"
        )


# =====================================================================
# 模块4: 恐贪指数 & 市场状态
# =====================================================================
class MarketSentiment:

    @staticmethod
    def calc_fear_greed(df, ak_data: dict) -> dict:
        """综合恐贪指数 0=极度恐惧 100=极度贪婪"""
        scores = {}

        # 1) RSI
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 50
        scores["RSI动量"] = int(min(max(rsi, 0), 100))

        # 2) 波动率 (高波=恐惧)
        rets = df["pct_chg"].dropna() / 100
        if len(rets) >= 20:
            vol20 = rets.tail(20).std() * np.sqrt(252)
            scores["波动率"] = int(max(0, min(100, 100 - vol20 * 300)))
        else:
            scores["波动率"] = 50

        # 3) 价格位置 (20日高低)
        if len(df) >= 20:
            r20 = df.tail(20)
            h, l = r20["high"].max(), r20["low"].min()
            scores["价格位置"] = int((df["close"].iloc[-1] - l) / (h - l) * 100) if h != l else 50
        else:
            scores["价格位置"] = 50

        # 4) 量能
        vr = df["VOL_RATIO"].iloc[-1] if "VOL_RATIO" in df.columns and pd.notna(df["VOL_RATIO"].iloc[-1]) else 1
        scores["量能水平"] = int(min(100, max(0, (vr - 0.5) / 2 * 100)))

        # 5) 涨跌停比
        lu, ld = ak_data.get("limit_up_count"), ak_data.get("limit_down_count")
        if lu is not None and ld is not None and (lu + ld) > 0:
            scores["涨跌停比"] = int(lu / (lu + ld) * 100)
        else:
            scores["涨跌停比"] = 50

        # 6) 均线趋势
        if all(f"MA{p}" in df.columns for p in [5, 10, 20]):
            lat = df.iloc[-1]
            cnt = sum(1 for p in [5, 10, 20] if pd.notna(lat.get(f"MA{p}")) and lat["close"] > lat[f"MA{p}"])
            scores["均线趋势"] = int(cnt / 3 * 100)
        else:
            scores["均线趋势"] = 50

        composite = int(sum(scores.values()) / len(scores))
        composite = min(100, max(0, composite))
        label = ("极度贪婪" if composite >= 80 else "贪婪" if composite >= 60
                 else "中性" if composite >= 40 else "恐惧" if composite >= 20 else "极度恐惧")
        return {"score": composite, "label": label, "components": scores}

    @staticmethod
    def detect_regime(df) -> dict:
        """简易市场状态判断 (无需 hmmlearn)"""
        if len(df) < 20:
            return {"regime": "数据不足", "detail": "", "mean_ret_20d": 0, "vol_20d": 0}
        rets = df["pct_chg"].tail(20) / 100
        mu, sigma = rets.mean(), rets.std()
        if mu > 0.005 and sigma < 0.025:
            regime, detail = "🟢 温和上涨", "趋势稳定向上, 波动可控, 适合趋势跟踪"
        elif mu > 0.005:
            regime, detail = "🟡 剧烈上涨", "涨势猛但波动大, 注意追高风险"
        elif mu < -0.005 and sigma < 0.025:
            regime, detail = "🔴 阴跌模式", "持续下跌, 空头趋势明确, 观望为主"
        elif mu < -0.005:
            regime, detail = "⚫ 恐慌抛售", "暴跌高波动, 可能酝酿超跌反弹"
        elif sigma < 0.015:
            regime, detail = "⚪ 窄幅整理", "方向不明, 等待突破"
        else:
            regime, detail = "🟡 宽幅震荡", "波动大无方向, 适合高抛低吸"
        return {"regime": regime, "detail": detail,
                "mean_ret_20d": round(mu * 100, 3), "vol_20d": round(sigma * 100, 3)}

    @staticmethod
    def format_text(fg: dict, regime: dict) -> str:
        parts = [f"【情绪与市场状态】",
                 f"  恐贪指数: {fg['score']}/100 ({fg['label']})"]
        for k, v in fg["components"].items():
            parts.append(f"    {k}: {v}/100")
        parts += [f"  市场状态: {regime['regime']}",
                  f"    {regime['detail']}",
                  f"    20日均收益:{regime['mean_ret_20d']}% | 20日波动:{regime['vol_20d']}%"]
        return "\n".join(parts)


# =====================================================================
# 模块5: 技术指标引擎
# =====================================================================
class TechEngine:

    @staticmethod
    def calc_all(df, ma_periods=(5, 10, 20), rsi_period=14, boll_period=20, boll_std=2):
        for p in ma_periods:
            df[f"MA{p}"] = df["close"].rolling(window=p).mean()
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = 2 * (df["DIF"] - df["DEA"])
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
        loss = (-delta).clip(lower=0).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["BOLL_MID"] = df["close"].rolling(window=boll_period).mean()
        std = df["close"].rolling(window=boll_period).std()
        df["BOLL_UP"] = df["BOLL_MID"] + boll_std * std
        df["BOLL_DN"] = df["BOLL_MID"] - boll_std * std
        df["VOL_MA5"] = df["vol"].rolling(window=5).mean()
        df["VOL_RATIO"] = df["vol"] / df["VOL_MA5"].replace(0, np.nan)
        return df

    @staticmethod
    def get_signals(df):
        if len(df) < 3:
            return []
        latest, prev = df.iloc[-1], df.iloc[-2]
        signals = []
        if latest["DIF"] > latest["DEA"] and prev["DIF"] <= prev["DEA"]:
            signals.append(("MACD 金叉", "bullish"))
        elif latest["DIF"] < latest["DEA"] and prev["DIF"] >= prev["DEA"]:
            signals.append(("MACD 死叉", "bearish"))
        elif latest["DIF"] > latest["DEA"]:
            signals.append(("MACD 多头", "bullish"))
        else:
            signals.append(("MACD 空头", "bearish"))
        rsi = latest.get("RSI", 50)
        if pd.notna(rsi):
            if rsi > 80: signals.append((f"RSI {rsi:.0f} 超买", "bearish"))
            elif rsi < 20: signals.append((f"RSI {rsi:.0f} 超卖", "bullish"))
        bu, bd = latest.get("BOLL_UP"), latest.get("BOLL_DN")
        if pd.notna(bu) and latest["close"] > bu:
            signals.append(("突破布林上轨", "bearish"))
        elif pd.notna(bd) and latest["close"] < bd:
            signals.append(("跌破布林下轨", "bullish"))
        ma_vals = [latest.get(f"MA{p}") for p in (5, 10, 20)]
        ma_vals = [v for v in ma_vals if pd.notna(v)]
        if len(ma_vals) >= 3:
            if ma_vals[0] > ma_vals[1] > ma_vals[2]: signals.append(("均线多头排列", "bullish"))
            elif ma_vals[0] < ma_vals[1] < ma_vals[2]: signals.append(("均线空头排列", "bearish"))
        if len(df) >= 10:
            r10 = df.tail(10)
            p_t = r10["close"].iloc[-1] - r10["close"].iloc[0]
            v_t = r10["vol"].iloc[-1] - r10["vol"].iloc[0]
            if p_t > 0 and v_t < 0: signals.append(("顶背离", "bearish"))
            elif p_t < 0 and v_t > 0: signals.append(("底背离", "bullish"))
        return signals

    @staticmethod
    def support_resistance(df):
        r20 = df.tail(20)
        if r20.empty:
            return {}
        c, h, l = r20["close"].iloc[-1], r20["high"].max(), r20["low"].min()
        pv = (h + l + c) / 3
        return {"强压力": round(pv + (h - l), 2), "弱压力": round(2 * pv - l, 2),
                "枢轴位": round(pv, 2), "弱支撑": round(2 * pv - h, 2),
                "强支撑": round(pv - (h - l), 2)}


# =====================================================================
# 模块6: 图表
# =====================================================================
def build_main_chart(df):
    d = df.tail(30).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=d["trade_date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#22c55e", decreasing_fillcolor="#22c55e", name="K线"), row=1, col=1)
    for ma, color in {"MA5": "#f59e0b", "MA10": "#3b82f6", "MA20": "#a855f7"}.items():
        if ma in d.columns:
            fig.add_trace(go.Scatter(x=d["trade_date"], y=d[ma],
                                     line=dict(width=1.2, color=color), name=ma), row=1, col=1)
    if "BOLL_UP" in d.columns:
        fig.add_trace(go.Scatter(x=d["trade_date"], y=d["BOLL_UP"],
                                  line=dict(width=0.8, color="rgba(100,100,100,0.5)", dash="dot"),
                                  name="布林上", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["trade_date"], y=d["BOLL_DN"],
                                  line=dict(width=0.8, color="rgba(100,100,100,0.5)", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(100,100,100,0.05)",
                                  name="布林下", showlegend=False), row=1, col=1)
    vc = ["#ef4444" if c >= o else "#22c55e" for c, o in zip(d["close"], d["open"])]
    fig.add_trace(go.Bar(x=d["trade_date"], y=d["vol"], marker_color=vc, name="成交量", opacity=0.7), row=2, col=1)
    fig.update_layout(height=480, margin=dict(l=5, r=5, t=30, b=5),
                       plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                       font=dict(color="#94a3b8", size=11),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
                       xaxis_rangeslider_visible=False, dragmode="pan",
                       yaxis=dict(gridcolor="#1e293b"), yaxis2=dict(gridcolor="#1e293b"),
                       xaxis=dict(gridcolor="#1e293b"), xaxis2=dict(gridcolor="#1e293b"))
    return fig


def build_macd_chart(df):
    d = df.tail(30).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["trade_date"], y=d["MACD_HIST"],
                          marker_color=["#ef4444" if v >= 0 else "#22c55e" for v in d["MACD_HIST"]],
                          name="MACD柱", opacity=0.7))
    fig.add_trace(go.Scatter(x=d["trade_date"], y=d["DIF"], line=dict(width=1.5, color="#f59e0b"), name="DIF"))
    fig.add_trace(go.Scatter(x=d["trade_date"], y=d["DEA"], line=dict(width=1.5, color="#3b82f6"), name="DEA"))
    fig.update_layout(height=220, margin=dict(l=5, r=5, t=10, b=5),
                       plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#94a3b8", size=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center", font_size=10),
                       yaxis=dict(gridcolor="#1e293b"), xaxis=dict(gridcolor="#1e293b"), dragmode="pan")
    return fig


def build_rsi_chart(df):
    d = df.tail(30).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["trade_date"], y=d["RSI"], line=dict(width=2, color="#a855f7"),
                              name="RSI(14)", fill="tozeroy", fillcolor="rgba(168,85,247,0.08)"))
    for lv, c, lb in [(80, "#ef4444", "超买"), (20, "#22c55e", "超卖")]:
        fig.add_hline(y=lv, line_dash="dash", line_color=c, opacity=0.5,
                       annotation_text=lb, annotation_font_color=c)
    fig.add_hline(y=50, line_dash="dot", line_color="#475569", opacity=0.3)
    fig.update_layout(height=200, margin=dict(l=5, r=5, t=10, b=5),
                       plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#94a3b8", size=10),
                       yaxis=dict(gridcolor="#1e293b", range=[0, 100]), xaxis=dict(gridcolor="#1e293b"),
                       showlegend=False, dragmode="pan")
    return fig


# =====================================================================
# 模块7: 多Agent + CoT 系统提示词
# =====================================================================
COT_PREFIX = """重要: 你必须遵循"思维链"分析法。先列关键数据, 再逐步推理, 最后结论。
格式: 数据观察 → 逻辑推理 → 结论。不允许跳过推理直接给结论。

你收到的数据包括: 大盘三大指数走势及系统性风险判断、个股所属板块表现及板块排名、
K线走势、技术指标、基本面、资金流向(主力/散户/北向)、融资融券、
涨跌停统计、风险指标(VaR/CVaR/最大回撤/Sharpe)、恐贪指数、市场状态。

⚠️ 核心原则: 大盘环境是一切分析的前提!
- 如果大盘暴跌/系统性风险爆发, 必须大幅调整个股结论
- 如果板块整体走弱, 个股逆势上涨的概率要打折
- 如果大盘健康+板块强势, 个股分析可以更积极
请充分利用所有维度数据, 特别是大盘和板块上下文。\n\n"""

TRADER_AGENTS = [
    {
        "id": "trend", "name": "趋势猎手", "emoji": "🐂",
        "style": "agent-trend", "tab_label": "🐂 趋势猎手",
        "desc": "趋势跟踪 · 只做主升浪",
        "system_prompt": COT_PREFIX + """你是"趋势猎手", 激进趋势交易员, 15年A股实战。
核心信仰: 趋势是唯一的朋友, 均线多头排列就是印钞机。

重点关注: MA排列、MACD方向力度、成交量趋势、北向资金方向、市场状态
你最看重: 趋势强度+量价配合+资金流向共振
你会参考: 恐贪指数判断阶段, 波动率判断策略适配度

输出格式:
## 数据观察 (3-5个关键数据)
## 推理过程 (数据→结论的逻辑链)
## 立场: [看多/看空/中性] — 一句话理由
## 操作建议: 方向 + 价位 + 时间维度
## 信心指数: X/10

语气: 果断自信, 交易员口吻。300字以内。""",
    },
    {
        "id": "value", "name": "价值守卫", "emoji": "🛡️",
        "style": "agent-value", "tab_label": "🛡️ 价值守卫",
        "desc": "基本面为王 · 安全边际",
        "system_prompt": COT_PREFIX + """你是"价值守卫", 保守价值投资者, 崇尚格雷厄姆。
核心信仰: 好公司+好价格=好投资, ROE才是真相。

重点关注: ROE、毛利率、利润增速、营收增速、负债率、EPS
额外看: 融资余额(杠杆泡沫)、VaR/最大回撤(安全边际)、Sharpe(性价比)
你质疑: 题材炒作、高波低Sharpe标的

输出格式:
## 数据观察 (3-5个关键财务+风险数据)
## 推理过程 (数据→结论)
## 立场: [看多/看空/中性] — 一句话理由
## 操作建议: 方向 + 安全边际价位 + 持有周期
## 信心指数: X/10

语气: 沉稳严谨, 常提风险。300字以内。""",
    },
    {
        "id": "contra", "name": "魔鬼代言人", "emoji": "😈",
        "style": "agent-contra", "tab_label": "😈 魔鬼代言人",
        "desc": "专业唱反调 · 找致命盲点",
        "system_prompt": COT_PREFIX + """你是"魔鬼代言人"(Devil's Advocate), 唯一使命是挑战共识, 找致命盲点。

核心原则: 数据看多→你找暴跌隐患; 数据看空→你找反转催化剂。永远站在共识对面。

重点关注: RSI极端值、量价背离、恐贪指数极端区域、波动率异常、
北向与主力方向不一致、融资余额急变(泡沫/踩踏)
你擅长: 识别"黑天鹅"前兆, 群体性情绪中的反向信号

输出格式:
## 市场共识 (一句话概括大多数人看法)
## 致命盲点 (3条, 每条引用数据, 说明市场为什么可能错)
## 反向立场: [看多/看空] — 与共识相反
## 触发条件: 什么情况下反向判断应验
## 信心指数: X/10

语气: 犀利挑衅, 不留情面。300字以内。""",
    },
    {
        "id": "swing", "name": "短线游侠", "emoji": "⚡",
        "style": "agent-swing", "tab_label": "⚡ 短线游侠",
        "desc": "快进快出 · 波段为王",
        "system_prompt": COT_PREFIX + """你是"短线游侠", 波段交易高手, 捕捉1-5天波动。
核心信仰: 不预测方向, 只捕捉波动, 风险收益比和纪律是一切。

重点关注: 日K形态、支撑压力位、RSI/布林带极端位置、量比异动
额外看: 恐贪(择时)、VaR(止损幅度)、波动率(仓位)、主力净流入(跟庄)、涨跌停(热度)

输出格式:
## 数据扫描 (3-5个精确到数字的观察)
## 推理链 (为什么指向你的结论)
## 立场: [做多/做空/观望]
## 精确方案: 入场价/止损价/止盈价/仓位%/持有天数
## 信心指数: X/10

语气: 精准干脆, 数字说话。300字以内。""",
    },
    {
        "id": "senti", "name": "情绪猎手", "emoji": "🧠",
        "style": "agent-senti", "tab_label": "🧠 情绪猎手",
        "desc": "资金流向 · 情绪周期 · 行为金融",
        "system_prompt": COT_PREFIX + """你是"情绪猎手", 行为金融专家, 专门分析市场情绪和资金行为。
核心信仰: 短期由情绪驱动, 资金流向是情绪最真实表达。散户在极端时刻永远是错的。

你的独特视角:
- 恐贪指数: >80或<20是反转信号
- 北向资金: "聪明钱", 连续流入/出确认趋势
- 主力vs散户: 主力净入+散户净出=筹码集中=启动前兆
- 融资余额: 快速增加=散户加杠杆=风险; 下降后企稳=底部
- 涨跌停: 涨停暴增=狂热末期; 跌停暴增=恐慌末期
- 量比/换手率: 突然放量=有人在行动
- 波动率: 低→高转折=变盘信号

输出格式:
## 情绪画像 (当前处于情绪周期哪个阶段)
## 资金行为解读 (3条: 北向/主力/融资/散户分别在做什么)
## 立场: [看多/看空/中性] — 情绪驱动的判断
## 关键转折信号: 什么情绪变化会改变判断
## 信心指数: X/10

语气: 洞察人性, 冷静分析群体行为。300字以内。""",
    },
]

JUDGE_SYSTEM_PROMPT = """你是"首席策略官", 主持5位交易员辩论会并做出最终裁决。

⚠️ 最高优先级: 大盘环境判断!
- 大盘暴跌/系统性风险 → 首先评估是否应"空仓观望", 仓位建议必须大幅下调
- 板块走弱 → 个股逆势概率要打折
- 大盘+板块都健康 → 可以给出较积极的操作建议

5位交易员:
- 趋势猎手🐂: 趋势跟踪派
- 价值守卫🛡️: 价值投资派
- 魔鬼代言人😈: 专业唱反调
- 短线游侠⚡: 波段交易派
- 情绪猎手🧠: 资金流向+行为金融派

裁决框架:

## 零、大盘环境研判
当前大盘属于: A)上涨期→积极 B)震荡期→精选控仓 C)下跌/暴跌→防守或空仓
大盘+板块对本次分析的影响?

## 一、观点交锋
核心分歧, 谁一致/对立, 魔鬼代言人的警告是否有理。

## 二、综合评分 (已计入大盘和板块)
| 维度 | 评分(1-10) | 依据 |
|------|-----------|------|
| 大盘环境 | | |
| 板块强弱 | | |
| 趋势强度 | | |
| 量价配合 | | |
| 基本面质量 | | |
| 资金面支撑 | | |
| 情绪面位置 | | |
| 风险收益比 | | |
| **综合** | | |

## 三、最终裁决
1. **操作方向**: 强烈买入/买入/持有/减仓/强烈卖出/空仓观望
2. **目标仓位**: %(大盘差时上限要低)
3. **关键价位**: 止损/目标/加仓
4. **执行节奏**: 分批方案
5. **时间维度**: 短线/波段/中线
6. **仓位上限**: 综合VaR+波动率+大盘环境

## 四、风险矩阵
| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 大盘系统性 | | | |
| 板块轮动 | | | |
| 个股特有 | | | |

大盘暴跌时最重要的结论可能就是"空仓观望"。
"""


# =====================================================================
# 模块8: 数据 Prompt 构建 (增强版)
# =====================================================================
def build_prompt(stock_name, ts_code, df, fina, business, news_text,
                 risk_text, sentiment_text, akshare_text, market_text="",
                 realtime_text=""):
    latest = df.iloc[-1]
    kline_lines = []
    for _, r in df.tail(15).iloc[::-1].iterrows():
        kline_lines.append(f"{r['trade_date']} O:{r['open']:.2f} H:{r['high']:.2f} "
                           f"L:{r['low']:.2f} C:{r['close']:.2f} {r['pct_chg']:+.2f}%")

    fina_parts = []
    for k, label in {"roe": "ROE", "grossprofit_margin": "毛利率", "netprofit_yoy": "净利增长",
                      "revenue_yoy": "营收增长", "debt_to_assets": "负债率", "eps": "EPS"}.items():
        v = fina.get(k)
        fina_parts.append(f"{label}:{v:.2f}" if v is not None and str(v) != "nan" else f"{label}:N/A")

    tech_lines = []
    for p in (5, 10, 20):
        v = latest.get(f"MA{p}")
        if pd.notna(v):
            tech_lines.append(f"MA{p}={v:.2f}(价{'上' if latest['close'] > v else '下'}方)")

    rsi_val = latest.get("RSI", 50)
    if pd.isna(rsi_val):
        rsi_val = 50
    levels = TechEngine.support_resistance(df)
    level_text = " | ".join(f"{k}:{v}" for k, v in levels.items()) if levels else "N/A"

    return f"""
{market_text}

===== 标的: {stock_name} ({ts_code}) =====
现价: {latest['close']:.2f} | 涨跌: {latest['pct_chg']:+.2f}%

{realtime_text}

【公司概况】{business}
【财务指标】{' | '.join(fina_parts)}

【K线(近15日)】
{chr(10).join(kline_lines)}

【技术指标】
均线: {' | '.join(tech_lines)}
MACD: DIF={latest.get('DIF', 0):.3f} DEA={latest.get('DEA', 0):.3f}
RSI(14): {rsi_val:.1f}
关键价位: {level_text}

{risk_text}

{sentiment_text}

{akshare_text}

{news_text}

请按照你的系统指令框架, 基于以上全部数据(特别注意大盘和板块环境)进行分析。
"""


# =====================================================================
# 模块9: Agent 调用
# =====================================================================
_PRIORITY_MAP = {
    "DeepSeek": ("deepseek", "qwen", "gemini"),
    "Qwen":     ("qwen", "deepseek", "gemini"),
    "Gemini":   ("gemini", "deepseek", "qwen"),
}
_DEFAULT_AGENT_PRIORITY = ("deepseek", "qwen", "gemini")
_DEFAULT_JUDGE_PRIORITY = ("qwen", "deepseek", "gemini")

def _get_priority(choice, default):
    """Convert user model choice to priority tuple."""
    if choice and choice in _PRIORITY_MAP:
        return _PRIORITY_MAP[choice]
    return default

def run_single_agent(hub, agent, data_prompt, priority=None):
    if priority is None:
        priority = _DEFAULT_AGENT_PRIORITY
    text, provider = hub.generate(
        sys_prompt=agent["system_prompt"],
        user_prompt=data_prompt,
        temperature=0.5, max_tokens=1500,
        priority=priority,
    )
    if not text:
        return "(该交易员未给出意见: 模型返回空内容)", "None"
    if text.startswith("[ALL FAILED]"):
        return text, "None"
    return text, provider

def run_judge(hub, stock_name, data_prompt, agent_opinions, priority=None):
    if priority is None:
        priority = _DEFAULT_JUDGE_PRIORITY
    opinions_text = ""
    for agent, opinion, prov in agent_opinions:
        opinions_text += chr(10) + "="*40 + chr(10)
        opinions_text += agent["emoji"] + " " + agent["name"] + " (" + agent["desc"] + ") [" + prov + "]" + chr(10)
        opinions_text += opinion + chr(10)
    eq50 = "="*50
    jp = "数据:" + chr(10) + data_prompt + chr(10) + eq50 + chr(10)
    jp += "5位交易员分析:" + chr(10) + opinions_text + chr(10) + eq50
    jp += chr(10) + "请综合裁决。"
    stream, provider = hub.generate_stream(
        sys_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=jp,
        temperature=0.3, max_tokens=3000,
        priority=priority,
    )
    return stream, provider

def extract_stance(text):
    t = text[:300]
    if any(k in t for k in ["看多", "做多", "买入", "强烈买入"]):
        return "看多", "bullish"
    elif any(k in t for k in ["看空", "做空", "卖出", "减仓", "强烈卖出"]):
        return "看空", "bearish"
    return "中性", "neutral"


# =====================================================================
# 模块10: 联网搜索
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
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2, max_output_tokens=2048,
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
# 模块11: 主界面
# =====================================================================
def main():
    # ---- Init clients (cached) ----
    _init_ok = False
    _init_err = ""
    try:
        hub, pro, cfg = init_clients()
        _init_ok = True
    except Exception as e:
        hub, pro, cfg = None, None, None
        _init_err = str(e)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ 分析设置</div>', unsafe_allow_html=True)
        kline_days = st.selectbox("K线回溯天数", [30, 60, 90, 120], index=1)
        enable_search = st.toggle("联网搜索", value=True)
        enable_akshare = st.toggle("增强数据(AKShare)", value=True, help="资金流向/北向/融资融券/涨跌停")
        enable_realtime = st.toggle("实时盘口", value=True, help="当日分时/盘口/涨速/内外盘")
        st.divider()

        # ---- Model status (always visible) ----
        st.markdown("**🤖 模型状态**")
        if _init_ok and hub:
            providers = hub.available_providers()
            init_errors = getattr(hub, "init_errors", {})
            for p in providers:
                st.caption(f"🟢 {p}")
            for name, err in init_errors.items():
                if name not in providers:
                    st.caption(f"🔴 {name}: {str(err)[:60]}")

            # ---- Model selection ----
            st.markdown("**🎯 模型选择**")
            _model_options = ["自动 (推荐)"] + providers
            agent_model_choice = st.selectbox(
                "Agent模型 (5位交易员)",
                _model_options, index=0,
                help="自动=DeepSeek优先, 失败自动切换")
            judge_model_choice = st.selectbox(
                "裁判模型 (首席策略官)",
                _model_options, index=0,
                help="自动=Qwen优先, 失败自动切换")

            # Diagnostic button - always visible
            if hasattr(hub, "diagnose"):
                if st.button("🔧 诊断模型连接"):
                    with st.spinner("测试中..."):
                        diag = hub.diagnose()
                    for name, res in diag.items():
                        if res["status"] == "OK":
                            st.success(f"✅ {name}: {res['response']}")
                        elif res["status"] == "NOT_INIT":
                            st.warning(f"⚪ {name}: {res['error'][:80]}")
                        else:
                            st.error(f"❌ {name}: {res['error'][:120]}")
        else:
            st.error(f"初始化失败: {_init_err if not _init_ok else '未知'}")
            agent_model_choice = "自动 (推荐)"
            judge_model_choice = "自动 (推荐)"

        st.divider()
        st.caption("⚠️ 分析仅供参考，不构成投资建议")
        st.caption(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ---- Main input ----
    st.markdown("## 📈 A股智能分析系统 v4.2")
    st.caption("多模型Fallback · 5Agent辩论 · CoT · 风险量化 · 大盘板块 · 情绪")

    input_col1, input_col2 = st.columns([3, 1])
    with input_col1:
        stock_code = st.text_input("股票代码", placeholder="输入6位代码，如 600519",
                                    max_chars=6, label_visibility="collapsed")
    with input_col2:
        analyze_btn = st.button("🚀 分析", type="primary", use_container_width=True)

    if not analyze_btn or not stock_code:
        # Show model status on main page too
        if _init_ok and hub:
            providers = hub.available_providers()
            init_errors = getattr(hub, "init_errors", {})
            status_parts = []
            for p in providers:
                status_parts.append(f"🟢 {p}")
            for name, err in init_errors.items():
                if name not in providers:
                    status_parts.append(f"🔴 {name}")
            st.caption("模型: " + "  ".join(status_parts))
        elif not _init_ok:
            st.error(f"⚠️ 模型初始化失败: {_init_err}")

        st.markdown("""
        <div class="landing-grid">
            <div class="landing-card"><h4>🔍 技术+风险</h4><p>MA/MACD/RSI/布林带<br>VaR · CVaR · 最大回撤 · Sharpe</p></div>
            <div class="landing-card"><h4>🌍 大盘+板块+情绪</h4><p>三大指数 · 板块排名 · 恐贪指数<br>北向资金 · 融资融券 · 涨跌停</p></div>
            <div class="landing-card"><h4>🎭 5人辩论</h4><p>趋势·价值·魔鬼代言人·短线·情绪<br>大盘暴跌自动降级 · CoT推理</p></div>
        </div>""", unsafe_allow_html=True)
        return

    if len(stock_code) != 6 or not stock_code.isdigit():
        st.error("请输入有效的6位股票代码")
        return

    if not _init_ok or hub is None:
        st.error("模型初始化失败，请检查 API Key 配置")
        return

    ts_code = build_ts_code(stock_code)

    # ==================== 数据获取 ====================
    with st.status("正在获取数据...", expanded=True) as status:
        st.write("📥 K线数据...")
        df = fetch_kline(pro, ts_code, kline_days)
        if df.empty:
            st.error(f"未找到 {ts_code} 的数据")
            return

        st.write("📊 财务数据...")
        fina = fetch_financial(pro, ts_code)

        st.write("🏢 公司信息...")
        stock_name = fetch_stock_name(pro, ts_code)
        business = fetch_company(pro, ts_code)

        st.write("⚙️ 技术指标...")
        df = TechEngine.calc_all(df)
        signals = TechEngine.get_signals(df)
        levels = TechEngine.support_resistance(df)

        st.write("📉 风险指标...")
        risk = RiskEngine.calc_all(df)
        risk_text = RiskEngine.format_text(risk)

        ak_data = {}
        akshare_text = "(增强数据已关闭)"
        if enable_akshare:
            st.write("💰 资金流向 / 北向 / 融资融券...")
            ak_data = fetch_akshare_data(stock_code)
            akshare_text = format_akshare_text(ak_data)

        rt_data = {}
        realtime_text = ""
        if enable_realtime:
            st.write("⚡ 实时盘口数据...")
            rt_data = fetch_realtime(stock_code, cfg.get("realtime_token", ""))
            realtime_text = format_realtime_text(rt_data)

        st.write("🧠 情绪分析...")
        fg = MarketSentiment.calc_fear_greed(df, ak_data)
        regime = MarketSentiment.detect_regime(df)
        sentiment_text = MarketSentiment.format_text(fg, regime)

        st.write("🌍 大盘 & 板块分析...")
        mkt_ctx = MarketContext.analyze_indices(pro)
        sector_ctx = MarketContext.analyze_sector(pro, ts_code)
        market_text = MarketContext.format_text(mkt_ctx, sector_ctx)

        news_text = ""
        if enable_search:
            st.write("🌐 联网搜索...")
            news_text = hub.search_news(stock_name, business)

        status.update(label="数据准备完毕 ✅", state="complete", expanded=False)

    # ==================== 标题 & 指标卡片 ====================
    latest = df.iloc[-1]

    # Use realtime data if available, else fall back to Tushare
    if rt_data and "error" not in rt_data and rt_data.get("price", 0) > 0:
        _price = rt_data["price"]
        _chg_pct = rt_data["change_pct"]
        _vol_ratio = rt_data["vol_ratio"]
        _amount = rt_data["amount"] / 10000  # 元 -> 万
        _time_tag = f"⚡ {rt_data.get('time', '')} 实时"
        _extra_cards = f"""
        <div class="metric-card"><div class="label">振幅</div><div class="value neutral">{rt_data['amplitude']:.2f}%</div></div>
        <div class="metric-card"><div class="label">换手率</div><div class="value neutral">{rt_data['turnover']:.2f}%</div></div>
        <div class="metric-card"><div class="label">委比</div><div class="value {'up' if rt_data['wei_bi'] > 0 else 'down'}">{rt_data['wei_bi']:+.1f}%</div></div>"""
    else:
        _price = latest['close']
        _chg_pct = latest["pct_chg"]
        _vol_ratio = latest.get("VOL_RATIO", 1.0)
        _amount = latest['amount'] / 10000
        _time_tag = ""
        _extra_cards = ""

    change_color = "up" if _chg_pct >= 0 else "down"

    st.markdown(f"## {stock_name}  `{ts_code}`")
    if _time_tag:
        st.caption(_time_tag)

    rsi_val = latest.get("RSI", 50)
    rsi_class = "up" if pd.notna(rsi_val) and rsi_val > 70 else ("down" if pd.notna(rsi_val) and rsi_val < 30 else "neutral")
    vr_class = "up" if _vol_ratio > 1.5 else ("down" if _vol_ratio < 0.5 else "neutral")

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card"><div class="label">最新价</div><div class="value {change_color}">{_price:.2f}</div></div>
        <div class="metric-card"><div class="label">涨跌幅</div><div class="value {change_color}">{_chg_pct:+.2f}%</div></div>
        <div class="metric-card"><div class="label">RSI(14)</div><div class="value {rsi_class}">{rsi_val:.1f}</div></div>
        <div class="metric-card"><div class="label">量比</div><div class="value {vr_class}">{_vol_ratio:.2f}</div></div>
        <div class="metric-card"><div class="label">成交额</div><div class="value neutral">{_amount:.0f}万</div></div>
        {_extra_cards}
    </div>""", unsafe_allow_html=True)

    # Signal tags
    if signals:
        tags_html = '<div class="signal-wrap"><strong style="margin-right:0.4rem">技术信号</strong>'
        for text, kind in signals:
            tags_html += f'<span class="signal-tag signal-{kind}">{text}</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # ==================== 大盘 & 板块概览 ====================
    st.divider()

    # Crash alert banner
    if mkt_ctx.get("crash_alert"):
        st.error(f"🚨 **大盘暴跌警报!** {mkt_ctx['summary']}")
    elif mkt_ctx.get("systemic_risk"):
        st.warning(f"⚠️ **大盘风险提示:** {mkt_ctx['summary']}")

    st.markdown("**🌍 大盘环境**")
    idx_items = [(n, d) for n, d in mkt_ctx.get("indices", {}).items() if isinstance(d, dict)]
    if idx_items:
        idx_cols = st.columns(len(idx_items))
        for i, (name, data) in enumerate(idx_items):
            with idx_cols[i]:
                if "close" in data:
                    chg = data["today_chg"]
                    st.metric(name, f"{data['close']}", delta=f"{chg:+.2f}%",
                              delta_color="inverse" if chg < 0 else "normal")
                    st.caption(f"5日:{data['chg_5d']:+.2f}%  20日:{data['chg_20d']:+.2f}%")
                else:
                    st.metric(name, "N/A")

    # Sector info
    sect_col1, sect_col2 = st.columns([1, 2])
    with sect_col1:
        st.markdown(f"**所属板块:** {sector_ctx['name']}")
        if sector_ctx.get("sector_trend"):
            st.caption(sector_ctx["sector_trend"])
    with sect_col2:
        if sector_ctx.get("rank_info"):
            top3 = sector_ctx["rank_info"]["top5"][:3]
            bot3 = sector_ctx["rank_info"]["bottom5"][-3:]
            top_str = " · ".join(f"{n}({float(c):+.1f}%)" for n, c in top3)
            bot_str = " · ".join(f"{n}({float(c):+.1f}%)" for n, c in bot3)
            st.caption(f"🔥 领涨: {top_str}")
            st.caption(f"💧 领跌: {bot_str}")

    # ==================== 恐贪仪表盘 + 市场状态 + 风险 ====================
    st.divider()

    col_fg, col_regime = st.columns([3, 2])
    with col_fg:
        fg_color = "#ef4444" if fg["score"] >= 60 else ("#22c55e" if fg["score"] <= 40 else "#eab308")
        comp_html = "".join(f'<span class="fg-label">{k}:{v}</span>' for k, v in fg["components"].items())
        st.markdown(f"""
        <div class="fg-gauge">
            <div>
                <div class="fg-score" style="color:{fg_color}">{fg['score']}</div>
                <div class="fg-label">{fg['label']}</div>
            </div>
            <div style="flex:1">
                <div class="fg-label" style="margin-bottom:0.3rem">恐惧 ← → 贪婪</div>
                <div class="fg-bar"><div class="fg-marker" style="left:{fg['score']}%"></div></div>
                <div class="fg-components">{comp_html}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_regime:
        st.markdown(f"**市场状态** {regime['regime']}")
        st.caption(regime["detail"])
        st.caption(f"20日均收益: {regime['mean_ret_20d']}% | 20日波动: {regime['vol_20d']}%")

    # Risk metrics
    if risk:
        st.markdown("**风险仪表盘**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("VaR(95%)", f"{risk['var_95']}%", help="95%概率下单日最大亏损")
        with rc2:
            st.metric("CVaR(95%)", f"{risk['cvar_95']}%", help="极端情况下平均亏损")
        with rc3:
            st.metric("最大回撤", f"{risk['max_drawdown']}%")
        with rc4:
            st.metric("Sharpe", f"{risk['sharpe']}", help="风险调整后收益, >1为佳")

    st.divider()

    # ==================== 图表 ====================
    tab_main, tab_macd, tab_rsi = st.tabs(["📈 K线+均线+布林带", "📊 MACD", "📉 RSI"])
    _cc = {"displayModeBar": False, "scrollZoom": True}
    with tab_main:
        st.plotly_chart(build_main_chart(df), use_container_width=True, config=_cc)
    with tab_macd:
        st.plotly_chart(build_macd_chart(df), use_container_width=True, config=_cc)
    with tab_rsi:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True, config=_cc)

    # Support/Resistance
    if levels:
        items_html = ""
        for label, val in levels.items():
            css = "lv-r" if "压力" in label else ("lv-s" if "支撑" in label else "lv-p")
            items_html += f'<div class="level-item"><div class="level-label">{label}</div><div class="level-val {css}">{val}</div></div>'
        st.markdown(f'<div style="margin-top:0.2rem"><strong>关键价位</strong></div><div class="levels-grid">{items_html}</div>', unsafe_allow_html=True)

    st.divider()

    # ==================== 5-Agent 辩论 ====================
    st.markdown("### 🎭 交易员辩论会")
    st.caption("5 位不同风格交易员 + CoT推理 → 首席策略官裁决")

    data_prompt = build_prompt(stock_name, ts_code, df, fina, business, news_text,
                                risk_text, sentiment_text, akshare_text, market_text,
                                realtime_text)
    agent_opinions = []

    # Convert user choices to priority tuples
    _agent_pri = _get_priority(agent_model_choice, _DEFAULT_AGENT_PRIORITY)
    _judge_pri = _get_priority(judge_model_choice, _DEFAULT_JUDGE_PRIORITY)

    with st.status("交易员辩论中...", expanded=True) as ds:
        for agent in TRADER_AGENTS:
            st.write(f"{agent['emoji']} {agent['name']}正在分析...")
            opinion, prov = run_single_agent(hub, agent, data_prompt, priority=_agent_pri)
            if prov == "None":
                st.write(f"  ⚠️ {agent['name']}失败: {opinion[:120]}")
            else:
                st.write(f"  ✅ {agent['name']} → {prov}")
            agent_opinions.append((agent, opinion, prov))
            time.sleep(0.3)
        st.write("🎯 首席策略官裁决中...")
        judge_stream, judge_prov = run_judge(hub, stock_name, data_prompt, agent_opinions, priority=_judge_pri)
        st.write(f"  裁判模型: {judge_prov}")
        ds.update(label="辩论完毕 ✅", state="complete", expanded=False)

    # Agent tabs
    agent_tabs = st.tabs([a["tab_label"] for a in TRADER_AGENTS])
    for i, (agent, opinion, prov) in enumerate(agent_opinions):
        with agent_tabs[i]:
            stance_text, stance_class = extract_stance(opinion)
            st.markdown(f"""
            <div class="{agent['style']}"><div class="agent-header">
                <span class="agent-emoji">{agent['emoji']}</span>
                <span>{agent['name']}<br><small style="font-weight:400;color:#8892b0">{agent['desc']}</small></span>
                <span class="agent-stance stance-{stance_class}">{stance_text}</span>
                <span style="background:#2a2a4a;padding:2px 8px;border-radius:8px;font-size:0.7rem;color:#8892b0;margin-left:auto">{prov}</span>
            </div></div>""", unsafe_allow_html=True)
            if opinion.startswith("[ALL FAILED]"):
                st.error(f"⚠️ 模型调用失败:\n{opinion}")
            else:
                st.markdown(opinion)

    # Judge verdict
    st.divider()
    st.markdown('<div class="verdict-title">🎯 首席策略官 · 最终裁决 <span style="background:#2a2a4a;padding:2px 8px;border-radius:8px;font-size:0.7rem;color:#8892b0">' + judge_prov + '</span></div>', unsafe_allow_html=True)
    if judge_stream:
        try:
            full_text = st.write_stream(judge_stream)
            if full_text and full_text.startswith("[ALL FAILED]"):
                st.error(f"⚠️ 裁决模型调用失败: {full_text}")
        except Exception as e:
            st.error(f"裁决失败: {type(e).__name__}: {e}")
    else:
        st.error("首席策略官调用失败")

    # ==================== 折叠面板 ====================
    with st.expander("🌍 大盘 & 板块详情"):
        st.text(market_text)

    with st.expander("📋 财务指标"):
        if fina:
            cols = st.columns(2)
            for i, (k, label) in enumerate({"roe": "ROE (%)", "grossprofit_margin": "毛利率 (%)",
                                             "netprofit_yoy": "净利润同比 (%)", "revenue_yoy": "营收同比 (%)",
                                             "debt_to_assets": "资产负债率 (%)", "eps": "每股收益 (元)"}.items()):
                v = fina.get(k)
                with cols[i % 2]:
                    st.metric(label, f"{v:.2f}" if v is not None and str(v) != "nan" else "N/A")
        else:
            st.info("暂无")

    if enable_akshare:
        with st.expander("💰 资金流向详情"):
            st.text(akshare_text)

    if enable_realtime and realtime_text:
        with st.expander("⚡ 实时盘口详情"):
            if "error" in rt_data:
                st.warning(realtime_text)
            else:
                st.text(realtime_text)

    if news_text and enable_search:
        with st.expander("🌐 联网搜索结果"):
            st.markdown(news_text)

    with st.expander("📝 完整辩论记录"):
        for agent, opinion, prov in agent_opinions:
            st.markdown(f"**{agent['emoji']} {agent['name']}** `{prov}`")
            st.markdown(opinion)
            st.divider()


if __name__ == "__main__":
    main()
