"""model_hub.py v2 - Multi-model router with error tracking & diagnostics"""
import streamlit as st
import traceback
from google import genai
from google.genai import types

# --- OpenAI SDK (DeepSeek / Qwen 共用) ---
_openai_import_err = None
try:
    from openai import OpenAI
except ImportError as _e:
    OpenAI = None
    _openai_import_err = str(_e)


def get_config():
    return {
        "gemini_key": st.secrets.get("GEMINI_API_KEY", ""),
        "tushare_token": st.secrets["TUSHARE_TOKEN"],
        "tushare_proxy": st.secrets.get("TUSHARE_PROXY_URL", ""),
        "gemini_model": st.secrets.get("GEMINI_MODEL",
                        "gemini-3.1-pro-preview"),
        "deepseek_key": st.secrets.get("DEEPSEEK_API_KEY", ""),
        "deepseek_model": st.secrets.get("DEEPSEEK_MODEL",
                          "deepseek-chat"),
        "deepseek_base_url": st.secrets.get("DEEPSEEK_BASE_URL",
                             "https://api.deepseek.com"),
        "qwen_key": st.secrets.get("QWEN_API_KEY", ""),
        "qwen_model": st.secrets.get("QWEN_MODEL", "qwen-plus"),
        "qwen_base_url": st.secrets.get("QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "realtime_token": st.secrets.get("REALTIME_TOKEN", ""),
    }


class ModelHub:
    """DeepSeek / Qwen / Gemini with auto fallback + full error tracking"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gemini_client = None
        self.deepseek_client = None
        self.qwen_client = None
        self.init_errors = {}       # provider -> error string
        self.call_log = []          # [(provider, success, error_str)]
        self._init_providers()

    def _init_providers(self):
        c = self.cfg

        # --- Gemini ---
        if c["gemini_key"]:
            try:
                self.gemini_client = genai.Client(api_key=c["gemini_key"])
            except Exception as e:
                self.init_errors["Gemini"] = f"Client init failed: {e}"
        else:
            self.init_errors["Gemini"] = "GEMINI_API_KEY not set"

        # --- DeepSeek ---
        if not c["deepseek_key"]:
            self.init_errors["DeepSeek"] = "DEEPSEEK_API_KEY not set"
        elif OpenAI is None:
            self.init_errors["DeepSeek"] = (
                f"openai package not installed! pip install openai. "
                f"Import error: {_openai_import_err}")
        else:
            try:
                self.deepseek_client = OpenAI(
                    api_key=c["deepseek_key"],
                    base_url=c["deepseek_base_url"],
                    timeout=60.0)
            except Exception as e:
                self.init_errors["DeepSeek"] = f"Client init failed: {e}"

        # --- Qwen ---
        if not c["qwen_key"]:
            self.init_errors["Qwen"] = "QWEN_API_KEY not set"
        elif OpenAI is None:
            self.init_errors["Qwen"] = (
                f"openai package not installed! pip install openai. "
                f"Import error: {_openai_import_err}")
        else:
            try:
                self.qwen_client = OpenAI(
                    api_key=c["qwen_key"],
                    base_url=c["qwen_base_url"],
                    timeout=60.0)
            except Exception as e:
                self.init_errors["Qwen"] = f"Client init failed: {e}"

    def available_providers(self):
        p = []
        if self.deepseek_client:
            p.append("DeepSeek")
        if self.qwen_client:
            p.append("Qwen")
        if self.gemini_client:
            p.append("Gemini")
        return p

    def _get_client(self, prov):
        """Return (client_or_None, model_name, is_oai_style)"""
        if prov == "deepseek":
            return self.deepseek_client, self.cfg["deepseek_model"], True
        elif prov == "qwen":
            return self.qwen_client, self.cfg["qwen_model"], True
        elif prov == "gemini":
            return self.gemini_client, self.cfg["gemini_model"], False
        return None, None, False

    def _call_oai(self, cl, mdl, sp, up, t=0.5, mt=1500):
        ms = []
        if sp:
            ms.append({"role": "system", "content": sp})
        ms.append({"role": "user", "content": up})
        r = cl.chat.completions.create(
            model=mdl, messages=ms,
            temperature=t, max_tokens=mt)
        return r.choices[0].message.content or ""

    def _stream_oai(self, cl, mdl, sp, up, t=0.3, mt=3000):
        ms = []
        if sp:
            ms.append({"role": "system", "content": sp})
        ms.append({"role": "user", "content": up})
        s = cl.chat.completions.create(
            model=mdl, messages=ms,
            temperature=t, max_tokens=mt, stream=True)
        for chunk in s:
            d = chunk.choices[0].delta if chunk.choices else None
            if d and d.content:
                yield d.content

    def _call_gemini(self, sp, up, t=0.5, mt=1500):
        r = self.gemini_client.models.generate_content(
            model=self.cfg["gemini_model"], contents=up,
            config=types.GenerateContentConfig(
                system_instruction=sp,
                temperature=t, max_output_tokens=mt))
        return r.text if r.text else ""

    def _stream_gemini(self, sp, up, t=0.3, mt=3000):
        s = self.gemini_client.models.generate_content_stream(
            model=self.cfg["gemini_model"], contents=up,
            config=types.GenerateContentConfig(
                system_instruction=sp,
                temperature=t, max_output_tokens=mt))
        for c in s:
            if c.text:
                yield c.text

    # ----- Main generate (non-stream) -----
    def generate(self, sys_prompt, user_prompt, temperature=0.5,
                 max_tokens=1500, priority=("deepseek", "qwen", "gemini")):
        errors = []
        _NAMES = {"deepseek": "DeepSeek", "qwen": "Qwen", "gemini": "Gemini"}
        for prov in priority:
            label = _NAMES.get(prov, prov)
            client, model, is_oai = self._get_client(prov)
            if client is None:
                reason = self.init_errors.get(label, "client not initialized")
                errors.append(f"{label}: {reason}")
                continue
            try:
                if is_oai:
                    txt = self._call_oai(client, model, sys_prompt,
                                         user_prompt, temperature, max_tokens)
                else:
                    txt = self._call_gemini(sys_prompt, user_prompt,
                                            temperature, max_tokens)
                self.call_log.append((label, True, ""))
                return txt, label
            except Exception as e:
                err_short = f"{type(e).__name__}: {str(e)[:200]}"
                errors.append(f"{label}: {err_short}")
                self.call_log.append((label, False, err_short))
                continue

        # All failed - return detailed error
        err_detail = " | ".join(errors)
        return f"[ALL FAILED] {err_detail}", "None"

    # ----- Stream generate (for judge) -----
    def generate_stream(self, sys_prompt, user_prompt, temperature=0.3,
                        max_tokens=3000, priority=("qwen", "deepseek", "gemini")):
        _NAMES = {"deepseek": "DeepSeek", "qwen": "Qwen", "gemini": "Gemini"}
        errors = []
        for prov in priority:
            label = _NAMES.get(prov, prov)
            client, model, is_oai = self._get_client(prov)
            if client is None:
                continue
            try:
                if is_oai:
                    gen = self._stream_oai(client, model, sys_prompt,
                                           user_prompt, temperature, max_tokens)
                else:
                    gen = self._stream_gemini(sys_prompt, user_prompt,
                                              temperature, max_tokens)
                return gen, label
            except Exception as e:
                errors.append(f"{prov}: {e}")
                continue

        # Return an error generator
        def _err_gen():
            yield f"[ALL FAILED] {' | '.join(errors)}"
        return _err_gen(), "None"

    # ----- News search (Gemini only) -----
    def search_news(self, stock_name, business):
        prompt = (f'A股 "{stock_name}" (主营:{business}) '
                  '最新:1.行业政策 2.公司公告 3.行业趋势 4.机构评级 5.风险事件'
                  ' 每条2-3句,标注来源和时间')
        if not self.gemini_client:
            return ('(联网搜索不可用: ' +
                    self.init_errors.get("Gemini", "Gemini未配置") + ')')
        try:
            r = self.gemini_client.models.generate_content(
                model=self.cfg['gemini_model'], contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2, max_output_tokens=2048))
            text = r.text if r.text else ''
            sources = []
            try:
                meta = r.candidates[0].grounding_metadata
                if meta and meta.grounding_chunks:
                    for ch in meta.grounding_chunks[:5]:
                        if hasattr(ch, 'web') and ch.web:
                            sources.append('- ' + ch.web.title)
            except Exception:
                pass
            src = '\n'.join(sources) if sources else '(无明确来源)'
            return f'【联网搜索】\n{text}\n\n来源:\n{src}'
        except Exception as e:
            return f'(联网搜索异常: {type(e).__name__}: {e})'

    def search_macro(self):
        """Search for international macro environment affecting A-shares."""
        prompt = """搜索今日影响A股的国际宏观环境, 请覆盖:
1. 隔夜美股三大指数(道琼斯/标普500/纳斯达克)涨跌及原因
2. 美联储最新政策信号/美债收益率/美元指数变化
3. 中美关系/关税政策/贸易摩擦最新动态
4. 地缘政治风险(中东/俄乌/台海等)
5. 全球重大经济数据(非农/CPI/PMI等近期发布的)
6. 亚太市场(港股/日经/韩国)今日表现
每条1-2句, 标注时间, 没有的标"暂无重大变化"。重点关注对A股的传导影响。"""
        if not self.gemini_client:
            return ('(宏观搜索不可用: ' +
                    self.init_errors.get("Gemini", "Gemini未配置") + ')')
        try:
            r = self.gemini_client.models.generate_content(
                model=self.cfg['gemini_model'], contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2, max_output_tokens=2048))
            text = r.text if r.text else ''
            sources = []
            try:
                meta = r.candidates[0].grounding_metadata
                if meta and meta.grounding_chunks:
                    for ch in meta.grounding_chunks[:5]:
                        if hasattr(ch, 'web') and ch.web:
                            sources.append('- ' + ch.web.title)
            except Exception:
                pass
            src = '\n'.join(sources) if sources else '(无明确来源)'
            return f'【国际宏观环境】\n{text}\n\n来源:\n{src}'
        except Exception as e:
            return f'(宏观搜索异常: {type(e).__name__}: {e})'

    # ----- Diagnostic test -----
    def diagnose(self):
        """Test each provider with a minimal prompt. Returns dict of results."""
        results = {}
        test_sp = "回答用中文,10字以内"
        test_up = "说一句话证明你在工作"

        for name, client, model, is_oai in [
            ("DeepSeek", self.deepseek_client,
             self.cfg["deepseek_model"], True),
            ("Qwen", self.qwen_client,
             self.cfg["qwen_model"], True),
            ("Gemini", self.gemini_client,
             self.cfg["gemini_model"], False),
        ]:
            if client is None:
                results[name] = {
                    "status": "NOT_INIT",
                    "error": self.init_errors.get(name, "client is None")}
                continue
            try:
                if is_oai:
                    txt = self._call_oai(client, model, test_sp, test_up,
                                         0.5, 50)
                else:
                    txt = self._call_gemini(test_sp, test_up, 0.5, 50)
                results[name] = {"status": "OK", "response": txt[:100]}
            except Exception as e:
                results[name] = {
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {str(e)[:300]}"}
        return results


# ----- Tushare init -----
import tushare as ts

@st.cache_resource
def init_clients():
    cfg = get_config()
    hub = ModelHub(cfg)
    ts.set_token(cfg['tushare_token'])
    pro = ts.pro_api()
    if cfg['tushare_proxy']:
        pro._DataApi__http_url = cfg['tushare_proxy']
    return hub, pro, cfg
