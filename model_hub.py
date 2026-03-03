"""model_hub.py - Multi-model router with auto fallback"""
import streamlit as st
from google import genai
from google.genai import types
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_config():
    return {
        "gemini_key": st.secrets.get("GEMINI_API_KEY", ""),
        "tushare_token": st.secrets["TUSHARE_TOKEN"],
        "tushare_proxy": st.secrets.get("TUSHARE_PROXY_URL", ""),
        "gemini_model": st.secrets.get("GEMINI_MODEL",
                        "gemini-3-flash-preview"),
        "deepseek_key": st.secrets.get("DEEPSEEK_API_KEY", ""),
        "deepseek_model": st.secrets.get("DEEPSEEK_MODEL",
                          "deepseek-reasoner"),
        "deepseek_base_url": st.secrets.get("DEEPSEEK_BASE_URL",
                             "https://api.deepseek.com"),
        "qwen_key": st.secrets.get("QWEN_API_KEY", ""),
        "qwen_model": st.secrets.get("QWEN_MODEL", "qwen3-max-2026-01-23"),
        "qwen_base_url": st.secrets.get("QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    }

class ModelHub:
    """DeepSeek > Qwen > Gemini with auto fallback"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gemini_client = None
        self.deepseek_client = None
        self.qwen_client = None
        self._init_providers()
        self.call_log = []

    def _init_providers(self):
        c = self.cfg
        if c["gemini_key"]:
            try:
                self.gemini_client = genai.Client(
                    api_key=c["gemini_key"])
            except Exception:
                pass
        if c["deepseek_key"] and OpenAI:
            try:
                self.deepseek_client = OpenAI(
                    api_key=c["deepseek_key"],
                    base_url=c["deepseek_base_url"])
            except Exception:
                pass
        if c["qwen_key"] and OpenAI:
            try:
                self.qwen_client = OpenAI(
                    api_key=c["qwen_key"],
                    base_url=c["qwen_base_url"])
            except Exception:
                pass

    def available_providers(self):
        p = []
        if self.deepseek_client:
            p.append("DeepSeek")
        if self.qwen_client:
            p.append("Qwen")
        if self.gemini_client:
            p.append("Gemini")
        return p

    def _call_oai(self, cl, mdl, sp, up, t=0.5, mt=1500):
        ms = []
        if sp: ms.append({"role":"system","content":sp})
        ms.append({"role":"user","content":up})
        r = cl.chat.completions.create(model=mdl,messages=ms,temperature=t,max_tokens=mt)
        return r.choices[0].message.content or ""

    def _stream_oai(self, cl, mdl, sp, up, t=0.3, mt=3000):
        ms = []
        if sp: ms.append({"role":"system","content":sp})
        ms.append({"role":"user","content":up})
        s = cl.chat.completions.create(model=mdl,messages=ms,temperature=t,max_tokens=mt,stream=True)
        for c in s:
            d = c.choices[0].delta if c.choices else None
            if d and d.content: yield d.content

    def _call_gemini(self, sp, up, t=0.5, mt=1500):
        r = self.gemini_client.models.generate_content(
            model=self.cfg["gemini_model"],contents=up,
            config=types.GenerateContentConfig(system_instruction=sp,temperature=t,max_output_tokens=mt))
        return r.text if r.text else ""

    def _stream_gemini(self, sp, up, t=0.3, mt=3000):
        s = self.gemini_client.models.generate_content_stream(
            model=self.cfg["gemini_model"],contents=up,
            config=types.GenerateContentConfig(system_instruction=sp,temperature=t,max_output_tokens=mt))
        for c in s:
            if c.text: yield c.text

    def generate(self, sys_prompt, user_prompt, temperature=0.5,
                  max_tokens=1500, priority=("deepseek","qwen","gemini")):
        last_err = None
        for prov in priority:
            try:
                if prov=="deepseek" and self.deepseek_client:
                    txt = self._call_oai(self.deepseek_client,self.cfg["deepseek_model"],sys_prompt,user_prompt,temperature,max_tokens)
                    self.call_log.append(("DeepSeek",True)); return txt,"DeepSeek"
                elif prov=="qwen" and self.qwen_client:
                    txt = self._call_oai(self.qwen_client,self.cfg["qwen_model"],sys_prompt,user_prompt,temperature,max_tokens)
                    self.call_log.append(("Qwen",True)); return txt,"Qwen"
                elif prov=="gemini" and self.gemini_client:
                    txt = self._call_gemini(sys_prompt,user_prompt,temperature,max_tokens)
                    self.call_log.append(("Gemini",True)); return txt,"Gemini"
            except Exception as e:
                last_err = e; self.call_log.append((prov,False)); continue
        return f"(all models failed: {last_err})","None"

    def generate_stream(self, sys_prompt, user_prompt, temperature=0.3,
                         max_tokens=3000, priority=("qwen","deepseek","gemini")):
        for prov in priority:
            try:
                if prov=="deepseek" and self.deepseek_client:
                    return self._stream_oai(self.deepseek_client,self.cfg["deepseek_model"],sys_prompt,user_prompt,temperature,max_tokens),"DeepSeek"
                elif prov=="qwen" and self.qwen_client:
                    return self._stream_oai(self.qwen_client,self.cfg["qwen_model"],sys_prompt,user_prompt,temperature,max_tokens),"Qwen"
                elif prov=="gemini" and self.gemini_client:
                    return self._stream_gemini(sys_prompt,user_prompt,temperature,max_tokens),"Gemini"
            except Exception: continue
        return None,"None"

    def search_news(self, stock_name, business):
        prompt = ('A股 "' + stock_name + '" (主营:' + business
                  + ') 最新:1.行业政策 2.公司公告 3.行业趋势 4.机构评级 5.风险事件'
                  + ' 每条2-3句,标注来源和时间')
        if not self.gemini_client:
            return '(Gemini未配置,联网搜索不可用)'
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
            src = chr(10).join(sources) if sources else '(无明确来源)'
            return '【联网搜索】' + chr(10) + text + chr(10)*2 + '来源:' + chr(10) + src
        except Exception as e:
            return '(联网搜索异常: ' + str(e) + ')'


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
