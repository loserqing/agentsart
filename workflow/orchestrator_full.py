#!/usr/bin/env python3
"""
多智能体协作编排器：生成可嵌入 evolution.html 的 Three.js r160 / npm 0.160.x（WebGL）生成艺术。
技术栈：LangGraph (StateGraph) + langchain-google-genai
"""

import os
import json
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import subprocess
import threading

load_dotenv(Path(__file__).parent.parent / ".env")


# --------------- helpers ---------------

def _evolution_chain_from_env() -> bool:
    return os.getenv("AGENTSART_EVOLUTION_CHAIN", "").strip().lower() in ("1", "true", "yes")


def _compact_prompts_enabled() -> bool:
    """缩短 Builder/Judge/Fixer 等 prompt，降低输入 token（质量请自行 A/B；默认关保留长版）。"""
    return os.getenv("AGENTSART_COMPACT_PROMPTS", "0").strip().lower() in ("1", "true", "yes", "on")


# iteration_log.json 最多保留条数（0 表示不截断）
_ITERATION_LOG_MAX_ENTRIES = int(os.getenv("AGENTSART_ITERATION_LOG_MAX", "30"))


def _clamp_animation_speeds_in_js(code: str) -> str:
    """
    程序化抬升过小的 *speed* 变量初值（不依赖 LLM），避免画面「看似静止」。
    仅匹配标识符中含 speed/Speed 的赋值，如 cameraOrbitSpeed = 0.00005
    """
    if not code or len(code) < 40:
        return code

    assign_re = re.compile(
        r"(?<![\w.])(?P<name>[A-Za-z_][\w]*[Ss]peed[\w]*)\s*=\s*"
        r"(?P<num>\d+(?:\.\d*)?|\.\d+)(?P<exp>[eE][+-]?\d+)?"
    )

    def pick_replacement(name: str) -> float:
        low = name.lower()
        if "camera" in low or "orbit" in low:
            return 0.07
        if any(k in low for k in ("emissive", "fog", "pulse", "density", "global", "ambient")):
            return 1.2
        return 1.0

    def repl(m: re.Match) -> str:
        raw = m.group("num") + (m.group("exp") or "")
        try:
            v = float(raw)
        except ValueError:
            return m.group(0)
        if v <= 0 or v >= 0.01:
            return m.group(0)
        new_v = pick_replacement(m.group("name"))
        return f"{m.group('name')} = {new_v}"

    return assign_re.sub(repl, code)


def _atomic_write(path: Path, content: str):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, data, **kwargs):
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, **kwargs))


def _build_code_inherit_block(use_chain: bool, prev_code: str) -> str:
    if use_chain and prev_code and str(prev_code).strip() not in ("", "无"):
        snippet = str(prev_code)[:800]
        return f"""【代码进化要求（链式模式）】
如果以下有上一代的代码片段，请基于它进行变异（比如调整几何形状、材质、光照或运动曲线）：
{snippet}..."""
    return """【独立实现（默认）】禁止参考、拼接或改写任何旧代码。从零编写完整可运行脚本；坚持一条主渲染路径，严格遵守粒子/InstancedMesh 上限。"""


_TECHNIQUE_SEEDS = {
    "曲线": "const pts=[]; for(let t=0;t<6.28;t+=0.02) pts.push(sin(3*t)*R, cos(5*t)*R, sin(7*t)*R2); scene.add(new THREE.Line(geom, lineMat));",
    "lissajous": "pts.push(sin(a*t+phi)*R, cos(b*t)*R, sin(c*t+psi)*R); // a,b,c 互质",
    "instanced": "im=new THREE.InstancedMesh(baseGeo,mat,COUNT); d=new THREE.Object3D(); for(i){d.position.set(螺旋坐标); d.updateMatrix(); im.setMatrixAt(i,d.matrix);}",
    "线": "segs=new THREE.LineSegments(geom, new THREE.LineBasicMaterial({transparent:true})); // 节点对<阈值距离连线",
    "点云": "cloud=new THREE.Points(geom, new THREE.PointsMaterial({size:0.08,sizeAttenuation:true})); // 噪声采样",
    "表皮": "geo=new THREE.IcosahedronGeometry(8,24); // animate: posArr[i*3+j]=orig+sin(orig*freq+time)*amp",
    "极简": "// 2-4 大尺度 Mesh + 3 不同色温 PointLight + camera 慢轨道",
    "分形": "// 递归/迭代细分: 每步 scale*=0.6+rotate+offset, InstancedMesh 显示叶节点",
}


def _technique_seed(technique: str) -> str:
    if not technique or not isinstance(technique, str):
        return ""
    t = technique.lower()
    for key, seed in _TECHNIQUE_SEEDS.items():
        if key in t:
            return f"\n【技术骨架提示（仅供参考，不可直接复制）】\n{seed}"
    return ""


def _compose_builder_prompt_compact(
    *,
    technique: str,
    style: str,
    concept: str,
    critic_block: str,
    use_chain: bool,
    prev_code: str,
) -> str:
    crit = critic_block.strip()
    seed = _technique_seed(technique)
    inherit = _build_code_inherit_block(use_chain, prev_code)
    inst = (
        "InstancedMesh 着色：mesh.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(n*3),3);"
        "mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);setColorAt/getColorAt；勿 geometry.setAttribute('instanceColor'…)。"
        "每帧矩阵：独立 Matrix4 接 getMatrixAt→decompose→Object3D→updateMatrix→setMatrixAt。"
    )
    return f"""编写 Three.js **r160** 展陈脚本（仅 JS；canvas 挂 document.getElementById('canvas-container')）。

风格：{style}
概念：{concept}
主技术：{technique}
{crit if crit else ''}

【形式】落实主技术；≥2 类几何（InstancedMesh/Points/LineSegments/Mesh 至少两类）；禁「单球点云慢转」惰性套路。
【动画】const t=performance.now()*0.001；可见运动项等效系数 ≥0.01（相机缓移约 0.04–0.12，体/粒子 0.5–2.0）。
【观感】ACESFilmicToneMapping、toneMappingExposure≥1.2；Ambient≥0.3+另有点/方向光≥0.8；背景勿纯黑。雾优先 null；FogExp2 density≤0.018、禁≥0.03；动雾振幅≤0.003。
【色彩】animate 至少一处 setHSL（color 或 emissive）。
【工程】无 import/export/HTML；禁 OrbitControls/Loader/EffectComposer；禁指针/滚轮/键盘监听。若拆成多个 for 更新几何，每个 for 内须声明本循环用到的参数（如沿闭合曲线的角 u），禁止在仅另一 for 内声明过的块外引用。
【r160】renderer.outputColorSpace=THREE.SRGBColorSpace；禁 outputEncoding/旧 Encoding；禁 Geometry/Face3；禁止 THREE.TorusKnot 类（用 TorusKnotGeometry+曲线参数）。
【实例】{inst}
【材质】主 MeshStandard；Physical+实例色则 sheen=0；Color 用 new THREE.Color / setHex。
【限额】Points≤4000；单 InstancedMesh≤1200、合计≤2500；勿多套海量系统叠罗汉。
【注释】禁 /* */ 与中文长说明；// 全程≤3 条。
{seed}
{inherit}
仅输出一个 ```javascript … ``` 代码块。"""


def _compose_judge_prompt_compact(
    *,
    concept80: str,
    visual80: str,
    code_summary: str,
) -> str:
    return f"""你是 Judge：合并代码扫视（文本）+ 艺术打分。勿回写代码。

【代码】1–2 句：OrbitControls/Loader/ShaderMaterial/Geometry/outputEncoding/交互监听/实例色+Physical+sheen 风险；无则写「代码结构健康」。
【艺术】宽容看展陈潜力；有动效/光影/构成/氛围之一即可。
概念：{concept80}
视觉：{visual80}
{code_summary}

评分：8.5+ 展陈级；7–8 勿硬压到 6；5–6 仅严重空壳/脱节。
publish_ready：≥7 倾向 true（非灰盒）；≥7.5 默认 true；≥8 应当 true；<6.5 或 6.5–6.9 且确认空壳则 false。勿因少粒子或未写 setHSL 判 false。

JSON：review_comments、total_score、conclusion、strengths、suggestions、publish_ready、next_iteration_focus。
```json
{{
    "review_comments": "…",
    "total_score": 7.5,
    "conclusion": "…",
    "strengths": ["…"],
    "suggestions": ["…"],
    "publish_ready": false,
    "next_iteration_focus": "…"
}}
```"""


def _compose_fixer_prompt_compact(*, error: str, code: str) -> str:
    return f"""JS 仅语法修复（等价 new Function(code)），勿为 WebGL 运行时大改场景。

错误：
{error}

代码：
```javascript
{code}
```

补全括号/引号/模板字符串/注释；删 import/export；保留 animate+resize；THREE 全局、domElement→canvas-container。可补 instanceColor+DynamicDrawUsage。输出完整 ```javascript … ```。"""


def _normalize_model_name(raw: str) -> str:
    """Strip litellm-style 'gemini/' prefix so langchain-google-genai gets a clean model id."""
    name = raw.strip()
    if name.startswith("gemini/"):
        name = name[len("gemini/"):]
    return name


# --------------- JSON extraction / repair ---------------
# LLM 返回的 JSON 常见毛病：尾逗号、// 与 /* */ 注释、解释文本包裹、嵌套大括号。
# 这组工具做「多路抽取 → 轻量修复 → json.loads」的三段式解析，避免原先 find/rfind
# + 单正则的方式在嵌套/解释文本场景下整段失败。


def _strip_json_comments(s: str) -> str:
    """删掉 // 与 /* */ 注释，字符串内的 // 保留。用字符级扫描，同时尊重转义。"""
    out = []
    i, n = 0, len(s)
    in_str = False
    str_ch = ""
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(s[i + 1])
                i += 2
                continue
            if ch == str_ch:
                in_str = False
            out.append(ch)
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_ch = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "/":
                while i < n and s[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                    i += 1
                i = min(i + 2, n)
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _scan_brace_block(text: str, start: int) -> str | None:
    """从 text[start] == '{' 开始扫描一个顶层平衡大括号块，字符串内的 { } 被忽略。"""
    n = len(text)
    depth = 0
    in_str = False
    str_ch = ""
    j = start
    while j < n:
        ch = text[j]
        if in_str:
            if ch == "\\" and j + 1 < n:
                j += 2
                continue
            if ch == str_ch:
                in_str = False
            j += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_ch = ch
            j += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
        j += 1
    return None


def _iter_json_candidates(text: str):
    """按优先级产出候选 JSON 文本片段。"""
    if not text:
        return

    # 1) ```json ... ``` 代码块（优先）
    for m in re.finditer(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE):
        yield m.group(1).strip()

    # 2) 任意 ``` ... ``` 代码块里形如 { ... } 的内容
    for m in re.finditer(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```", text, re.DOTALL):
        content = m.group(1).strip()
        if content.startswith("{") and content.endswith("}"):
            yield content

    # 3) 顶层平衡大括号扫描（最长优先，兼容嵌套 + 字符串含大括号）
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            block = _scan_brace_block(text, i)
            if block:
                yield block
                i += len(block)
                continue
        i += 1

    # 4) 整段 strip 后是 { ... } 的朴素情况
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        yield stripped


def _repair_json_candidate(s: str) -> str:
    """轻量修复：BOM、注释、尾逗号。保持字符串内容不变。"""
    if not s:
        return s
    s = s.lstrip("\ufeff").strip()
    s = _strip_json_comments(s)
    # 尾逗号：},] 前的逗号
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _try_json_loads(candidate: str):
    """尝试 raw + 修复两遍 json.loads。成功返回 dict/list；失败返回 None。"""
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    repaired = _repair_json_candidate(candidate)
    if repaired and repaired != candidate:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


_log_lock = threading.Lock()

# --------------- LangGraph state schema ---------------

class PipelineState(TypedDict, total=False):
    iteration: int
    face_influence: str
    creative_keywords: list
    use_chain: bool
    prev_concept: str
    prev_visual: str
    prev_code: str
    director: dict
    narrative: dict
    visual: dict
    builder: dict
    reviewer: dict
    critic: dict
    syntax_valid: bool
    syntax_error: str
    fix_attempts: int
    regen_attempts: int


# --------------- orchestrator class ---------------

class FullOrchestrator:
    def __init__(self, project_dir: str = None, model=None, on_event=None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).parent.parent
        raw_model = model or os.getenv("DEFAULT_MODEL", "gemini/gemini-2.0-flash")
        self.model_name = _normalize_model_name(raw_model)

        raw_code_model = os.getenv("CODE_MODEL", "")
        code_model_name = _normalize_model_name(raw_code_model) if raw_code_model.strip() else self.model_name

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.85,
        )
        self.llm_code = ChatGoogleGenerativeAI(
            model=code_model_name,
            temperature=0.72,
        )
        if code_model_name != self.model_name:
            print(f"   📐 文本模型: {self.model_name} | 代码模型: {code_model_name}")

        _tpl_path = Path(__file__).parent / "evolution_template.html"
        self.html_template = _tpl_path.read_text(encoding="utf-8")

        self._on_event = on_event
        self._graph = self._build_graph()

    def _emit(self, agent: str, phase: str, iteration: int, summary: str = "", **kwargs):
        if self._on_event:
            self._on_event(agent, phase, iteration, summary, **kwargs)

    # ---- graph construction ----

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("director", self._node_director)
        g.add_node("narrative", self._node_narrative)
        g.add_node("visual", self._node_visual)
        g.add_node("builder", self._node_builder)
        g.add_node("reviewer", self._node_reviewer)
        g.add_node("syntax_guard", self._node_syntax_guard)
        g.add_node("code_fixer", self._node_code_fixer)
        g.add_node("critic", self._node_critic)
        g.add_edge(START, "director")
        g.add_edge("director", "narrative")
        g.add_edge("director", "visual")
        g.add_edge("narrative", "builder")
        g.add_edge("visual", "builder")
        g.add_edge("builder", "reviewer")
        g.add_edge("reviewer", "syntax_guard")
        g.add_conditional_edges("syntax_guard", self._route_after_syntax)
        g.add_edge("code_fixer", "syntax_guard")
        g.add_conditional_edges("critic", self._route_after_critic)
        return g.compile()

    # ---- LLM helper ----

    @staticmethod
    def _sanitize_code_preview(raw: str, max_len: int) -> str:
        """流式 SSE 用尾部片段：脱敏 + 控长（不保证仍是合法 JS）。"""
        if not raw or max_len < 8:
            return ""
        tail = raw[-max_len:]
        # API / 密钥形态
        tail = re.sub(r"AIza[0-9A-Za-z_-]{24,}", "AIza[REDACTED]", tail)
        tail = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", tail)
        tail = re.sub(
            r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*[\"']?[^\s\"']{6,}",
            r"\1=[REDACTED]",
            tail,
        )
        tail = re.sub(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END", "[PEM REDACTED]", tail)
        # 控制字符（保留换行制表）
        tail = "".join(ch if ch in "\n\t\r" or ord(ch) >= 32 else " " for ch in tail)
        return tail.rstrip()

    @staticmethod
    def _stream_chunk_text(chunk) -> str:
        """将 LangChain 流式 chunk 的 content 规范为字符串（兼容 str / list[dict]）。"""
        c = getattr(chunk, "content", None)
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            out: list[str] = []
            for p in c:
                if isinstance(p, str):
                    out.append(p)
                elif isinstance(p, dict):
                    t = p.get("text")
                    if t:
                        out.append(str(t))
                elif hasattr(p, "text"):
                    out.append(str(p.text))
            return "".join(out)
        return str(c)

    def _llm_call(
        self,
        prompt: str,
        *,
        code_mode: bool = False,
        stream_iteration: int | None = None,
        stream_agent: str | None = None,
    ) -> str:
        llm = self.llm_code if code_mode else self.llm

        want_stream = (
            code_mode
            and stream_iteration is not None
            and stream_agent
            and os.getenv("AGENTSART_CODE_STREAM", "0").strip().lower() in ("1", "true", "yes", "on")
        )
        if want_stream:
            min_chars = max(200, int(os.getenv("AGENTSART_CODE_STREAM_CHARS", "900")))
            min_interval = max(0.12, float(os.getenv("AGENTSART_CODE_STREAM_INTERVAL_SEC", "0.35")))
            prev_max = max(60, int(os.getenv("AGENTSART_CODE_STREAM_PREVIEW_CHARS", "220")))
            parts: list[str] = []
            last_emit_n = 0
            last_emit_t = 0.0
            try:
                for chunk in llm.stream([HumanMessage(content=prompt)]):
                    t = self._stream_chunk_text(chunk)
                    if t:
                        parts.append(t)
                    n = sum(len(x) for x in parts)
                    now = time.monotonic()
                    if (
                        self._on_event
                        and n - last_emit_n >= min_chars
                        and now - last_emit_t >= min_interval
                    ):
                        acc = "".join(parts)
                        pv = self._sanitize_code_preview(acc, prev_max)
                        self._on_event(
                            stream_agent,
                            "stream",
                            stream_iteration,
                            f"{n} chars",
                            preview=pv or None,
                        )
                        last_emit_n = n
                        last_emit_t = now
                text = "".join(parts)
                if self._on_event and text and len(text) != last_emit_n:
                    pv = self._sanitize_code_preview(text, prev_max)
                    self._on_event(
                        stream_agent,
                        "stream",
                        stream_iteration,
                        f"{len(text)} chars",
                        preview=pv or None,
                    )
                if text.strip():
                    return text
                print("   ⚠️ 流式返回为空，回退同步 invoke")
            except Exception as ex:
                print(f"   ⚠️ 代码流式输出异常，回退同步 invoke：{ex}")

        resp = llm.invoke([HumanMessage(content=prompt)])
        c = getattr(resp, "content", None)
        if isinstance(c, str):
            return c or ""
        if isinstance(c, list):
            return self._stream_chunk_text(resp) or ""
        return str(c) if c is not None else ""

    # ---- JSON / code parsers (unchanged) ----

    def _parse_json(self, text: str) -> dict:
        """多路抽取 + 轻量修复的 JSON 解析：
        1) ```json 代码块 → 2) 任意代码块 → 3) 顶层平衡大括号 → 4) 整段。
        每个候选都尝试 raw + 去注释/去尾逗号两遍 json.loads。
        最终失败时保留 JS 代码块回退（Director/Visual 误把 JS 贴到 JSON 槽的情况）。
        """
        if not text:
            print("\n   [警告] JSON 解析输入为空")
            return {}

        last_err_sample = ""
        for candidate in _iter_json_candidates(text):
            parsed = _try_json_loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                # 极少见：LLM 输出了数组。包一层 dict 避免下游 .get 崩。
                return {"items": parsed}
            if not last_err_sample:
                last_err_sample = candidate[:120]

        js_match = re.search(r'```(?:javascript|js)\s*(.*?)\s*```', text, re.DOTALL)
        if js_match:
            print("\n   [恢复] JSON 解析失败，但成功从 Markdown 中提取到了代码块")
            return {"threejs_code": js_match.group(1).strip()}

        print(
            f"\n   [警告] AI 没有返回合法的 JSON！已尝试多路抽取均失败。"
            f"\n     最后候选片段: {last_err_sample[:120]}"
            f"\n     原始前 120 字符: {text[:120]}"
        )
        return {}

    def _extract_js_code(self, text: str) -> str:
        match = re.search(r'```(?:javascript|js)\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        matches = re.finditer(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        for m in matches:
            content = m.group(1).strip()
            if "THREE." in content or "requestAnimationFrame" in content:
                return content
        match = re.search(r'```(?:javascript|js)\s*\n(.*)', text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if "THREE." in content or "requestAnimationFrame" in content:
                return content
        return ""

    # ---- JS syntax validation + conditional fix loop ----

    def _validate_js(self, code: str) -> tuple:
        """Return (is_valid, error_message). Uses Node.js when available, falls back to bracket matching."""
        if not code:
            return False, "Code is too short or empty"
        code = code.lstrip("\ufeff").strip()
        if len(code) < 80:
            return False, "Code is too short or empty"

        try:
            wrapper = f"try{{new Function({json.dumps(code)})}}catch(e){{process.stderr.write(e.message);process.exit(1)}}"
            r = subprocess.run(
                ["node", "-e", wrapper],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0:
                return True, ""
            return False, (r.stderr.strip()[:200] or "Unknown syntax error")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        stack: list[str] = []
        pairs = {")": "(", "}": "{", "]": "["}
        i, n = 0, len(code)
        while i < n:
            ch = code[i]
            if ch == "/" and i + 1 < n and code[i + 1] == "/":
                while i < n and code[i] != "\n":
                    i += 1
                continue
            if ch == "/" and i + 1 < n and code[i + 1] == "*":
                i += 2
                while i < n - 1 and not (code[i] == "*" and code[i + 1] == "/"):
                    i += 1
                i += 2
                continue
            if ch in ('"', "'", "`"):
                q = ch
                i += 1
                while i < n:
                    if code[i] == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if code[i] == q:
                        break
                    i += 1
                i += 1
                continue
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                if not stack:
                    return False, f"Unexpected '{ch}' near char {i}"
                if stack[-1] != pairs[ch]:
                    return False, f"Mismatched '{ch}' near char {i}"
                stack.pop()
            i += 1

        if stack:
            return False, f"Unclosed brackets: {len(stack)} remaining ({''.join(stack[-8:])})"
        return True, ""

    def _coerce_llm_js_response(self, raw: str, fallback: str) -> str:
        """从修复器 LLM 输出中提取 JS；优先 fenced 代码块，其次整段去围栏（避免空提取导致白跑一轮）。"""
        extracted = self._extract_js_code(raw)
        if extracted:
            return extracted.lstrip("\ufeff").strip()
        t = (raw or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:javascript|js)?\s*", "", t, count=1, flags=re.IGNORECASE)
            t = re.sub(r"\s*```\s*$", "", t).strip()
            if "THREE." in t and len(t) >= 80:
                return t.lstrip("\ufeff").strip()
        return fallback.lstrip("\ufeff").strip() if fallback else ""

    def _node_syntax_guard(self, state: PipelineState) -> dict:
        code = (state.get("reviewer") or state.get("builder") or {}).get("threejs_code", "")
        valid, err = self._validate_js(code)
        if valid:
            print("   ✅ 语法校验通过")
        else:
            print(f"   ⚠️ 语法校验失败: {err[:100]}")
        return {"syntax_valid": valid, "syntax_error": err if not valid else ""}

    def _route_after_syntax(self, state: PipelineState) -> str:
        if state.get("syntax_valid", False):
            return "critic"
        if state.get("fix_attempts", 0) >= 2:
            print("   ⚠️ 修复次数已达上限，使用当前代码继续")
            return "critic"
        return "code_fixer"

    def _route_after_critic(self, state: PipelineState) -> str:
        critic_data = state.get("critic", {})
        score = critic_data.get("total_score", 10)
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 10
        publish_ready = critic_data.get("publish_ready", False)
        iteration = state.get("iteration", 0)
        if state.get("regen_attempts", 0) < 1:
            reason = ""
            # 与 Critic 宽松口径对齐：仅极低分或「低分且未达展陈」才重生，避免略严即打回
            if score < 6.0:
                reason = f"评分 {score}/10 < 6.0"
            elif not publish_ready and score < 6.5:
                reason = f"评分 {score}/10 · publish_ready=false"
            if reason:
                print(f"   🔄 {reason}，触发 Builder 重新生成")
                self._emit("critic", "regen", iteration, reason)
                return "builder"
        return END

    def _node_code_fixer(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        attempt = state.get("fix_attempts", 0) + 1
        print(f"🔧 代码语法修复（第 {attempt} 次）...")
        self._emit("fixer", "start", iteration)

        code = (state.get("reviewer") or state.get("builder") or {}).get("threejs_code", "")
        error = state.get("syntax_error", "")

        if _compact_prompts_enabled():
            prompt = _compose_fixer_prompt_compact(error=error, code=code)
        else:
            prompt = f"""以下代码在 **JavaScript 解析阶段**校验失败（等价于用 `new Function(code)` 做语法检查）。你的任务是把它修到 **能通过解析**，不是排查 WebGL 运行时问题。

【校验失败信息（来自 Node 解析或括号扫描）】
{error}

【原始代码】
```javascript
{code}
```

【职责边界】
- **本会话只处理**：`Unexpected token`、`Unexpected end of input`、未闭合的 `'` / `"` / `` ` ``、注释未闭合、`import`/`export` 顶层语句（须删或改写，本环境为脚本片段无打包器）、重复 `const`/`let` 等同作用域声明、括号/大括号不匹配。
- **本会话不处理**（勿臆测大改材质/渲染）：浏览器控制台里的 **WebGL / `uniform3fv` / @@iterator** 等 **运行时**错误——那由生成规范与 Reviewer 规避；此处 **禁止** 为「可能跑不起来」而整段重写 Three 场景。

【修复要求】
1. 优先检查 **截断**：补全未闭合的 {{ }}、( )、[ ]、模板字符串与多行注释。
2. 确保末尾仍有 **`animate()` 调用**（或等价的 `requestAnimationFrame` 启动）与 **`resize` 监听**（若原文已有则保留）。
3. 保持 **全局 `THREE`**、**无 import/export**、**domElement 挂到 `canvas-container`**；只做解析/结构修复，不删减主体创意逻辑。
4. **InstancedMesh + `setColorAt`**：若原文已有实例色逻辑但缺 `instanceColor` 预分配，可 **仅补** `InstancedBufferAttribute` + `DynamicDrawUsage`（属防崩溃小补全）；勿借机重写整套材质。
5. 输出 **完整** 代码，且 **必须** 用 ```javascript 与 ``` 包裹（便于流水线提取）。

【Three r160 脚本体约束（避免修完仍解析失败）】
- 不要使用 `import` / `export`；不要写 `<script>` 或 HTML。
- 避免把 **Markdown 反引号或说明文字** 粘进 JS 字符串外。"""

        raw = self._llm_call(prompt, code_mode=True, stream_iteration=iteration, stream_agent="fixer")
        fixed = _clamp_animation_speeds_in_js(self._coerce_llm_js_response(raw, code))

        result = dict(state.get("reviewer") or state.get("builder") or {})
        result["threejs_code"] = fixed
        print(f"   ✓ 修复代码 ({len(fixed)} 字符)")
        self._emit("fixer", "done", iteration, f"Syntax fix #{attempt}")

        self._update_log(
            iteration,
            state.get("director"), state.get("narrative"), state.get("visual"),
            state.get("builder"), result,
        )
        return {"reviewer": result, "fix_attempts": attempt}

    # ---- graph nodes (each receives full state, returns partial update) ----

    def _node_director(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        use_chain = state.get("use_chain", False)
        face_influence = state.get("face_influence", "")
        prev_concept = state.get("prev_concept", "")

        print("🎬 [1/6] 创意总监 · 合并叙事策划...")
        self._emit("director", "start", iteration)

        if face_influence:
            influence_prompt = f"""
【外部观察者 / 现场观众（重要！）】
系统刚刚完成一次**摄像头侧写**，「幻棱」侧写引擎的分析简报如下（含可供看板展示的**创作关键词**行，观众会看到这一总结）：
"{face_influence}"

你必须同时完成两件事：
1. **语义转译**：把侧写里的情绪与气质**隐喻化**写入抽象艺术概念，禁止在作品里出现可辨认的真人人脸或写实肖像。
2. **关键词落实**：简报中以「【创作关键词】」列出 2～3 个短语时，须在 `concept`、`theme_focus` 或至少一条 `visual_requirements` 中**显式呼应**这些词（可文学化改写，但不可整轮忽略）；让观众感到「福尔摩斯的锐评」被转译成这一代生成艺术的气质锚点。
"""
        elif use_chain:
            influence_prompt = """
【无现场观察者 · 链式模式】
本轮**没有**新的面部采集与观众侧写：禁止捏造具体观众外貌、表情或「刚才某人」类叙事。
【进化要求】在上一代基础上自然演进，不要完全推翻重来；艺术方向仅靠主题「AI 降临对人世的影响」与上一代结果**自主推演**。
"""
        else:
            influence_prompt = """
【无现场观察者 · 独立成章】
本轮**没有**新的面部采集与观众侧写：禁止虚构观察者或套用不存在的【创作关键词】。
【生成模式】不与上一代绑定，从零提出艺术方向；优先可落地、GPU 友好的抽象表达，由你**独立构思**本轮气质与形式。
"""

        if use_chain:
            director_tail = f"""
上一代核心概念：{prev_concept}
请在上一代的基础上进行深化、变异或情感上的反转（例如从和谐走向冲突，或从混乱走向秩序）。
"""
        else:
            director_tail = """
只需回应本迭代编号与主题，不要假设存在上一代作品。
"""

        prompt = f"""同时扮演【创意总监】与【叙事策划】，一次性产出第{iteration}次迭代的艺术方向与展陈文案。

主题："AI 降临对人世的影响"

## 创意总监部分
【核心要求】：必须是高度抽象（Abstract）、结构复杂（Complex）且极具前卫艺术感的表达！绝对不要具象化或任何现实主义的场景。
【形式与谱系】在抽象前提下主动**拉开套路**：`theme_focus` 或 `visual_requirements` 里须体现可归类的**形式气质**（不必照抄措辞）——例如：生成艺术式秩序、极简场域、混沌与秩序对位、有机抽象、晶体/网格仪式、数据诗学、动态雕塑感、声可视化式律动、单色冥想场、霓虹撞色剧场、故障修辞（仅用几何/闪烁表达，禁止 UI/HUD）。至少有一条 `visual_requirements` **明确指向形态类别**（线场 / 粒子场 / 体块阵列 / 曲线族 / 层叠薄片 / 单表皮呼吸等之一）。
{influence_prompt}
{director_tail}

## 叙事策划部分
偏**展陈大屏**语态：`title` 远看可读，`description` 小字有诗意，`overlay_text` 如诗行/展览标签。三字段须与上面 `concept` / `theme_focus` 的气质**一致咬合**，不要另起炉灶；避免口语与说明书腔。

请输出 JSON 格式（所有字段一次性输出，不要分段）：
```json
{{
    "concept": "核心概念描述（100 字内）",
    "theme_focus": "主题焦点（可含形式气质关键词）",
    "visual_requirements": ["要求 1（建议含主形态倾向）", "要求 2", "要求 3"],
    "title": "English Title | 中文标题（可有副标气质，勿过长）",
    "description": "English description | 中文描述（各约 40 字内；可一格一词概括展陈副标）",
    "overlay_text": "English overlay | 中文悬浮文字（如诗行/展览标签，忌口语与说明书腔）"
}}
```"""

        combined = self._parse_json(self._llm_call(prompt))
        director_result = {
            "concept": combined.get("concept", ""),
            "theme_focus": combined.get("theme_focus", ""),
            "visual_requirements": combined.get("visual_requirements", []),
        }
        narrative_result = {
            "title": combined.get("title", ""),
            "description": combined.get("description", ""),
            "overlay_text": combined.get("overlay_text", ""),
        }
        print(f"   ✓ 概念：{director_result.get('concept', '')[:50]}...")
        print(f"   ✓ 标题：{narrative_result.get('title', '')}")
        self._update_log(iteration, director_result, narrative_result)
        self._emit("director", "done", iteration, director_result.get('concept', '')[:80])
        return {"director": director_result, "narrative": narrative_result}

    def _node_narrative(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        narrative = state.get("narrative", {}) or {}
        title = narrative.get("title", "")

        # Combined with director — this node stays in the graph purely to emit
        # dashboard events so the "叙事策划" card still lights up on B screen.
        print("📖 [2/6] 叙事策划（与创意总监合议完成，镜像事件）...")
        self._emit("narrative", "start", iteration)
        time.sleep(0.6)
        print(f"   ✓ 标题：{title}")
        self._emit("narrative", "done", iteration, title[:80])
        return {}

    def _node_visual(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})
        result2 = state.get("narrative", {})
        use_chain = state.get("use_chain", False)
        prev_visual = state.get("prev_visual", "")

        print("🎨 [3/6] 视觉设计师...")
        self._emit("visual", "start", iteration)

        if use_chain:
            visual_style_line = """2. 要求：表现出极高的视觉复杂度与**可辨识的计算美学**。**不要**每轮停在「旋转粒子球+一两个多面体」的舒适区；每轮在下列方向中做**显性换轨**（仍遵守后续 Builder 的单场景性能上限）：
   - 奇异吸引子或混沌微分方程轨迹；多组曲线在空间中编织「磁带」或年轮
   - 大规模 InstancedMesh：螺旋城墙、沉降晶格、环面透镜阵列、迷宫式体素矩阵
   - 拓扑/分形感：莫比乌斯/克莱因参数化片段、迭代细分形、折纸式折痕动画
   - 动态线框与剖分：嵌套 Wireframe、神经过线、 Voronoi 风格剖分（程序化近似）
   - 流体/群集**拟象**：用粒子或线条的宏观规则模拟「卷吸、涡旋、鸟群转向」（不必物理精确）
   - 光与材质戏剧：强 emissive、冷暖多点光、磷光边、远近明暗剪影（禁止真实 HDRI；勿依赖厚雾糊场）
   - 数字浮雕/层叠：半透明片层前后漂移，形成纵深感（片数勿贪多）
   - 故障修辞：几何抖动、双色分裂、断续闪烁（禁止 mimicking 操作系统界面）"""
            visual_extra = f"""
上一代视觉风格：{prev_visual}
请在上一代视觉基础上，结合新概念进行**彻底的变异**（例如：粒子主导 → 本代改线场主导；曲线主导 → 本代改阵列体块），用颠覆性手段保持展陈新鲜感。
"""
        else:
            visual_style_line = """2. **主形态（必选唯一）**：从下列类别中**只选一类**作为本页主导结构，并在 `rendering_technique` 中写明；尽量与常见「默认粒子球」拉开距离，并与 `theme_focus` 气质一致：
   - **轨迹/曲线族**：参数曲线、Lissajous、折线在 3D 铺展、漂移或缠绕
   - **阵列/晶格**：**单组** InstancedMesh — 螺旋、环带、迷宫、流星雨或沉降矩阵
   - **线场/网络**：LineSegments 构成动态剖分、图式连接或「神经束」
   - **点云/颗粒场**：Points 形成密度层、噪声采样、呼吸式聚散
   - **表皮变形**：单一 BufferGeometry（如平面/球/二十面体）顶点在 JS 中扭结、鼓胀、褶皱
   - **极少形体**：≤5 个简单 mesh，靠尺度比、转速、明暗与撞色构成强构图（雕塑感）
3. **气质修饰**：在 `visual_style_summary` 中并入 1～2 个可感知形容词（如：墓室静穆、日冕炫光、釉质高光、脉冲呼吸、深海冷调、信号波纹、磷光裂隙等）。"""
            visual_extra = """
【独立视觉 · 性能仍优先】仅允许**一条**主形态链路做足时间演化（相位、色相、缓变相机）。禁止 Second 套「海量」系统（例如已 800 实例 InstancedMesh 再叠 3000 Points）；若需点缀，只用**少量**附加线或物体。
"""

        prompt = f"""设计视觉语言。

创意概念：{result1.get('concept', '')}
叙事：{result2.get('title') or result1.get('theme_focus', '')}

【视觉进化要求】
1. 警告：绝对禁止出现任何现实世界的背景、环境贴图（如公园、房间等真实 HDRI）或具象物体！背景应为算法生成的抽象空间或**富有色彩层次的渐变/纯色**——**不要总是使用接近纯黑的背景**，允许深蓝、靛紫、暗绿、深红、炭灰等**有明度的暗色调**，让前景元素有足够的对比度和辨识度。**大气透视**：优先靠明暗、色相层次与粒子渐隐表达纵深；若不用雾，可完全不设 `scene.fog`。
{visual_style_line}
{visual_extra}

请输出 JSON 格式：
```json
{{
    "visual_style_summary": "English Style | 中文风格总结",
    "color_palette": {{"background": ["#0d1b2a", "#1b263b"], "primary": ["#667eea", "#764ba2"], "accent": ["#f093fb"]}},
    "rendering_technique": "指定一种具体的技术手段 (如 InstancedMesh, 线条拓扑, 数学分形, 发光点云等)"
}}
```"""

        result = self._parse_json(self._llm_call(prompt))
        print(f"   ✓ 风格：{result.get('visual_style_summary', '')}")
        self._update_log(iteration, result1, result2, result)
        self._emit("visual", "done", iteration, result.get('visual_style_summary', '')[:80])
        return {"visual": result}

    def _node_builder(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})
        result2 = state.get("narrative", {})
        result3 = state.get("visual", {})
        use_chain = state.get("use_chain", False)
        prev_code = state.get("prev_code", "")

        # 必须用「是否含 total_score」判断，不能用 bool(score)：score 为 0 时 bool(0) 为 False，会导致 regen_attempts 不递增、Critic 低分路径无限重生。
        cd0 = state.get("critic") or {}
        is_regen = isinstance(cd0, dict) and "total_score" in cd0
        regen_count = state.get("regen_attempts", 0) + (1 if is_regen else 0)

        label = '[重生]' if is_regen else '[4/6]'
        suffix = '（评审反馈驱动）' if is_regen else ''
        print(f"💻 {label} 技术实现{suffix}...")
        self._emit("builder", "regen" if is_regen else "start", iteration)

        critic_block = ""
        if is_regen:
            cd = state.get("critic", {})
            critic_block = f"""
【上一轮评审反馈 — 本次必须针对性改进】
评分：{cd.get('total_score', 'N/A')}/10
结论：{cd.get('conclusion', '')}
改进建议：{'；'.join(cd.get('suggestions', []))}
请大幅提升视觉复杂度与艺术性，直接回应以上每条建议。不要输出与上一轮相同的代码。"""

        technique = result3.get('rendering_technique', '粒子或几何体')

        if _compact_prompts_enabled():
            prompt = _compose_builder_prompt_compact(
                technique=technique,
                style=result3.get("visual_style_summary", "") or "",
                concept=result1.get("concept", "") or "",
                critic_block=critic_block,
                use_chain=use_chain,
                prev_code=prev_code,
            )
        else:
            prompt = f"""编写基于 Three.js 的生成艺术代码。

视觉风格：{result3.get('visual_style_summary', '')}
概念：{result1.get('concept', '')}

【创意与艺术性 — 最高优先级】
1. 本轮渲染技术：「{technique}」。用 InstancedMesh / Points / LineSegments / BufferGeometry 等实现**清晰可辨的独特形式**；**严禁**「泛用旋转粒子球」或「球面散点+连线」等惰性套路。落实调色板（多点光色温、material.color / emissive / metalness）、群体相位错频与相机慢遥，使**远观**也能感知本轮独特气质。
2. **动画呼吸感**（⚠️ 最常见致命错误：速度常数太小导致画面看似完全静止）：
   设 `const t = performance.now() * 0.001`（单位：秒），则：
   ① 快层（周期 2–8 s，speed 系数 `0.8 ~ 3.0`）：如 `Math.sin(t * 1.2 + phase)`、`Math.cos(t * 2.5)`。用于几何自转、emissive 脉冲、粒子色彩闪烁。
   ② 慢层（周期 30–90 s，speed 系数 `0.02 ~ 0.2`）：如 `camera.position.x = R * Math.cos(t * 0.07)`。用于相机漂移、全局色温缓移、**背景色**或 emissive 的极缓呼吸；**勿**把 `FogExp2.density` 拉到肉眼「灰幕」级别（若动雾，振幅 ≤ 0.003）。
   **校验公式**：周期 T = 2π / speed ≈ 6.28 / speed。speed=0.001 → T=6280 秒 ≈ **永远看不到变化**。**任何动画参数 speed < 0.01 都是错误的**——肉眼完全无法感知。相机轨道 speed 推荐 `0.04–0.12`，实例运动 speed 推荐 `0.5–2.0`。
3. **画面亮度**：`renderer.toneMapping = THREE.ACESFilmicToneMapping`，`toneMappingExposure >= 1.2`；至少一盏 AmbientLight 强度 >= 0.3 + 一盏方向/点光强度 >= 0.8；`scene.background` 避免纯黑，推荐 HSL 中 L >= 8%。
4. **雾与空气感（避免「灰雾糊墙」）**：无明确「浓雾/霾」叙事时**优先 `scene.fog = null`**，仅靠背景色与光照塑造空间。若使用 `FogExp2`，**密度 ≤ 0.018**（推荐 **0.008～0.015**）；**禁止** `0.03` 及以上或每帧把密度推到发灰。`THREE.Fog` 线性雾须保证 **`far` 明显大于**主体分布半径（远处仍留结构，不要近裁切糊成一团）。
5. **追求结构复杂度与前卫感**：多层叠加、空间编织、参数曲线缠绕、分形细分、拓扑变形、吸引子轨迹等。让代码本身成为算法艺术品，不是技术 demo。
6. **多层视觉系统**：场景中至少 2 种不同几何体类型（如 InstancedMesh + Points、Mesh + LineSegments、大 Mesh + 围绕粒子群等），且各层使用**不同材质参数与色相**。严禁全场只有一个孤零零的单色物体——即使概念是"极简"，也需至少有主体 + 背景粒子/线条/光晕等辅助层。
7. **色彩动态**：animate 中必须有至少一处 `.color.setHSL()` 或 `.emissive.setHSL()` 随时间变化（哪怕幅度极小如 hue ±0.02）。纯单色静态着色 = 失败。
{critic_block}{_technique_seed(technique)}
{_build_code_inherit_block(use_chain, prev_code)}

【技术规范（精简）】
- 已引入全局 `THREE`（Three.js r160，与 evolution 页 `three.min.js` 一致）。`renderer.domElement` 插入 `getElementById('canvas-container')`。
- 推荐在 `WebGLRenderer` 创建后设置：`renderer.outputColorSpace = THREE.SRGBColorSpace`（色彩与显示器一致；不设通常也可运行）。
- 必须有 Scene + PerspectiveCamera + WebGLRenderer + animate()（含 requestAnimationFrame + renderer.render）+ resize 监听。
- 输出纯 JavaScript，不含 HTML / Markdown / `<script>` 标签。
- 零外部依赖：禁止 OrbitControls、任何 Loader（GLTF/Font/Texture/Cube）、EffectComposer。
- 禁止用户交互（mousemove / pointermove / touch / wheel / keydown），纯自动演化。
- 所有数据**程序化生成**（for / while + Math），禁止硬编码超长数组。
- **注释最小化**：禁止任何 `/* ... */` 块注释与中文解说；`//` 行内注释仅在**不可替代**处（如罕见算法参考、WebGL 坑位提醒）使用，整段代码合计不超过 **3 条**。不要写章节横幅（`// ============ 场景初始化 ============`）、步骤编号（`// 1. 创建场景`）或重述代码作用的行尾注释。清晰命名优先，评审质量完全不依赖注释。

【性能上限】
- Points 粒子 ≤ 4000；单 InstancedMesh ≤ 1200，全部 InstancedMesh 合计 ≤ 2500。
- 禁止同时叠加多种「海量实例」系统。

【Three r160 兼容（精简）】
- 禁止 ShaderMaterial / RawShaderMaterial / 自定义 GLSL，只用内置材质，顶点动画在 JS animate 循环中计算。
- 禁止 THREE.Geometry / Face3 / fromGeometry()，只用 BufferGeometry + setAttribute。
- 禁止 `THREE.TorusKnot`（r160 无此曲线类）；需结线轨迹时用 `THREE.TorusKnotGeometry` 或对中心线参数 `u` 按官方公式采样点/切线，勿调用不存在的 `.getPoints()`。
- `color` / `emissive` / `sheenColor` 等：构造参数里用 `new THREE.Color(0x……)` 或 `.setHex()` / `.set()`；animate 里对已有 Color 再 `.copy()` / `.setHSL()`。避免把「应持续为 Color 对象」的属性改成裸数字或普通对象（易在 uniform 上传时报错）。
- **InstancedMesh（含 `setColorAt` / `vertexColors` / 每帧改实例色）**：主体材质**优先 `MeshStandardMaterial`**（metalness / roughness / emissive 即可做出金属与高光感）。**规避** `MeshPhysicalMaterial` + **`sheen > 0`**（及 `sheenColor` / `sheenRoughness`）与该组合同用——在部分 WebGL2 上会触发 **`uniform3fv` /「@@iterator」类 TypeError**（与 r160 引擎 + 升级后 shader 路径有关）。若必须用 Physical（如 clearcoat）且带实例色，则 **`sheen` 必须为 0** 且不传 sheen 相关色参。
- 非 Instanced、单体 Mesh 使用 `MeshPhysicalMaterial` 时：`sheen` 为 **0～1 数值**；`sheenColor` 用 `new THREE.Color()`。**禁止**把 `THREE.Color` 赋给 `sheen`。尽量不要用 `transmission`+`thickness` 薄玻璃栈（易与光照不匹配）；非需要时不要设 `thickness`。
- **禁止** `renderer.outputEncoding`、`THREE.sRGBEncoding`、`THREE.LinearEncoding` 等已移除 API；统一用 `renderer.outputColorSpace = THREE.SRGBColorSpace`。
- 对 `sheenColor`、`clearcoatNormalMap` 等调用 `.copy`/`.setHex` 前，确认材质实例上该属性存在。
- **InstancedMesh 每帧 `setColorAt`**：须在首帧前创建 `instanceColor`（`new THREE.InstancedBufferAttribute(new Float32Array(count*3), 3)`）并 `setUsage(THREE.DynamicDrawUsage)`，勿依赖首次 `setColorAt` 隐式创建（易在 WebGL2 上与 uniform 路径叠加出问题）。`instanceColor` 必须赋给 **`mesh.instanceColor`**，勿用 `geometry.setAttribute('instanceColor', …)` 冒充实例色。读回颜色用 **`mesh.getColorAt(i, color)`**，禁止对 `InstancedBufferAttribute` 调用 `.get`（r160 无此 API）。
- **InstancedMesh 每帧改矩阵**：`getMatrixAt(i, matrix)` 的 `matrix` 须为**单独的 `THREE.Matrix4()`**，再 `decompose` 到独立 `Vector3`/`Quaternion`/`Vector3`，赋值给 `Object3D` 后 `updateMatrix()`；**禁止**把 `dummy.matrix` 既当 `getMatrixAt` 写入目标又立刻 `dummy.matrix.decompose(...)`（易触发内部 `Matrix4.copy(undefined)` 类运行时错误）。
- **版本对齐**：生成代码须与页内 **Three.js r160（three.min.js）** 一致；勿混用其它大版本的 API（升级/换 CDN 后常见「能编译、渲染期 uniform 崩」）。

【输出格式】
不要输出 JSON。将完整 Three.js 代码写在 ```javascript 和 ``` 之间。"""

        raw = self._llm_call(prompt, code_mode=True, stream_iteration=iteration, stream_agent="builder")
        extracted_code = self._extract_js_code(raw)
        if not extracted_code:
            print("\n   [警告] Builder 未能生成合法的代码块，返回为空")
            extracted_code = ""
        result = {"threejs_code": extracted_code}
        print(f"   ✓ 代码已生成 ({len(extracted_code)} 字符)")
        self._update_log(iteration, state.get("director"), state.get("narrative"), result3, result)
        self._emit("builder", "done", iteration, f"Code generated ({len(extracted_code)} chars)")
        return {"builder": result, "regen_attempts": regen_count, "syntax_valid": False, "syntax_error": "", "fix_attempts": 0}

    def _node_reviewer(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result4 = state.get("builder", {})

        # 深度复查已合并进 Judge（见 _node_critic）。此处仅做程序化预检：
        # ① 克隆 Builder 代码到 reviewer 槽位，作为 syntax_guard / code_fixer 的修复目标；
        # ② 对过小的 speed 赋值统一 clamp；
        # ③ 发射 start/done 事件，让 dashboard 的「代码审查」卡片仍然亮起。
        print("🔬 [5/6] 代码审查（程序化预检，深度复查合并至 Judge）...")
        self._emit("reviewer", "start", iteration)

        builder_code = result4.get("threejs_code", "")
        clamped = _clamp_animation_speeds_in_js(builder_code)
        result = {
            "threejs_code": clamped,
            "review_comments": "程序化速度参数校准完成；深度复查已合并至 Judge。",
        }
        time.sleep(0.6)
        print("   ✓ 预检完毕，交由 Judge 合议")
        self._emit("reviewer", "done", iteration, "速度参数校准完成 · 交由 Judge 合议")
        self._update_log(
            iteration,
            state.get("director"), state.get("narrative"), state.get("visual"),
            state.get("builder"), result,
        )
        return {"reviewer": result}

    def _node_critic(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})
        result3 = state.get("visual", {})
        code = state.get("reviewer", {}).get("threejs_code", "") or state.get("builder", {}).get("threejs_code", "")

        code_len = len(code)
        has_setHSL = ".setHSL(" in code
        geo_types = sum(["THREE.Points" in code or "Points(" in code,
                         "InstancedMesh" in code,
                         "LineSegments" in code or "Line(" in code,
                         "new THREE.Mesh" in code])

        code_summary = f"代码统计:{code_len} 字符 | 几何体类型数:{geo_types} | 色彩动态(.setHSL):{has_setHSL}"

        print("🔍 [6/6] 艺术评审 · Judge 合议（合并 Reviewer 复查）...")
        self._emit("critic", "start", iteration)

        if _compact_prompts_enabled():
            prompt = _compose_judge_prompt_compact(
                concept80=(result1.get("concept", "") or "")[:80],
                visual80=(result3.get("visual_style_summary", "") or "")[:120],
                code_summary=code_summary,
            )
        else:
            prompt = f"""你是 Judge，合并原先 Reviewer 代码复查与 Critic 艺术评审两道流程，一次性产出两段输出。

## 职责 1 · 代码复查（原 Reviewer）
快速扫视代码，只产出「观察文本」——**不回写代码**（语法校验与 clamp 已由 syntax_guard / 程序化预检完成）。重点关注：
- 依赖越界：OrbitControls / GLTFLoader / EffectComposer 等未引入依赖
- 禁用 API：ShaderMaterial / RawShaderMaterial / THREE.Geometry / Face3 / 已移除的 outputEncoding / sRGBEncoding
- 运行时风险：InstancedMesh 未预分配 instanceColor、MeshPhysicalMaterial + sheen > 0 与实例色同用、材质属性错误赋值、undefined 调用
- 交互监听残留（mousemove / pointermove / keydown 等）
- 动画速度残留极小值（程序已 clamp，通常无需复述）
如无问题写「代码结构健康」。1~2 句即可，勿展开成长篇审查单。

## 职责 2 · 艺术评审（原 Critic）
原则：**偏宽容、看整体可展陈潜力**，不要拿「美术馆终审」标准卡生成艺术；代码统计仅供参考，**勿**因缺 .setHSL 或几何体种类数少就压分。
1. 概念与画面气质是否大致咬合（允许执行粗糙）
2. 是否至少有**一处**可感知亮点:动效、光影层次、独特点线面构成、色彩氛围其一即可

概念:{result1.get('concept', '')[:80]}
视觉:{result3.get('visual_style_summary', '')}
{code_summary}

评分参考（整体分数**宁高勿低**，除非明确翻车）:
- 8.5-10:有辨识度、能站住展陈；不必完美
- 7.0-8.4:诚意够、能跑能看，有瑕疵也**正常给 7.5+**，勿因「不够前卫」硬压到 6 段
- 5.0-6.9:明显套路空壳、几乎静止无设计、或与概念**严重**脱节才给这段

**publish_ready（再放宽）**:
- `true`:**总分 ≥ 7.0** 即可倾向 `true`，只要**不是**纯占位几何 + 完全无动势/无氛围；**禁止**因「少一层粒子」「没写 setHSL」就判 `false`。
- **总分 ≥ 7.5**:默认 `true`，除非有硬伤（严重跑题、像未完成的灰盒）。
- **总分 ≥ 8.0**:**应当 `true`**，除非极个别硬伤。
- `false`:仅当总分 **< 6.5**，或 **6.5–6.9** 且确认是套路空壳/严重脱节；**不要**因「还能再 polish」或「不够惊艳」判 `false`。

## 输出 JSON（两段合一）
```json
{{
    "review_comments": "1~2 句代码层面观察，无问题写「代码结构健康」",
    "total_score": 7.5,
    "conclusion": "（必填）1~3 句中文定论:是否达展陈标准、形式与概念是否咬合、最突出得失；勿只罗列优点。",
    "strengths": ["优点 1", "优点 2"],
    "suggestions": ["建议 1"],
    "publish_ready": false,
    "next_iteration_focus": "下次迭代重点"
}}
```"""

        result = self._parse_json(self._llm_call(prompt))
        if not (result.get("conclusion") or "").strip():
            score_disp = result.get("total_score", "N/A")
            result["conclusion"] = (
                f"未返回完整结论文本。当前总分 {score_disp}/10，请结合 strengths、suggestions 理解评审意涵。"
            )

        # Judge 也同时产出 review_comments —— 回写到 reviewer 状态槽，让 dashboard 的
        # 「代码审查」时间线条目拿到真正的复查结论，而不是程序化预检的占位文本。
        reviewer_state = dict(state.get("reviewer") or {})
        judge_review = (result.get("review_comments") or "").strip()
        if judge_review:
            reviewer_state["review_comments"] = judge_review

        print(
            f"   ✓ 评分:{result.get('total_score', 'N/A')}/10 — "
            f"{(result.get('conclusion') or '')[:72]}{'…' if len((result.get('conclusion') or '')) > 72 else ''}"
        )
        self._update_log(
            iteration,
            state.get("director"), state.get("narrative"), state.get("visual"),
            state.get("builder"), reviewer_state, result,
        )
        conclusion = (result.get('conclusion') or '')[:80]
        score = result.get('total_score', 'N/A')
        self._emit("critic", "done", iteration, f"Score: {score}/10 — {conclusion}")
        return {"critic": result, "reviewer": reviewer_state}

    # ---- public entry point ----

    def run(self, iteration: int = None, face_influence: str = None, creative_keywords: list = None):
        if iteration is None:
            state_file = self.project_dir / "state.json"
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                iteration = state.get("iteration", 0) + 1
            else:
                iteration = 1

        print(f"\n{'='*60}")
        print(f"🎨 第 {iteration} 次迭代 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        prev_data = {}
        if iteration > 1:
            log_file = self.project_dir / "iteration_log.json"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                        if logs:
                            prev_data = logs[-1]
                except Exception:
                    pass

        use_chain = _evolution_chain_from_env() and iteration > 1
        if iteration > 1 and not use_chain:
            print("   ℹ 生成策略：独立成章（未设置 AGENTSART_EVOLUTION_CHAIN=1，不继承上一代以控制性能）")

        prev_concept = prev_data.get("director", {}).get("concept", "无（这是初代）")
        prev_visual = prev_data.get("visual", {}).get("visual_style_summary", "无（这是初代）")
        prev_code = prev_data.get("builder", {}).get("threejs_code", "无")
        if not use_chain:
            prev_concept = "（无：本轮独立生成）"
            prev_visual = "（无：本轮独立生成）"
            prev_code = ""

        initial_state: PipelineState = {
            "iteration": iteration,
            "face_influence": face_influence or "",
            "creative_keywords": creative_keywords or [],
            "use_chain": use_chain,
            "prev_concept": prev_concept,
            "prev_visual": prev_visual,
            "prev_code": prev_code,
            "director": {},
            "narrative": {},
            "visual": {},
            "builder": {},
            "reviewer": {},
            "critic": {},
            "syntax_valid": False,
            "syntax_error": "",
            "fix_attempts": 0,
            "regen_attempts": 0,
        }

        final = self._graph.invoke(initial_state)

        self._generate_output(
            iteration, final, creative_keywords or [],
        )
        return iteration

    # ---- output generation (unchanged logic) ----

    def _generate_output(self, iteration: int, state: dict, creative_keywords: list):
        print("📄 生成 HTML...")

        result1 = state.get("director", {})
        result2 = state.get("narrative", {})
        result3 = state.get("visual", {})
        result4 = state.get("builder", {})
        result5 = state.get("reviewer", {})
        result6 = state.get("critic", {})

        default_threejs = result5.get('threejs_code') or result4.get('threejs_code', '')

        if len(default_threejs) < 100:
            default_threejs = '''
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
const material = new THREE.MeshNormalMaterial({ wireframe: true });
const torusKnot = new THREE.Mesh(geometry, material);
scene.add(torusKnot);

camera.position.z = 30;

function animate() {
    requestAnimationFrame(animate);
    torusKnot.rotation.x += 0.01;
    torusKnot.rotation.y += 0.01;
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
'''

        kw_display = ''
        if creative_keywords:
            kw_display = ''.join(f'<span class="kw-tag">{k}</span>' for k in creative_keywords[:3])

        replacements = {
            "title": result2.get("title", "The Arrival"),
            "description": result2.get("description", "A Multi-Agent AI Art Piece"),
            "iteration": str(iteration),
            "visual_style_summary": result3.get("visual_style_summary", ""),
            "narrative_summary": result2.get("overlay_text", ""),
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "threejs_code": default_threejs,
            "director_concept": (result1.get("concept") or "").strip() or "—",
            "creative_keywords_display": kw_display,
        }
        html = self.html_template
        for key, value in replacements.items():
            html = html.replace(f"{{{key}}}", str(value))

        iterations_dir = self.project_dir / "iterations"
        if iterations_dir.exists():
            for old_version in iterations_dir.glob("v*"):
                if old_version.is_dir():
                    shutil.rmtree(old_version)
        else:
            iterations_dir.mkdir(parents=True, exist_ok=True)

        html_file = iterations_dir / "the-arrival.html"
        _atomic_write(html_file, html)
        _atomic_write(self.project_dir / "evolution.html", html)

        _atomic_write_json(iterations_dir / "metadata.json", {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "director": result1,
            "narrative": result2,
            "visual": result3,
            "builder": result4,
            "reviewer": result5,
            "critic": result6,
        })

        _atomic_write_json(self.project_dir / "state.json", {
            "iteration": iteration,
            "last_run": datetime.now().isoformat(),
            "latest_title": result2.get("title", ""),
            "latest_concept": result1.get("concept", ""),
            "latest_score": result6.get("total_score", 0),
        })

        print(f"   ✓ 已保存至 {html_file}")
        print(f"   ✓ 已更新 evolution.html")
        print(f"   🎉 迭代 {iteration} 完成!\n")

    def _update_log(self, iteration, result1=None, result2=None, result3=None, result4=None, result5=None, result6=None):
        log_file = self.project_dir / "iteration_log.json"

        log_entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
        }
        if result1:
            log_entry["director"] = result1
        if result2:
            log_entry["narrative"] = result2
        if result3:
            log_entry["visual"] = result3
        if result4:
            log_entry["builder"] = result4
        if result5:
            log_entry["reviewer"] = result5
        if result6:
            log_entry["critic"] = result6

        with _log_lock:
            logs = []
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    try:
                        logs = json.load(f)
                    except Exception:
                        pass

            idx = next((i for i, log in enumerate(logs) if log.get("iteration") == iteration), -1)
            if idx >= 0:
                logs[idx].update(log_entry)
            else:
                logs.append(log_entry)

            if _ITERATION_LOG_MAX_ENTRIES > 0 and len(logs) > _ITERATION_LOG_MAX_ENTRIES:
                logs = logs[-_ITERATION_LOG_MAX_ENTRIES:]
            _atomic_write_json(log_file, logs)


if __name__ == "__main__":
    orchestrator = FullOrchestrator()
    orchestrator.run()
