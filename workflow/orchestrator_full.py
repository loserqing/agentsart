#!/usr/bin/env python3
"""
完整版多智能体协作编排器 - 生成可运行的 WebGPU 作品
技术栈：LangGraph (StateGraph) + langchain-google-genai
"""

import os
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).parent.parent / ".env")


# --------------- helpers ---------------

def _evolution_chain_from_env() -> bool:
    return os.getenv("AGENTSART_EVOLUTION_CHAIN", "").strip().lower() in ("1", "true", "yes")


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


def _normalize_model_name(raw: str) -> str:
    """Strip litellm-style 'gemini/' prefix so langchain-google-genai gets a clean model id."""
    name = raw.strip()
    if name.startswith("gemini/"):
        name = name[len("gemini/"):]
    return name


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


# --------------- orchestrator class ---------------

class FullOrchestrator:
    def __init__(self, project_dir: str = None, model=None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).parent.parent
        raw_model = model or os.getenv("DEFAULT_MODEL", "gemini/gemini-2.0-flash")
        self.model_name = _normalize_model_name(raw_model)

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.85,
        )
        self.llm_code = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.5,
        )

        _tpl_path = Path(__file__).parent / "evolution_template.html"
        self.html_template = _tpl_path.read_text(encoding="utf-8")

        self._graph = self._build_graph()

    # ---- graph construction ----

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("director", self._node_director)
        g.add_node("narrative", self._node_narrative)
        g.add_node("visual", self._node_visual)
        g.add_node("builder", self._node_builder)
        g.add_node("reviewer", self._node_reviewer)
        g.add_node("critic", self._node_critic)
        g.add_edge(START, "director")
        g.add_edge("director", "narrative")
        g.add_edge("narrative", "visual")
        g.add_edge("visual", "builder")
        g.add_edge("builder", "reviewer")
        g.add_edge("reviewer", "critic")
        g.add_edge("critic", END)
        return g.compile()

    # ---- LLM helper ----

    def _llm_call(self, prompt: str, *, code_mode: bool = False) -> str:
        llm = self.llm_code if code_mode else self.llm
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content or ""

    # ---- JSON / code parsers (unchanged) ----

    def _parse_json(self, text: str) -> dict:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e:
                print(f"\n   [警告] JSON 代码块解析失败: {e}")

        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception as e:
                print(f"\n   [警告] 提取花括号内容解析失败: {e}")

        js_match = re.search(r'```(?:javascript|js)\s*(.*?)\s*```', text, re.DOTALL)
        if js_match:
            print("\n   [恢复] JSON 解析失败，但成功从 Markdown 中提取到了代码块")
            return {"threejs_code": js_match.group(1).strip()}

        print(f"\n   [警告] AI 没有返回合法的 JSON！截取前 100 个字符: {text[:100]}...")
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

    # ---- graph nodes (each receives full state, returns partial update) ----

    def _node_director(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        use_chain = state.get("use_chain", False)
        face_influence = state.get("face_influence", "")
        prev_concept = state.get("prev_concept", "")

        print("🎬 [1/6] 创意总监...")

        if face_influence:
            influence_prompt = f"""
【外部观察者 / 现场观众（重要！）】
系统刚刚完成一次**摄像头侧写**，「AgentsArt-X」风格的分析简报如下（含可供看板展示的**创作关键词**行，观众会看到这一总结）：
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

        prompt = f"""定义第{iteration}次迭代的艺术方向。

主题："AI 降临对人世的影响"
【核心要求】：必须是高度抽象（Abstract）、结构复杂（Complex）且极具前卫艺术感的表达！绝对不要具象化或任何现实主义的场景。
【形式与谱系】在抽象前提下主动**拉开套路**：`theme_focus` 或 `visual_requirements` 里须体现可归类的**形式气质**（不必照抄措辞）——例如：生成艺术式秩序、极简场域、混沌与秩序对位、有机抽象、晶体/网格仪式、数据诗学、动态雕塑感、声可视化式律动、单色冥想场、霓虹撞色剧场、故障修辞（仅用几何/闪烁表达，禁止 UI/HUD）。至少有一条 `visual_requirements` **明确指向形态类别**（线场 / 粒子场 / 体块阵列 / 曲线族 / 层叠薄片 / 单表皮呼吸等之一）。
{influence_prompt}
{director_tail}

请输出 JSON 格式：
```json
{{
    "concept": "核心概念描述（100 字内）",
    "theme_focus": "主题焦点（可含形式气质关键词）",
    "visual_requirements": ["要求 1（建议含主形态倾向）", "要求 2", "要求 3"]
}}
```"""

        result = self._parse_json(self._llm_call(prompt))
        print(f"   ✓ 概念：{result.get('concept', '')[:50]}...")
        self._update_log(iteration, result)
        return {"director": result}

    def _node_narrative(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})

        print("📖 [2/6] 叙事策划...")

        prompt = f"""基于以下创意方向构建叙事（偏**展陈大屏**：标题远看可读，小字有诗意）：
创意概念：{result1.get('concept', '')}
主题焦点：{result1.get('theme_focus', '')}

请输出 JSON 格式：
```json
{{
    "title": "English Title | 中文标题（可有副标气质，勿过长）",
    "description": "English description | 中文描述（各约 40 字内；可一格一词概括展陈副标）",
    "overlay_text": "English overlay | 中文悬浮文字（如诗行/展览标签，忌口语与说明书腔）"
}}
```"""

        result = self._parse_json(self._llm_call(prompt))
        print(f"   ✓ 标题：{result.get('title', '')}")
        self._update_log(iteration, result1, result)
        return {"narrative": result}

    def _node_visual(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})
        result2 = state.get("narrative", {})
        use_chain = state.get("use_chain", False)
        prev_visual = state.get("prev_visual", "")

        print("🎨 [3/6] 视觉设计师...")

        if use_chain:
            visual_style_line = """2. 要求：表现出极高的视觉复杂度与**可辨识的计算美学**。**不要**每轮停在「旋转粒子球+一两个多面体」的舒适区；每轮在下列方向中做**显性换轨**（仍遵守后续 Builder 的单场景性能上限）：
   - 奇异吸引子或混沌微分方程轨迹；多组曲线在空间中编织「磁带」或年轮
   - 大规模 InstancedMesh：螺旋城墙、沉降晶格、环面透镜阵列、迷宫式体素矩阵
   - 拓扑/分形感：莫比乌斯/克莱因参数化片段、迭代细分形、折纸式折痕动画
   - 动态线框与剖分：嵌套 Wireframe、神经过线、 Voronoi 风格剖分（程序化近似）
   - 流体/群集**拟象**：用粒子或线条的宏观规则模拟「卷吸、涡旋、鸟群转向」（不必物理精确）
   - 光与材质戏剧：强 emissive、冷暖多点光、磷光边、雾中剪影（禁止真实 HDRI）
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
叙事：{result2.get('title', '')}

【视觉进化要求】
1. 警告：绝对禁止出现任何现实世界的背景、环境贴图（如公园、房间等真实 HDRI）或具象物体！背景应为算法生成的抽象空间或**富有色彩层次的渐变/纯色**——**不要总是使用接近纯黑的背景**，允许深蓝、靛紫、暗绿、深红、炭灰等**有明度的暗色调**，让前景元素有足够的对比度和辨识度。
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
        return {"visual": result}

    def _node_builder(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result1 = state.get("director", {})
        result2 = state.get("narrative", {})
        result3 = state.get("visual", {})
        use_chain = state.get("use_chain", False)
        prev_code = state.get("prev_code", "")

        print("💻 [4/6] 技术实现...")

        prompt = f"""编写基于 Three.js 的生成艺术代码。

视觉：{result3.get('visual_style_summary', '')}

【严格技术规范 - 必须遵守】
1. 前端已引入 `THREE`，请直接使用（无需 import）。
2. 必须将生成的 renderer 的 canvas 插入到指定的容器中：`document.getElementById('canvas-container').appendChild(renderer.domElement);`
3. 必须包含基础要素：`Scene`, `PerspectiveCamera`, `WebGLRenderer`。
4. 必须有 `animate()` 循环，内部必须调用 `requestAnimationFrame(animate);` 和 `renderer.render(scene, camera);`。
5. 监听窗口调整事件以更新宽高比。
6. 生成的代码必须是纯 JavaScript 代码，绝不能包含 ```javascript 等 Markdown 标签或 <script> 标签。
7. 彻底封杀外部贴图：绝对禁止使用 `CubeTextureLoader` 或 `TextureLoader` 加载任何现实世界的网络图片/环境图！
8. 艺术性与复杂度（**展陈辨识度**）：结合前面指定的渲染技术（{result3.get('rendering_technique', '粒子或几何体')}），用 `THREE.InstancedMesh`、`THREE.Points`、`THREE.LineSegments` 等实现**清晰可辨的形式**；避免「泛用旋转粒子球」式惰性设计——落实调色板（多点光色温、`material.color`/`emissive`/`metalness`）、群体相位错频与相机慢遥，使**远观**也能感知本轮独特气质。
9. 零外部依赖：绝对禁止使用 `GLTFLoader`, `FontLoader` 等任何外部资源加载器！所有视觉必须纯通过数学和代码（Procedural）生成。
10. 禁用附加库与交互输入：绝对禁止使用 `THREE.OrbitControls`、`EffectComposer` 等需要额外引入的扩展库；同时**禁止任何用户交互输入逻辑**（如 `mousemove`、`pointermove`、`touch*`、`wheel`、`keydown`）。作品必须是纯自动演化播放。
11. 性能与安全（硬性约束）：**全程可流畅 60fps 优先**。`THREE.Points` 总粒子数必须 <= 4000；`InstancedMesh` 单个 mesh 实例数必须 <= 1200，且若存在多个 InstancedMesh，**全部实例数之和**必须 <= 2500。禁止在同一个作品里同时使用多种"海量实例"（例如 3 组 InstancedMesh 各 1200 + 5000 点云）。默认单场景总 draw 压力越小越好。
12. 绝对禁用 ShaderMaterial：为了防止 GLSL 编译报错，**绝对禁止**使用 `ShaderMaterial`、`RawShaderMaterial` 或编写任何自定义着色器代码！所有材质必须使用 Three.js 自带的内置材质（如 `MeshPhysicalMaterial`, `MeshStandardMaterial`, `PointsMaterial` 等），顶点动画必须在 JS 的 animate 循环中通过修改 BufferGeometry 计算！
13. 容错与兼容性：运行环境为 Three.js r128。为防止 `Cannot read properties of undefined` 报错，请优先操作 `material.color` 或 `material.emissive`。如果修改高级材质属性（如 `sheen`, `clearcoat`），必须在更新前进行非空安全检查（如 `if (material.sheen) {{ ... }}`）。
14. 颜色赋值规范：绝对不能直接给颜色属性赋值数字（例如 `material.color = 0xff0000` 会导致 `uniform3fv` 崩溃报错）。必须使用 `material.color.setHex(0xff0000)` 或 `material.color.setHSL(...)`。**特别警告：在 r128 版本中，MeshPhysicalMaterial 的 `sheen` 属性必须是一个 `THREE.Color` 对象！** 绝对不能写成 `sheen: 1.0` 这种数字，否则会引发 `uniform3fv` 的致命报错。r128 没有 `sheenColor` 属性，请直接使用 `sheen: new THREE.Color(...)`。**另：r128 的 MeshPhysicalMaterial 没有 `thickness` 属性**（传入会在控制台报「is not a property」）；需要半透明/折射感时请只用 `transmission`、`opacity`、`ior` 等已支持字段，**禁止**写 `thickness:`。
15. 废弃 API 拦截：运行环境为 Three.js r128，该版本已完全移除 `Geometry` 类，只保留 `BufferGeometry`。绝对禁止使用 `new THREE.Geometry()`、`THREE.Face3` 或 `new THREE.BufferGeometry().fromGeometry()` 等废弃 API，否则会引发 `fromGeometry is not a function` 错误！请直接使用 `THREE.BufferGeometry`，并通过设置 `position` 等属性 (`setAttribute('position', new THREE.Float32BufferAttribute(..., 3))`) 来创建自定义几何体，或者直接使用内置的几何体（如 `THREE.SphereGeometry`, `THREE.BoxGeometry` 等）。
16. **严格语法与长度警告**：为了避免大模型输出被截断导致 `Unexpected end of input` 的语法错误，**所有模型的数据点（如顶点、颜色、向量）必须通过 `for` 或 `while` 循环配合 `Math.sin`/`Math.random` 算法进行程序化生成（Procedural generation）**！绝对禁止在代码中硬编码任何超长的大型数组（如 `[1.0, 2.0, 0.5, ... 几千个数字]`）！这会导致代码断头崩溃。确保所有的括号和逗号完美匹配。
17. **动画呼吸感与时间层次**：代码必须包含**至少两层不同速率**的连续运动——① 快层（2–8 秒周期）：几何体自转、粒子相位偏移、颜色脉冲或 emissive 闪烁；② 慢层（20–60 秒周期）：相机缓慢轨道漂移（如 `camera.position.x = R * Math.cos(time * 0.05)`）、全局色温缓移（修改灯光 `.color.setHSL(...)` ）、或雾密度吐纳。让不同停留时长的观众都能感知到变化。
18. **画面亮度与可读性**：作品在展陈大屏上须保持**足够的视觉亮度**——设置 `renderer.toneMapping = THREE.ACESFilmicToneMapping` 和 `renderer.toneMappingExposure = 1.2`（或更高，视场景而定）；场景中至少有一盏 `AmbientLight` 强度 >= 0.3 和一盏方向/点光强度 >= 0.8；`scene.background` 避免使用接近纯黑（`#000`~`#0a0a0a`）的颜色，推荐 HSL 中 L >= 8% 的深色调。
{_technique_seed(result3.get('rendering_technique', ''))}
{_build_code_inherit_block(use_chain, prev_code)}

【重要输出格式】
为了避免 JSON 解析崩溃，请**绝对不要**输出 JSON 格式。
请直接将完整的 Three.js 代码写在 ```javascript 和 ``` 之间。
代码简洁可运行即可。"""

        raw = self._llm_call(prompt, code_mode=True)
        extracted_code = self._extract_js_code(raw)
        if not extracted_code:
            print("\n   [警告] Builder 未能生成合法的代码块，返回为空")
            extracted_code = ""
        result = {"threejs_code": extracted_code}
        print(f"   ✓ 代码已生成 ({len(extracted_code)} 字符)")
        self._update_log(iteration, state.get("director"), state.get("narrative"), result3, result)
        return {"builder": result}

    def _node_reviewer(self, state: PipelineState) -> dict:
        iteration = state["iteration"]
        result4 = state.get("builder", {})

        print("🔬 [5/6] 代码审查...")

        prompt = f"""严格审查并修复前一位技术工程师生成的 Three.js 代码，必须保证生成的 WebGL 作品完美可运行，绝不崩溃！
请务必排查以下常见致命错误：
1. [Canvas容器]: 确保 renderer 的 domElement 被正确添加到了 id 为 'canvas-container' 的 DOM 元素中。
2. [基础要素]: 确保 Scene, PerspectiveCamera, WebGLRenderer 都被正确初始化。
3. [动画循环]: 确保 animate() 中调用了 requestAnimationFrame 和 renderer.render()。
4. [变量作用域]: 确保所有必要的变量在正确的作用域内声明，不要在初始化函数内被局部隐藏。
5. [语法污染]: 检查代码中是否混入了 `<canvas>`、`<script>` 标签或 ```javascript 等 Markdown 语法！如果包含，必须彻底删除。
6. [未引入依赖崩溃]: 严查代码中是否使用了 `THREE.OrbitControls` 或任何 `Loader` (GLTF/Font/Texture)。如果发现，必须立即删除相关代码，改为纯自动播放逻辑，防止 ReferenceError！
6.1 [交互禁用]: 严查是否存在任何交互监听（`mousemove`、`pointermove`、`touchstart/touchmove`、`wheel`、`keydown` 等）。如果有，必须全部移除，确保 evolution 页面无交互依赖。
7. [着色器崩溃拦截]: 严查代码中是否使用了 `THREE.ShaderMaterial` 或任何自定义 GLSL 字符串！如果有，必须立即用 `THREE.MeshStandardMaterial` 或 `THREE.PointsMaterial` 等内置材质替换掉，将顶点动画逻辑移至 JS 循环中，彻底杜绝 WebGL 编译错误！
8. [Undefined 调用拦截]: 严查 `animate` 循环中是否对可能为 `undefined` 的属性调用了方法。例如 `material.sheen.setHSL()` 在 `sheen` 尚未初始化时会报错。必须为其增加存在性判断（例如 `if (mesh.material.sheen) {{ ... }}`），或将其替换为更安全的基础属性，彻底消除 TypeError！
9. [uniform3fv 崩溃拦截]: 严查所有对材质颜色的赋值！绝对不允许出现 `material.color = 数字` 的写法，如果有，必须将其修复为 `material.color.setHex(数字)` 或 `.set(数字)`。**特别警告**：严查 `MeshPhysicalMaterial` 中的 `sheen`，在 r128 中 `sheen` 必须是 `THREE.Color`，如果发现写了 `sheen: 数字`，必须立即将其改为 `sheen: new THREE.Color(数字)`，并将所有的 `sheenColor` 也修正为 `sheen`！否则会引发 `uniform3fv` 致命崩溃！**并删除**所有 `MeshPhysicalMaterial` 构造/字面量中的 `thickness` 字段（r128 无此属性，会报警）；可略调高 `transmission` 补偿观感。
10. [废弃 API 崩溃拦截]: 严查代码中是否使用了 `THREE.Geometry` 或调用了 `.fromGeometry()`！Three.js r128 已经删除了 `Geometry` 和 `Face3`，必须将其彻底替换为直接操作 `THREE.BufferGeometry` 并使用 `setAttribute('position', new THREE.Float32BufferAttribute(..., 3))`，否则会导致 `TypeError: ...fromGeometry is not a function` 致命崩溃！
11. [SyntaxError 与长度拦截]: 通读生成的 JS 代码。**如果代码末尾缺少闭合的括号（`}}`, `)`）或在中间被生硬截断，直接引发了 `Uncaught SyntaxError: Unexpected end of input`，你必须在修复时补充完整的逻辑循环并封口代码块**。如果是因为前人硬编码了过长的数据数组（`[...千字...]`），请直接把数组换成简短的 `for` 循环程序化生成逻辑！
12. 仔细排查遗漏的逗号、未定义的变量（如 `let`/`const`）、和拼写错误（如 `Unexpected identifier`）。确保 `animate` 闭环完整调用 `requestAnimationFrame`。
13. 性能复核：检查所有 `POINTS`/`SLICE_COUNT`/`fragmentCount`/`instanceCount`/`particleCount` 等，必须满足：粒子 <=4000、Instanced 合计 <=2500；超标则**删减数量或去掉一整类效果**后再输出。

[生成的代码]
{result4.get('threejs_code', '')}

【重要输出格式】
为了避免 JSON 解析崩溃，请**绝对不要**输出 JSON 格式。
请直接按照以下格式输出：
1. 你的审查意见（直接用文本输出）。
2. 修复后的代码，必须用 ```javascript 和 ``` 包裹。
"""

        raw = self._llm_call(prompt, code_mode=True)
        fixed_code = self._extract_js_code(raw)
        if fixed_code:
            comments = re.sub(r'```.*?```', '', raw, flags=re.DOTALL).strip()
            if not comments:
                comments = "代码审查与修复已完成。"
        else:
            print("\n   [警告] Reviewer 未返回有效的修复代码，将安全回退使用 Builder 的原始代码")
            fixed_code = result4.get('threejs_code', '')
            comments = raw.strip()

        result = {"threejs_code": fixed_code, "review_comments": comments}
        print("   ✓ 审查完毕")
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

        print("🔍 [6/6] 艺术评审...")

        prompt = f"""评审本次迭代作品（请关注**形式是否独特**、是否摆脱俗套 demo 感，而非一味堆粒子数）。

概念：{result1.get('concept', '')[:50]}
视觉：{result3.get('visual_style_summary', '')}

请输出 JSON 格式：
```json
{{
    "total_score": 7.5,
    "conclusion": "（必填）1～3 句中文定论：是否达展陈标准、形式与概念是否咬合、最突出得失；勿只罗列优点。",
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
        print(
            f"   ✓ 评分：{result.get('total_score', 'N/A')}/10 — "
            f"{(result.get('conclusion') or '')[:72]}{'…' if len((result.get('conclusion') or '')) > 72 else ''}"
        )
        self._update_log(
            iteration,
            state.get("director"), state.get("narrative"), state.get("visual"),
            state.get("builder"), state.get("reviewer"), result,
        )
        return {"critic": result}

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

    # ---- iteration log (unchanged) ----

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

        logs = []
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    pass

        idx = next((i for i, log in enumerate(logs) if log.get("iteration") == iteration), -1)
        if idx >= 0:
            logs[idx] = log_entry
        else:
            logs.append(log_entry)

        logs = logs[-1:] if logs else []
        _atomic_write_json(log_file, logs)


if __name__ == "__main__":
    orchestrator = FullOrchestrator()
    orchestrator.run()
