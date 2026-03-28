import gc
import os
import re
import json
import time
import hashlib
import base64
import binascii
import asyncio
import signal
import threading
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from pydantic import BaseModel
import uvicorn
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
import backoff
from contextlib import asynccontextmanager
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Import the orchestrator
from workflow.orchestrator_full import FullOrchestrator
from prompts import AVATAR_ANALYSIS_PROMPT

# 仓库布局（FastAPI 仍以项目根目录为静态资源根）:
#   workflow/       多智能体 FullOrchestrator + HTML 模板
#   js/             前端共用（api_handler、vision_capture 等）
#   iterations/     迭代产物（生成页、metadata.json）
#   assets/         MediaPipe 等资源（若存在）
#   prompts.py      头像分析 Prompt（Gemini）
#   主入口 HTML: index.html → sensor.html / dashboard.html / evolution.html

REPO_ROOT = Path(__file__).resolve().parent

# Load env
load_dotenv(REPO_ROOT / ".env")


def _default_global_state(boot_at: float | None = None) -> dict:
    """头像分析 + 编排节拍；boot_at 用于 startup 与 lifespan 重置时对齐 timestamp / server_boot_at。"""
    t = time.time() if boot_at is None else boot_at
    return {
        "human_director_brief": None,
        "last_analysis_report": None,
        "last_face_image": None,
        "last_face_upload_at": None,
        "creative_keywords": [],
        "human_metrics": {},
        "face_analysis_status": "idle",
        "face_analysis_error": None,
        "timestamp": t,
        "server_boot_at": t,
        "orchestrator_phase": "running",
        "next_iteration_earliest_at": None,
        "iteration_interval_sec": None,
    }


GLOBAL_STATE = _default_global_state()
_state_lock = threading.RLock()


def _state_snapshot() -> dict:
    """Thread-safe shallow copy for API responses."""
    with _state_lock:
        return dict(GLOBAL_STATE)


def _state_update(**kwargs):
    """Thread-safe batch update of GLOBAL_STATE fields."""
    with _state_lock:
        GLOBAL_STATE.update(kwargs)


# --------------- SSE event stream ---------------
_agent_events: list[dict] = []
_agent_events_lock = threading.Lock()
_agent_event_counter = 0


def push_agent_event(agent: str, phase: str, iteration: int, summary: str = ""):
    global _agent_event_counter
    with _agent_events_lock:
        _agent_event_counter += 1
        _agent_events.append({
            "id": _agent_event_counter,
            "agent": agent,
            "phase": phase,
            "iteration": iteration,
            "summary": summary,
            "ts": time.time(),
        })
        if len(_agent_events) > 100:
            _agent_events[:] = _agent_events[-60:]


def _on_orchestrator_event(agent: str, phase: str, iteration: int, summary: str = ""):
    push_agent_event(agent, phase, iteration, summary)


# Shutdown event for graceful termination
shutdown_event = threading.Event()


async def _sleep_interruptible(total_sec: float, step: float = 0.05) -> None:
    """可被 shutdown_event 打断的 sleep，避免 SSE 在 asyncio.sleep(0.8) 上卡满 graceful 窗口。"""
    elapsed = 0.0
    while elapsed < total_sec:
        if shutdown_event.is_set():
            return
        chunk = min(step, total_sec - elapsed)
        await asyncio.sleep(chunk)
        elapsed += chunk


class CacheControlMiddleware:
    """纯 ASGI：只改 response headers，不包装 body_iterator（避免 StreamingResponse 在 Ctrl+C 时被 BaseHTTPMiddleware 取消并报错）。"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                if (
                    path.startswith("/api")
                    or path.endswith(".html")
                    or path in ("/state.json", "/iteration_log.json", "/events")
                ):
                    headers = MutableHeaders(scope=message)
                    headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            await send(message)

        await self.app(scope, receive, send_wrapper)

# Lifespan handler for graceful shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 启动 AgentsArt API 服务...")

    # Clear previous iterations on startup
    try:
        iteration_log_path = REPO_ROOT / "iteration_log.json"
        state_path = REPO_ROOT / "state.json"
        iterations_dir = REPO_ROOT / "iterations"

        if iteration_log_path.exists():
            iteration_log_path.unlink()
            print("🗑️ 已清除之前的迭代日志")

        if state_path.exists():
            state_path.unlink()
            print("🗑️ 已清除之前的状态文件")

        # Clear previous iterations directory
        if iterations_dir.exists():
            shutil.rmtree(iterations_dir)
            iterations_dir.mkdir()
            print("🗑️ 已清除迭代文件夹，重新创建空目录")

        with _state_lock:
            GLOBAL_STATE.clear()
            GLOBAL_STATE.update(_default_global_state(time.time()))
        print(
            "🔄 已重置全局状态：人类分析、头像图与简报已清空，"
            "Dashboard「人类分析」将显示待机自我介绍；前端可凭 server_boot_at 对齐缓存。"
        )

    except Exception as e:
        print(f"⚠️ 清除文件时出错: {e}")

    # Start the orchestrator background thread
    t = threading.Thread(target=orchestrator_loop, daemon=True)
    t.start()

    yield

    # Shutdown
    print("\n🛑 正在关闭 AgentsArt API 服务...")
    shutdown_event.set()
    try:
        t.join(timeout=1.5)
        if t.is_alive():
            print("⚠️ 编排线程仍在运行（守护线程将随进程结束）。")
    except Exception:
        pass
    print("✓ 服务已关闭")

app = FastAPI(title="AgentsArt API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CacheControlMiddleware)

# --- Models ---
class ProcessFaceRequest(BaseModel):
    image: str

# --- Orchestrator Background Loop ---
def orchestrator_loop():
    print("="*60)
    print("🤖 AgentsArt 多智能体自动迭代服务启动")
    print("="*60)

    model_name = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.0-flash")
    orchestrator = FullOrchestrator(model=model_name, on_event=_on_orchestrator_event)
    print(f"📎 编排器 LLM：{model_name}")

    iteration_gap_default = float(os.getenv("AGENTSART_ITERATION_INTERVAL_SEC", "300"))

    while not shutdown_event.is_set():
        try:
            start_time = datetime.now()
            _state_update(
                orchestrator_phase="running",
                next_iteration_earliest_at=None,
                iteration_interval_sec=iteration_gap_default,
            )
            print(f"\n⏰ [{start_time.strftime('%H:%M:%S')}] 开始新一轮迭代...")

            # 从 sensor 持续上传的缓存人脸中取最新截图，执行 Gemini 分析
            # 超过 FACE_STALE_SEC 秒未更新的截图视为过期（人已离开但 sensor 未清理）
            face_stale_sec = float(os.getenv("AGENTSART_FACE_STALE_SEC", "120"))
            influence = None
            with _state_lock:
                face_img = GLOBAL_STATE.get("last_face_image")
                upload_at = GLOBAL_STATE.get("last_face_upload_at")
                if upload_at and (time.time() - upload_at) > face_stale_sec:
                    face_img = None

            if face_img:
                print("📷 发现缓存人脸截图，启动 Gemini 分析…")
                _state_update(face_analysis_status="analyzing", face_analysis_error=None)
                result = _run_face_analysis(face_img)
                if result is not None and result.get("subject_present") is False:
                    print("📷 无可分析人脸：已清空服务端截图与人类分析（与画面无人一致）")
                    _clear_human_observer_state()
                    influence = None
                elif result:
                    kw_line = ""
                    if result["creative_keywords"]:
                        kw_line = "\n【创作关键词】" + "、".join(result["creative_keywords"])
                    _state_update(
                        human_director_brief=(result["director_influence"] or "").strip() + kw_line,
                        last_analysis_report=result["summary"],
                        creative_keywords=result["creative_keywords"],
                        human_metrics=result["human_metrics"],
                        face_analysis_status="ready",
                        face_analysis_error=None,
                        timestamp=time.time(),
                    )
                    influence = GLOBAL_STATE.get("human_director_brief")
                    print(f"🔥 人类观察者简报注入迭代: {(influence or '')[:48]}...")
                else:
                    _state_update(face_analysis_status="error", face_analysis_error="Gemini 分析未返回有效结果")
                    print("⚠️ 人脸分析失败，本轮不含人类观察者简报")
            else:
                if GLOBAL_STATE.get("last_face_image"):
                    print(f"📷 缓存截图已过期（>{face_stale_sec:.0f}s 未更新），跳过分析")
                else:
                    print("📷 无缓存人脸截图，本轮不含人类简报（创意总监将自主推演）")

            # Run the iteration
            with _state_lock:
                kw_for_iteration = list(GLOBAL_STATE.get("creative_keywords") or [])
            iteration = orchestrator.run(face_influence=influence, creative_keywords=kw_for_iteration)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"✅ [{end_time.strftime('%H:%M:%S')}] 迭代 {iteration} 完成 (耗时 {duration:.1f}秒)")

            gc.collect()

            # 两轮「迭代开始」之间至少间隔 iteration_gap（默认 300s = 5 分钟，可用 AGENTSART_ITERATION_INTERVAL_SEC 调整）
            iteration_gap = float(os.getenv("AGENTSART_ITERATION_INTERVAL_SEC", "300"))
            wait_seconds = max(0.0, iteration_gap - duration)
            eta = end_time + timedelta(seconds=wait_seconds)
            _state_update(
                orchestrator_phase="waiting",
                next_iteration_earliest_at=eta.isoformat(),
                iteration_interval_sec=iteration_gap,
            )
            if wait_seconds > 0:
                print(
                    f"💤 本轮已耗时 {duration:.1f}s，最短间隔 {iteration_gap:.0f}s，"
                    f"再等待 {wait_seconds:.0f}s 后进入下一轮…"
                )
                # Check for shutdown frequently during sleep (every 0.5 seconds)
                elapsed = 0
                check_interval = 0.5
                while elapsed < wait_seconds:
                    if shutdown_event.is_set():
                        print("🛑 收到关闭信号，立即停止等待")
                        return
                    sleep_time = min(check_interval, wait_seconds - elapsed)
                    time.sleep(sleep_time)
                    elapsed += sleep_time

        except Exception as e:
            print(f"❌ 多智能体循环出错：{e}")
            _state_update(orchestrator_phase="error_backoff", next_iteration_earliest_at=None)
            if "cannot schedule new futures after interpreter shutdown" in str(e):
                print("🔄 检测到事件循环关闭，正在优雅退出...")
                break
            # 可中断退避，避免 Ctrl+C 后仍长时间阻塞无法进入 lifespan shutdown
            for _ in range(60):
                if shutdown_event.is_set():
                    print("🛑 收到关闭信号，退出错误退避")
                    return
                time.sleep(0.5)

    print("🛑 多智能体循环已停止")


def _compact_human_analysis_report(text: str) -> str:
    """去掉空行与行首尾空白，段间仅保留单换行，便于看板紧凑展示。"""
    if not text or not isinstance(text, str):
        return text if isinstance(text, str) else ""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _strip_response_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


_MIN_CREATIVE_KW = 2
_MAX_CREATIVE_KW = 3

# 模型未返回关键词时，用于确定性补位（与 summary/influence 哈希挂钩，避免每轮同一套）
_KEYWORD_FALLBACK_POOL = (
    "湿件余像",
    "数据回眸",
    "边界颤动",
    "霓虹静默",
    "义体潮汐",
    "幽灵协议",
    "算法褶皱",
    "瞳孔残差",
    "面部熵增",
    "取景残差",
    "光线曲率",
    "肤质噪点",
    "情绪缓存",
    "观测余温",
    "屏缘裂隙",
)


def _normalize_creative_keywords(raw) -> list:
    """解析模型输出为至多 3 条短语（不含补全）。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,，、;；]", raw)
        return [p.strip() for p in parts if p.strip()][: _MAX_CREATIVE_KW]
    if isinstance(raw, list):
        out = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            if len(out) >= _MAX_CREATIVE_KW:
                break
        return out
    return []


def _keywords_from_influence(text: str) -> list:
    """从总监影响短句中切出可用作关键词的片段。"""
    if not text or not isinstance(text, str):
        return []
    out = []
    for part in re.split(r"[，,、；;\s·]+", text.strip()):
        p = part.strip()
        if 2 <= len(p) <= 8:
            out.append(p[:6])
        if len(out) >= _MAX_CREATIVE_KW:
            break
    return out


def _ensure_creative_keywords(raw, summary: str, director_influence: str) -> list:
    """
    截图分析成功后看板必须具备创作关键词：至少 2 条、至多 3 条。
    模型若缺省、JSON 解析失败或只返 1 条，则从 influence 短语或备选池补全。
    """
    base = _normalize_creative_keywords(raw)
    seen: set[str] = set()
    merged: list[str] = []
    for k in base:
        if k not in seen:
            seen.add(k)
            merged.append(k)
        if len(merged) >= _MAX_CREATIVE_KW:
            return merged

    if len(merged) < _MIN_CREATIVE_KW:
        for k in _keywords_from_influence(director_influence):
            if k not in seen:
                seen.add(k)
                merged.append(k)
            if len(merged) >= _MAX_CREATIVE_KW:
                return merged

    if len(merged) < _MIN_CREATIVE_KW:
        h = int(
            hashlib.sha256(f"{summary or ''}\x1f{director_influence or ''}".encode("utf-8")).hexdigest()[:12],
            16,
        )
        pool = _KEYWORD_FALLBACK_POOL
        for off in range(len(pool) * 2):
            cand = pool[(h + off) % len(pool)]
            if cand not in seen:
                seen.add(cand)
                merged.append(cand)
            if len(merged) >= _MIN_CREATIVE_KW:
                break

    return merged[:_MAX_CREATIVE_KW]


def _clear_human_observer_state():
    """与 /clear_presence 一致：无人脸或模型判定无可分析主体时清空截图与人类简报。"""
    _state_update(
        human_director_brief=None,
        last_analysis_report=None,
        last_face_image=None,
        last_face_upload_at=None,
        creative_keywords=[],
        human_metrics={},
        face_analysis_status="idle",
        face_analysis_error=None,
        timestamp=time.time(),
    )


MAX_FACE_IMAGE_BYTES = 6 * 1024 * 1024
_face_genai_client = None

# 使用 google-genai（google.genai.Client），与旧版 google.generativeai.GenerativeModel 不兼容
# gemini-2.0-flash 等已对新建 API 密钥停用，默认改用当前文档推荐型号
FACE_VISION_MODEL = os.environ.get("GEMINI_FACE_MODEL", "gemini-2.5-flash")


def _get_face_genai_client():
    global _face_genai_client
    if _face_genai_client is None:
        # 未传 api_key 时与文档一致：使用环境变量 GOOGLE_API_KEY
        _face_genai_client = genai.Client()
    return _face_genai_client


def _face_vision_retry_giveup(exc: BaseException) -> bool:
    """giveup=True 表示不再重试；仅对限流与部分可恢复错误重试。"""
    if isinstance(exc, (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests)):
        return False
    if isinstance(exc, genai_errors.ServerError):
        return False
    if isinstance(exc, genai_errors.ClientError):
        return exc.code not in (408, 429)
    return True


@backoff.on_exception(
    backoff.expo,
    (
        genai_errors.ClientError,
        genai_errors.ServerError,
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
    ),
    max_tries=5,
    jitter=backoff.full_jitter,
    giveup=_face_vision_retry_giveup,
    on_backoff=lambda details: print(
        f"🚦 Gemini 面部分析请求将重试，约 {details['wait']:.1f}s 后（第 {details['tries']} 次）"
    ),
)
def generate_face_vision_with_retry(client: genai.Client, *, model: str, contents):
    """google-genai：多模态 generate_content，带退避重试。"""
    return client.models.generate_content(model=model, contents=contents)


def _generate_face_analysis_sync(image_bytes: bytes):
    """在线程池中执行；若在 async 路由里直接调用会阻塞事件循环导致全站请求卡死。"""
    return generate_face_vision_with_retry(
        _get_face_genai_client(),
        model=FACE_VISION_MODEL,
        contents=[
            genai_types.Part.from_text(text=AVATAR_ANALYSIS_PROMPT),
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
    )


def _run_face_analysis(image_b64: str) -> dict | None:
    """
    对已缓存的 base64 人脸图执行 Gemini 分析，返回解析后的字典
    （summary / director_influence / creative_keywords / human_metrics）或 None。
    在 orchestrator 线程内同步调用。
    """
    try:
        base64_data = image_b64.split(",")[1] if "," in image_b64 else image_b64
        try:
            image_bytes = base64.b64decode(base64_data, validate=True)
        except (TypeError, binascii.Error):
            image_bytes = base64.b64decode(base64_data)
        if len(image_bytes) > MAX_FACE_IMAGE_BYTES:
            print("❌ 缓存人脸图片过大，跳过分析")
            return None

        print(f"🤖 调用 Gemini ({FACE_VISION_MODEL}) 进行人脸分析…")
        response = _generate_face_analysis_sync(image_bytes)
        analysis_text = _strip_response_json_fence(response.text or "")
        raw_kw = None

        try:
            analysis_json = json.loads(analysis_text)
            sp = analysis_json.get("subject_present", True)
            if isinstance(sp, str):
                sp = sp.strip().lower() in ("1", "true", "yes", "是")
            if sp is False:
                print("📷 模型判定 subject_present=false（无可分析人脸），不写入人类简报")
                return {"subject_present": False}

            raw_summary = analysis_json.get("summary", analysis_text)
            if raw_summary is not None and not isinstance(raw_summary, str):
                raw_summary = str(raw_summary)
            summary = (raw_summary or analysis_text or "").strip()
            director_influence = analysis_json.get("director_influence", (summary or "")[:50])
            if director_influence is not None and not isinstance(director_influence, str):
                director_influence = str(director_influence)
            raw_kw = analysis_json.get("creative_keywords")
            human_metrics = analysis_json.get("metrics") or {}
            if not isinstance(human_metrics, dict):
                human_metrics = {}
        except json.JSONDecodeError:
            summary = (analysis_text or "").strip()
            director_influence = summary[:50] if summary else ""
            human_metrics = {}

        if not summary:
            summary = "（模型返回为空或无法解析为 JSON，原始片段：{}）".format(
                (analysis_text[:120] + "…") if len(analysis_text) > 120 else analysis_text
            )

        summary = _compact_human_analysis_report(summary)
        director_influence = (director_influence or "").strip() or (summary or "")[:50]
        creative_keywords = _ensure_creative_keywords(raw_kw, summary, director_influence)

        print(f"✅ 人脸分析完成（约 800 字叙述 + {len(creative_keywords)} 条创作关键词）")
        return {
            "subject_present": True,
            "summary": summary,
            "director_influence": director_influence,
            "creative_keywords": creative_keywords,
            "human_metrics": human_metrics,
        }
    except Exception as e:
        print(f"❌ 人脸分析异常: {e}")
        return None


# --- API Routes ---
@app.post("/upload_face", tags=["Core"])
async def upload_face(req: ProcessFaceRequest):
    """sensor 持续上传的人脸截图（轻量存储，不触发分析；分析在每轮迭代开头执行）。"""
    image_b64 = req.image
    if not image_b64:
        raise HTTPException(status_code=400, detail="No image provided")
    if len(image_b64) > MAX_FACE_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image payload too large")
    _state_update(last_face_image=image_b64, last_face_upload_at=time.time())
    return JSONResponse({"status": "ok"})

@app.post("/clear_presence", tags=["Core"])
async def clear_presence():
    """画面内持续无人时由 sensor 调用：清空服务端截图与人类分析展示，看板回到待机自我介绍。"""
    _clear_human_observer_state()
    print("🧹 已清空人类分析展示状态（画面内无人）")
    return JSONResponse({"status": "ok", "face_analysis_status": "idle"})


def build_sensor_client_hints() -> dict:
    """展陈/笔记本降载：AGENTSART_SENSOR_PERF=low 或 TARGET_FPS / MAX_DPR。"""
    perf = os.getenv("AGENTSART_SENSOR_PERF", "normal").strip().lower()
    hints: dict = {"perf_mode": perf}
    if perf in ("low", "eco", "eco1"):
        hints["target_fps"] = 24
        hints["max_dpr"] = 1.25
        return hints
    try:
        tf = int(os.getenv("AGENTSART_SENSOR_TARGET_FPS", "60"))
    except ValueError:
        tf = 60
    hints["target_fps"] = max(15, min(60, tf))
    try:
        md = float(os.getenv("AGENTSART_SENSOR_MAX_DPR", "2"))
    except ValueError:
        md = 2.0
    hints["max_dpr"] = max(1.0, min(3.0, md))
    return hints


@app.get("/events", tags=["SSE"])
async def agent_events(request: Request):
    """SSE: real-time agent phase push (start / done)."""
    last_event_id = request.headers.get("last-event-id")
    last_id = int(last_event_id) if last_event_id else 0

    async def generate():
        nonlocal last_id
        try:
            while not shutdown_event.is_set():
                with _agent_events_lock:
                    new = [e for e in _agent_events if e["id"] > last_id]
                for evt in new:
                    last_id = evt["id"]
                    yield f"id: {evt['id']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
                await _sleep_interruptible(0.8)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/get_data", tags=["State"])
async def get_data():
    payload = _state_snapshot()
    has_face = payload.pop("last_face_image", None) is not None
    payload.pop("last_face_upload_at", None)
    payload["has_face_image"] = has_face
    payload["sensor_client_hints"] = build_sensor_client_hints()
    return JSONResponse(content=payload)


@app.get("/state.json", include_in_schema=False)
async def get_state_json():
    """显式路由 + no-store，避免静态托管时 state.json 被浏览器强缓存导致 evolution 不刷新。"""
    path = REPO_ROOT / "state.json"
    nocache = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
    if not path.is_file():
        return JSONResponse(content={}, headers=nocache)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return JSONResponse(content=data, headers=nocache)


@app.get("/health", tags=["Ops"])
async def health():
    """展陈环境健康检查：返回服务运行状态、当前迭代轮次与编排器阶段。"""
    snap = _state_snapshot()
    state_path = REPO_ROOT / "state.json"
    iteration = None
    if state_path.is_file():
        try:
            iteration = json.loads(state_path.read_text("utf-8")).get("iteration")
        except Exception:
            pass
    return JSONResponse({
        "status": "ok",
        "uptime_sec": round(time.time() - snap.get("server_boot_at", time.time()), 1),
        "orchestrator_phase": snap.get("orchestrator_phase"),
        "current_iteration": iteration,
        "face_analysis_status": snap.get("face_analysis_status"),
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

# Mount static files（相对仓库根，避免工作目录不在项目内时静态资源找不到）
app.mount("/", StaticFiles(directory=str(REPO_ROOT), html=False), name="static")
    
if __name__ == "__main__":
    import argparse
    import webbrowser
    parser = argparse.ArgumentParser(description="AgentsArt FastAPI Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()
    
    # Hide some logging noise
    import logging
    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return all(x not in msg for x in ["/get_data", "/state.json", "/iteration_log.json", "/upload_face", "/favicon.ico", "/evolution.html", "/sensor.html", "/dashboard.html", "/index.html", "/events"])
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

    def _exc_graph_has_shutdown_noise(exc: BaseException | None) -> bool:
        """遍历 __cause__ / __context__ 全链；force_exit 时多为 CancelledError + KeyboardInterrupt 嵌套。"""

        def walk(e: BaseException | None, seen: set[int]) -> bool:
            if e is None or id(e) in seen:
                return False
            seen.add(id(e))
            if isinstance(e, asyncio.CancelledError | KeyboardInterrupt):
                return True
            return walk(e.__cause__, seen) or walk(e.__context__, seen)

        return walk(exc, set())

    class _ShutdownNoiseLoggingFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "timeout graceful shutdown exceeded" in msg:
                return False
            if record.exc_info and record.exc_info[1] is not None:
                if _exc_graph_has_shutdown_noise(record.exc_info[1]):
                    return False
            # 兜底：部分路径把整段 Traceback 写进 message
            if record.levelno >= logging.ERROR and "Traceback" in msg:
                if "asyncio.exceptions.CancelledError" in msg or "CancelledError" in msg:
                    return False
                if "KeyboardInterrupt" in msg:
                    return False
            return True

    logging.getLogger("uvicorn.error").addFilter(_ShutdownNoiseLoggingFilter())

    # 启动时自动打开浏览器（守护线程，避免拖到进程无法退出）
    if not args.no_browser:
        _open_browser_timer = threading.Timer(
            1.5,
            lambda: webbrowser.open(f"http://localhost:{args.port}/index.html"),
        )
        _open_browser_timer.daemon = True
        _open_browser_timer.start()

    print("="*50)
    print("🚀 AgentsArt 统一服务启动中...")
    print(f"👉 访问地址: http://localhost:{args.port}")
    print("⌨️  Ctrl+C：立即退出进程（编排线程为守护线程，随进程结束）。")
    print("="*50)

    # 使用自定义日志配置以加快响应
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
        },
    }

    class AgentsArtServer(uvicorn.Server):
        """SIGINT：立刻 force_exit（等同连按两次 Ctrl+C），避免等满 graceful 窗口；SIGTERM 仍走较短优雅关闭。"""

        def handle_exit(self, sig, frame=None):
            shutdown_event.set()
            super().handle_exit(sig, frame)
            if sig == signal.SIGINT:
                self.force_exit = True
                print("\n🛑 收到 Ctrl+C，正在退出…", flush=True)

    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
        log_config=log_config,
        timeout_keep_alive=2,
        timeout_graceful_shutdown=2,
    )
    import sys

    try:
        AgentsArtServer(config).run()
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        shutdown_event.set()
    sys.exit(0)
