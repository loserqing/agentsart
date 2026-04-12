# CLAUDE.md - 幻棱 Phantaprism

## Project Identity

**幻棱 (Phantaprism)** — AI-driven generative art installation with cyberpunk aesthetics.
Core concept: capture human faces via camera, refract them through 6 AI agents into abstract Three.js artworks.
"幻" = phantom/illusion (AI-generated abstraction), "棱" = prism edge (multi-agent refraction layers).

Exhibition form: 3-screen triptych (镜 → 判 → 衍).

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / Uvicorn
- **AI**: Google Gemini API (face analysis via gemini-2.5-pro, creative agents via gemini-2.5-flash)
- **Agent Orchestration**: LangGraph StateGraph + LangChain
- **Vision**: MediaPipe Tasks (WASM, browser-side face/hand/iris detection + OK gesture trigger)
- **Sensor Rendering**: WebGPU / WGSL full-screen fragment shader (contour flow field, FBM domain warping)
- **Artwork Rendering**: Three.js r160 (WebGL, procedural only, no asset loading)
- **Frontend**: Vanilla HTML/JS/CSS (no framework), cyberpunk UI (neon-gold, cyan, magenta)

## Architecture

```
Camera → MediaPipe (browser) → face preheat (background Gemini call)
       → OK gesture or auto-trigger → /analyze_face_preview → dashboard preview
       → /api/generate (reuses cached analysis) → LangGraph 6-agent pipeline
       → Three.js artwork → evolution.html (auto-refresh via state.json polling)

Agent pipeline:
  Director → Narrative + Visual Designer (parallel) → Builder → Reviewer/SyntaxGuard → Critic
  (Critic can loop back to Builder if score < threshold)

Three screens:
  sensor.html (镜) — WebGPU face visualization + MediaPipe capture
  dashboard.html (判) — agent orchestration + face profiling display
  evolution.html (衍) — full-screen generated artwork
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server, state management, all API endpoints, SSE event stream, face preheat |
| `prompts.py` | Gemini facial analysis prompt template (Chinese, "Sherlock-style" profiling) |
| `workflow/orchestrator_full.py` | 6-agent LangGraph StateGraph orchestrator (core creative pipeline) |
| `workflow/evolution_template.html` | HTML template for generated artworks (with idle state + connection lost) |
| `index.html` | Navigation hub ("幻棱 PHANTAPRISM") |
| `sensor.html` | WebGPU shader + MediaPipe face/hand capture, OK gesture detection |
| `dashboard.html` | Agent orchestration panel, SSE live updates with auto-reconnect, face profiling |
| `evolution.html` | Full-screen artwork display, auto-refresh, idle fallback scene |
| `js/vision_capture.js` | MediaPipe WASM integration (FaceLandmarker + HandLandmarker) |
| `js/api_handler.js` | API call wrapper + loading UI |

## Data Storage

- `state.json` — current iteration state (written by orchestrator, read by evolution.html for auto-refresh)
- `iteration_log.json` — complete history of all agent outputs per iteration
- `iterations/metadata.json` — per-iteration metadata index
- `iterations/*.html` — self-contained Three.js artwork files (one per iteration)
- GLOBAL_STATE (in-memory) — face analysis results, orchestrator phase, SSE events

## Key Constraints & Design Decisions

- **Abstract only**: no realistic/photographic content, every artwork must be purely abstract
- **No user interaction in artworks**: pure autonomous viewing (no mouse/keyboard), camera orbits automatically
- **Performance budgets**: particles <= 4000, InstancedMesh <= 1200 per instance, animation speed >= 0.01
- **Procedural only**: no external textures/models, everything generated via code
- **Three.js r160**: strict API compliance, no deprecated Geometry/encoding APIs
- **Atomic file writes**: temp file + rename to prevent JSON corruption
- **All prompts in Chinese**: the working language for all AI agent prompts
- **Sensor shader time-bounded**: all FBM time offsets use cyclic/bounded functions (no linear time growth)
- **Face interior**: rendered as transparent black (96% black veil) in sensor shader

## Configuration (.env)

- `GEMINI_API_KEY` — required
- `GEMINI_FACE_MODEL` — model for face analysis (default: `gemini-2.5-flash`)
- `DEFAULT_MODEL` — LLM model for agents (default: `gemini/gemini-2.5-flash`)
- `CODE_MODEL` — optional separate model for Builder/Fixer code generation
- `AGENTSART_ITERATION_INTERVAL_SEC` — cooldown between iterations (default 300s)
- `AGENTSART_EVOLUTION_CHAIN` — 0=independent iterations, 1=inherit previous code/concept
- `AGENTSART_SENSOR_PERF` — sensor page performance preset (low/eco/normal)
- `AGENTSART_CODE_STREAM` — 0=sync invoke, 1=streaming with dashboard preview

## API Endpoints

- `POST /api/analyze` — submit face image for Gemini analysis (full pipeline)
- `POST /analyze_face_preview` — background face analysis, publishes to dashboard state
- `POST /api/generate` — trigger new art generation iteration
- `POST /clear_presence` — clear face analysis state when person leaves
- `GET /api/iterations` — list all iteration history
- `GET /get_data` — state snapshot (face status, orchestrator phase, agent states, ETA)
- `GET /events` — SSE stream for real-time agent progress (with Last-Event-Id support)
- `GET /health` — service health check (uptime, phase, iteration)

## Resilience Features

- **SSE auto-reconnect**: dashboard reconnects with exponential backoff (1-8s), status indicator in header
- **Connection lost overlay**: all 3 screens show "系统重启中 · Reconnecting…" after 3 consecutive fetch failures
- **Evolution idle state**: fallback TorusKnot scene with "幻棱 · 等待凝视" when no iteration exists
- **Dashboard state recovery**: agent card states restored from `/get_data` on page refresh
- **Countdown ETA**: prominent display of next iteration timing in dashboard collaboration section

## Development Notes

- Server startup clears previous iterations (fresh slate each run)
- Node.js used for syntax validation of generated Three.js code (falls back to bracket-matching)
- Code Fixer agent attempts up to 2 repairs on syntax errors before giving up
- Critic scores 0-10; low scores trigger Builder regeneration with feedback
- Default port: 8080
