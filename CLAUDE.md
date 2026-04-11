# CLAUDE.md - 幻棱 Phantaprism

## Project Identity

**幻棱 (Phantaprism)** — AI-driven generative art system with cyberpunk aesthetics.
Core concept: capture human faces via camera, refract them through 6 AI agents into abstract Three.js artworks.
"幻" = phantom/illusion (AI-generated abstraction), "棱" = prism edge (multi-agent refraction layers).

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / Uvicorn
- **AI**: Google Gemini API (face analysis + all creative agents)
- **Agent Orchestration**: LangGraph StateGraph + LangChain
- **Vision**: MediaPipe Tasks (WASM, browser-side face/pose/hand/iris detection)
- **Rendering**: Three.js r160 (WebGL, procedural only, no asset loading)
- **Frontend**: Vanilla HTML/JS/CSS (no framework), cyberpunk UI (neon-gold, cyan, magenta)

## Architecture

```
Camera → MediaPipe (browser) → /api/analyze → Gemini face profiling
→ LangGraph 6-agent pipeline → Three.js artwork → /evolution.html

Agent pipeline:
  Director → Narrative → Visual Designer → Builder → Reviewer/SyntaxGuard → Critic
  (Critic can loop back to Builder if score < threshold)
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server, state management, all API endpoints, SSE event stream |
| `prompts.py` | Gemini facial analysis prompt template (Chinese, "Sherlock-style" profiling) |
| `workflow/orchestrator_full.py` | 6-agent LangGraph StateGraph orchestrator (core creative pipeline) |
| `workflow/evolution_template.html` | HTML template for generated artworks |
| `index.html` | Navigation hub ("Master Control") |
| `sensor.html` | Real-time MediaPipe face capture with 3D landmark visualization |
| `dashboard.html` | Agent orchestration control panel with live SSE updates |
| `evolution.html` | Full-screen artwork display with metadata overlay |
| `js/vision_capture.js` | MediaPipe WASM integration |
| `js/api_handler.js` | API call wrapper + loading UI |

## Data Storage

- `state.json` — current system state (last face, metrics, timestamps)
- `iteration_log.json` — complete history of all agent outputs per iteration
- `iterations/metadata.json` — per-iteration metadata index
- `iterations/*.html` — self-contained Three.js artwork files (one per iteration)

## Key Constraints & Design Decisions

- **Abstract only**: no realistic/photographic content, every artwork must be purely abstract
- **No user interaction in artworks**: pure autonomous viewing (no mouse/keyboard), camera orbits automatically
- **Performance budgets**: particles <= 4000, InstancedMesh <= 1200 per instance, animation speed >= 0.01
- **Procedural only**: no external textures/models, everything generated via code
- **Three.js r160**: strict API compliance, no deprecated Geometry/encoding APIs
- **Atomic file writes**: temp file + rename to prevent JSON corruption
- **All prompts in Chinese**: the working language for all AI agent prompts

## Configuration (.env)

- `GEMINI_API_KEY` — required
- `DEFAULT_MODEL` — LLM model (default: `gemini/gemini-2.5-flash`)
- `AGENTSART_ITERATION_INTERVAL_SEC` — cooldown between iterations (default 300s)
- `AGENTSART_EVOLUTION_CHAIN` — 0=independent iterations, 1=inherit previous code/concept
- `AGENTSART_SENSOR_PERF` — sensor page performance preset (low/eco/normal)

## API Endpoints

- `POST /api/analyze` — submit face image for Gemini analysis
- `POST /api/generate` — trigger new art generation iteration
- `GET /api/iterations` — list all iteration history
- `GET /events` — SSE stream for real-time agent progress

## Development Notes

- Server startup clears previous iterations (fresh slate each run)
- Node.js used for syntax validation of generated Three.js code (falls back to bracket-matching)
- Code Fixer agent attempts up to 2 repairs on syntax errors before giving up
- Critic scores 0-10; low scores trigger Builder regeneration with feedback
- SSE events streamed to dashboard for live multi-agent progress visualization
