# 幻棱 Phantaprism

> **AI-Driven Generative Art Installation** — Facial analysis powered iterative creation, exploring the aesthetic boundary between carbon observers and silicon algorithms
>
> **AI 驱动的生成艺术装置** — 通过面部分析驱动迭代创作，探索碳基观测者与硅基算法的美学边界
>
> 幻 — AI 生成的抽象幻象 | 棱 — 多面折射，层层转化

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-4285F4.svg)](https://ai.google.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF5733.svg)](https://langchain-ai.github.io/langgraph)

</div>

---

## Overview / 项目简介

**幻棱 (Phantaprism)** is a cyberpunk-styled AI generative art installation presented as a three-screen triptych. It captures human faces through cameras, uses **Gemini AI** for deep facial analysis, then drives a multi-agent workflow to generate iterative abstract artworks — like a prism refracting human essence into algorithmic phantoms.

**幻棱** 是一件赛博朋克风格的 AI 生成艺术装置，以三屏联展形式呈现。它通过摄像头捕捉人脸，使用 **Gemini AI** 进行深度面部分析，然后驱动多智能体工作流生成迭代抽象艺术作品——如同棱镜将人类本质折射为算法幻象。

### Three Screens / 三屏联展

| Screen | Name | Role |
|--------|------|------|
| **镜** Mirror | `sensor.html` | WebGPU shader visualization of face contours in real-time / 实时 WebGPU 面部轮廓可视化 |
| **判** Judgment | `dashboard.html` | Multi-agent orchestration panel with AI profiling / 多智能体编排面板与 AI 侧写 |
| **衍** Derivation | `evolution.html` | Full-screen generated abstract artwork / 全屏生成的抽象艺术作品 |

### Core Features / 核心特性

- **Real-time Face Capture** / 实时面部捕捉 — MediaPipe face + hand tracking, OK gesture trigger / MediaPipe 面部+手部追踪，OK 手势触发
- **AI Profiling Engine** / AI 侧写引擎 — Gemini-powered "Sherlock-style" facial micro-feature deconstruction / Gemini 驱动的「福尔摩斯式」面部微特征解构
- **Multi-Agent Orchestration** / 多智能体编排 — 6-agent LangGraph pipeline (Director → Narrative → Visual → Builder → Reviewer → Critic) / 6 智能体 LangGraph 流水线
- **WebGPU Shader** / WebGPU 着色器 — Full-screen WGSL fragment shader with contour flow field and FBM domain warping / 全屏 WGSL 着色器，轮廓流场与 FBM 域扭曲
- **Iterative Evolution** / 迭代进化 — Each face analysis drives unique abstract artwork generation / 每次面部分析驱动独特的抽象艺术生成

---

## System Architecture / 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│                    幻棱 Phantaprism System                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Camera ──▶ MediaPipe (browser) ──▶ Face Preheat (Gemini)     │
│               │                         │                      │
│          OK gesture /              Background                  │
│          auto-trigger              analysis                    │
│               │                         │                      │
│               ▼                         ▼                      │
│  ┌──────────────────┐    ┌─────────────────────────────┐      │
│  │  sensor.html 镜   │    │  dashboard.html 判           │      │
│  │  WebGPU shader   │    │  SSE live updates           │      │
│  │  face contours   │    │  agent cards + profiling    │      │
│  └──────────────────┘    └─────────────────────────────┘      │
│                                    │                           │
│                          LangGraph 6-agent                     │
│                          orchestration                         │
│                                    │                           │
│                                    ▼                           │
│                          ┌─────────────────────┐              │
│                          │  evolution.html 衍    │              │
│                          │  Three.js artwork    │              │
│                          │  auto-refresh        │              │
│                          └─────────────────────┘              │
└────────────────────────────────────────────────────────────────┘
```

---

## Quick Start / 快速开始

### Requirements / 环境要求

- Python 3.11+
- Node.js 18+ (optional, for Three.js syntax validation)
- Camera device / 摄像头设备
- WebGPU-capable browser (Chrome 113+) for sensor page

### Installation / 安装

```bash
# Clone repository / 克隆仓库
git clone https://github.com/loserqing/agentsart.git
cd agentsart

# Create virtual environment / 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

### Configuration / 配置

```bash
# Copy environment variable template / 复制环境变量模板
cp .env.example .env

# Edit .env file and fill in your API keys / 编辑 .env 文件，填入你的 API 密钥
# Required: GEMINI_API_KEY
```

### Run / 运行

```bash
# Start server / 启动服务
python main.py

# Access application (default port 8080) / 访问应用（默认端口 8080）
# Hub:        http://localhost:8080
# Sensor 镜:  http://localhost:8080/sensor.html
# Dashboard 判: http://localhost:8080/dashboard.html
# Evolution 衍: http://localhost:8080/evolution.html
```

---

## Project Structure / 项目结构

```
phantaprism/
├── main.py                     # FastAPI server, state, SSE, face preheat
├── prompts.py                  # Gemini face analysis prompt (Chinese)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── index.html                  # Navigation hub / 导航中枢
├── sensor.html                 # 镜 — WebGPU shader + MediaPipe capture
├── dashboard.html              # 判 — Agent orchestration + profiling
├── evolution.html              # 衍 — Generated artwork display
├── iterations/                 # Generated artworks storage
├── assets/
│   └── mediapipe-tasks/        # MediaPipe WASM + model files
│       └── models/             # face_landmarker.task, hand_landmarker.task
├── js/
│   ├── vision_capture.js       # MediaPipe integration (face + hand)
│   ├── api_handler.js          # API call wrapper
│   └── lib/three.min.js        # Three.js r160 UMD bundle
├── workflow/
│   ├── orchestrator_full.py    # 6-agent LangGraph orchestrator
│   └── evolution_template.html # HTML template for generated artworks
└── docs/
    ├── exhibition_statement.md # Exhibition wall text
    └── artist_statement.md     # Artist statement
```

---

## Core Modules / 核心模块

### 1. Facial Analysis Engine / 面部分析引擎 (`prompts.py`)

Gemini AI "Sherlock-style" facial profiling / Gemini AI「福尔摩斯式」面部侧写:

- **Hardware Specification Audit** / 硬件规格审计 — Age, gender presentation, complexion
- **Neural Circuit Profiling** / 神经回路侧写 — Micro-expressions, temperament
- **Aesthetic Entropy** / 美学熵值 — Makeup, lighting, composition taste
- **Social Protocol** / 社会协议 — Class/role association
- **Real-time Emotion Cache** / 实时情绪缓存 — Current emotional state

### 2. Multi-Agent Orchestrator / 多智能体编排器 (`workflow/orchestrator_full.py`)

6-agent **LangGraph StateGraph** pipeline:

| Agent | Role |
|-------|------|
| **Director** 总监 | Creative direction from face analysis |
| **Narrative** 叙事 | Story/concept development (parallel with Visual) |
| **Visual Designer** 视觉 | Color palette, composition design (parallel with Narrative) |
| **Builder** 构筑 | Three.js code generation |
| **Reviewer** 审查 | Syntax validation + code fixing |
| **Critic** 评审 | Quality scoring, can trigger Builder regen |

### 3. WebGPU Sensor Shader / WebGPU 传感器着色器 (`sensor.html`)

Full-screen WGSL fragment shader:
- Contour-based flow field from MediaPipe face landmarks
- FBM domain warping with bounded cyclic time
- Face interior rendered as transparent black
- Real-time motion/emotion uniform mapping

---

## API Reference / API 接口

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Full face analysis pipeline |
| `POST` | `/analyze_face_preview` | Background face analysis → dashboard preview |
| `POST` | `/api/generate` | Trigger new art generation iteration |
| `POST` | `/clear_presence` | Clear face state when person leaves |
| `GET` | `/get_data` | State snapshot (face, agents, ETA) |
| `GET` | `/api/iterations` | List all iteration history |
| `GET` | `/events` | SSE stream for agent progress |
| `GET` | `/health` | Service health check |

---

## Acknowledgments / 致谢

- [Google Gemini AI](https://ai.google.dev) — AI analysis engine
- [MediaPipe](https://mediapipe.dev) — Vision tracking
- [LangGraph](https://langchain-ai.github.io/langgraph) — Agent orchestration
- [Three.js](https://threejs.org) — WebGL rendering
- [FastAPI](https://fastapi.tiangolo.com) — Web framework

---

<div align="center">

**幻棱 Phantaprism** — 探索碳基与硅基的美学边界

*Built by @loserqing*

</div>
