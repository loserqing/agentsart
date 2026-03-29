# AgentsArt 🎭

> **AI-Driven Generative Art System** — Facial analysis powered iterative creation, exploring the aesthetic boundary between carbon observers and silicon algorithms
>
> **AI 驱动的生成艺术系统** — 通过面部分析驱动迭代创作，探索碳基观测者与硅基算法的美学边界

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-4285F4.svg)](https://ai.google.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF5733.svg)](https://langchain-ai.github.io/langgraph)

</div>

---

## 🌌 Overview / 项目简介

**AgentsArt** is a cyberpunk-styled AI generative art system. It captures human faces through cameras, uses **Gemini AI** for deep facial analysis, then drives a multi-agent workflow to generate iterative artworks.

**AgentsArt** 是一个赛博朋克风格的 AI 生成艺术系统。它通过摄像头捕捉人脸，使用 **Gemini AI** 进行深度面部分析，然后驱动多智能体工作流生成迭代艺术作品。

### Core Features / 核心特性

- 🎯 **Real-time Face Capture** / 实时面部捕捉 - MediaPipe full-body pose tracking (face, hands, iris) / MediaPipe 全身姿态追踪（面部、手部、虹膜）
- 🧠 **AI Profiling Engine** / AI 侧写引擎 - Gemini-powered "Sherlock-style" facial micro-feature deconstruction / Gemini 驱动的「福尔摩斯式」面部微特征解构
- 🤖 **Multi-Agent Orchestration** / 多智能体编排 - LangGraph StateGraph agent collaboration for iterative art generation / LangGraph StateGraph 智能体协作生成艺术迭代
- 🎨 **Iterative Evolution** / 迭代进化 - Each analysis drives the next artistic creation / 每次分析驱动下一次艺术创作
- 📊 **Visual Dashboard** / 可视化仪表盘 - Real-time analysis results and generation history / 实时查看分析结果与生成历史

---

## 🏗️ System Architecture / 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentsArt System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Frontend  │───▶│  FastAPI     │───▶│   Gemini AI   │  │
│  │  (HTML/JS)  │    │   Backend    │    │   Analysis    │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                  │                      │          │
│         ▼                  ▼                      ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ MediaPipe   │    │  LangGraph   │    │  Iteration    │  │
│  │ Vision      │    │  StateGraph  │    │  Storage      │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start / 快速开始

### Requirements / 环境要求

- Python 3.11+
- Node.js 18+ (optional, for frontend development) / 可选，用于前端开发
- Camera device / 摄像头设备

### Installation / 安装

```bash
# Clone repository / 克隆仓库
git clone https://github.com/loserqing/agentsart.git
cd agentsart

# Create virtual environment / 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

### Configuration / 配置

```bash
# Copy environment variable template / 复制环境变量模板
cp .env.example .env

# Edit .env file and fill in your API keys / 编辑 .env 文件，填入你的 API 密钥
# - GEMINI_API_KEY: Google Gemini API Key / Google Gemini API 密钥
```

### Run / 运行

```bash
# Start server / 启动服务
python main.py

# Access application / 访问应用
# Main Interface: http://localhost:8000 / 主界面
# Sensor Page: http://localhost:8000/sensor.html / 传感器页面
# Dashboard: http://localhost:8000/dashboard.html / 仪表盘
# Evolution History: http://localhost:8000/evolution.html / 进化历史
```

---

## 📁 Project Structure / 项目结构

```
agentsart/
├── main.py                     # FastAPI main entry / FastAPI 主入口
├── prompts.py                  # AI analysis prompt templates / AI 分析 Prompt 模板
├── requirements.txt            # Python dependencies / Python 依赖
├── .env.example                # Environment variable template / 环境变量模板
├── index.html                  # Main interface / 主界面
├── sensor.html                 # Camera capture page / 摄像头捕捉页面
├── dashboard.html              # Analysis result dashboard / 分析结果仪表盘
├── evolution.html              # Iteration history display / 迭代历史展示
├── iterations/                 # Generated artworks storage / 生成的艺术作品存储
├── assets/
│   └── mediapipe-tasks/        # MediaPipe model files / MediaPipe 模型文件
├── js/
│   ├── vision_capture.js       # Vision capture logic / 视觉捕捉逻辑
│   └── api_handler.js          # API call wrapper / API 调用封装
└── workflow/
    ├── orchestrator_full.py    # Multi-agent orchestrator / 多智能体编排器
    └── evolution_template.html # Evolution history template / 进化历史模板
```

---

## 🎭 Core Modules / 核心模块

### 1. Facial Analysis Engine (`prompts.py`) / 面部分析引擎

Uses Gemini AI for "Sherlock-style" facial profiling / 使用 Gemini AI 进行「福尔摩斯式」面部侧写:

- **Hardware Specification Audit** / 硬件规格审计 - Age perception, gender presentation, complexion rough judgment / 年龄感、性别呈现、气色粗判
- **Neural Circuit Profiling** / 神经回路侧写 - Micro-expressions, temperament tendency / 微表情、气质倾向
- **Aesthetic Entropy** / 美学熵值 - Makeup, lighting, composition taste / 妆发、光线、构图品味
- **Social Protocol** / 社会协议 - Rough class/role association / 阶层/角色联想
- **Real-time Emotion Cache** / 实时情绪缓存 - Current emotional state reading / 当下情绪状态读取

### 2. Multi-Agent Orchestrator (`workflow/orchestrator_full.py`) / 多智能体编排器

**LangGraph StateGraph** based agent collaboration system / 基于 LangGraph StateGraph 的智能体协作系统:

- **State Management** / 状态管理 - TypedDict defines iteration state / TypedDict 定义迭代状态
- **Node Orchestration** / 节点编排 - START → Analysis → Generation → Rendering → END / START → 分析 → 生成 → 渲染 → END
- **Conditional Branching** / 条件分支 - Dynamically adjust generation strategy based on analysis results / 根据分析结果动态调整生成策略
- **Code Generation** / 代码生成 - Output Three.js (WebGL) generative art code / 输出 Three.js (WebGL) 生成艺术代码

### 3. Vision Capture (`js/vision_capture.js`) / 视觉捕捉

MediaPipe full-body pose tracking / MediaPipe 全身姿态追踪:

- Face Landmark / 面部地标
- Hand Landmark / 手部地标
- Iris Tracking / 虹膜追踪
- Pose Detection / 全身姿态

---

## 📊 API Reference / API 接口

### `POST /api/analyze`

Analyze uploaded face image / 分析上传的人脸图像

**Request / 请求:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response / 响应:**
```json
{
  "subject_present": true,
  "summary": "AI profiling text...",
  "director_influence": "Director influence description",
  "creative_keywords": ["cyberpunk", "neon", "chrome"],
  "metrics": {
    "aesthetic_entropy": 64,
    "cyborgization_pct": 38,
    "ai_survival_pct": 45
  }
}
```

### `GET /api/iterations`

Get all iteration history / 获取所有迭代历史

### `POST /api/generate`

Trigger new art generation iteration / 触发新的艺术生成迭代

---

## 🎨 Gallery / 作品展示

After running the system, visit these pages to view generated results / 运行系统后，访问以下页面查看生成结果:

| Page / 页面 | URL | Description / 描述 |
|------|-----|------|
| Main Interface / 主界面 | `/` | System entry point / 系统入口 |
| Sensor / 传感器 | `/sensor.html` | Real-time camera capture / 实时摄像头捕捉 |
| Dashboard / 仪表盘 | `/dashboard.html` | AI analysis result visualization / AI 分析结果可视化 |
| Evolution History / 进化史 | `/evolution.html` | Iterative artwork timeline / 迭代作品时间线 |

---

## 🔧 Development / 开发

### Debug Mode / 调试模式

```bash
# Enable verbose logging / 启用详细日志
export DEBUG=true
python main.py
```

### Testing / 测试

```bash
# Run test suite / 运行测试套件
pytest tests/
```

---

## 📝 Changelog / 更新日志

- **2026-03-28** - Initial public release / 初始公开版本
  - Gemini AI facial analysis / Gemini AI 面部分析
  - LangGraph multi-agent orchestration / LangGraph 多智能体编排
  - MediaPipe vision capture / MediaPipe 视觉捕捉
  - FastAPI backend service / FastAPI 后端服务

---

## 📄 License / 许可证

MIT License - See [LICENSE](LICENSE) for details / 详见 [LICENSE](LICENSE)

---

## 🙏 Acknowledgments / 致谢

- [Google Gemini AI](https://ai.google.dev) - AI analysis engine / AI 分析引擎
- [MediaPipe](https://mediapipe.dev) - Vision tracking / 视觉追踪
- [LangGraph](https://langchain-ai.github.io/langgraph) - Agent orchestration / 智能体编排
- [LangChain](https://python.langchain.com) - LLM application framework / LLM 应用框架
- [FastAPI](https://fastapi.tiangolo.com) - Web framework / Web 框架

---

<div align="center">

**AgentsArt** — Exploring the aesthetic boundary between carbon and silicon 🌌

**AgentsArt** — 探索碳基与硅基的美学边界 🌌

*Built with 🖤 by @loserqing*

</div>
