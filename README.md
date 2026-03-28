# AgentsArt 🎭

> **AI 驱动的生成艺术系统** — 通过面部分析驱动迭代创作，探索碳基观测者与硅基算法的美学边界

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-4285F4.svg)](https://ai.google.dev)

</div>

---

## 🌌 项目简介

**AgentsArt** 是一个赛博朋克风格的 AI 生成艺术系统。它通过摄像头捕捉人脸，使用 **Gemini AI** 进行深度面部分析，然后驱动多智能体工作流生成迭代艺术作品。

### 核心特性

- 🎯 **实时面部捕捉** - MediaPipe 全身姿态追踪（面部、手部、虹膜）
- 🧠 **AI 侧写引擎** - Gemini 驱动的「福尔摩斯式」面部微特征解构
- 🤖 **多智能体编排** - CrewAI 智能体协作生成艺术迭代
- 🎨 **迭代进化** - 每次分析驱动下一次艺术创作
- 📊 **可视化仪表盘** - 实时查看分析结果与生成历史

---

## 🏗️ 系统架构

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
│  │ MediaPipe   │    │  CrewAI      │    │  Iteration    │  │
│  │ Vision      │    │  Orchestrator│    │  Storage      │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+ (可选，用于前端开发)
- 摄像头设备

### 安装

```bash
# 克隆仓库
git clone https://github.com/loserqing/agentsart.git
cd agentsart

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# - GEMINI_API_KEY: Google Gemini API 密钥
```

### 运行

```bash
# 启动服务
python main.py

# 访问应用
# 主界面：http://localhost:8000
# 传感器页面：http://localhost:8000/sensor.html
# 仪表盘：http://localhost:8000/dashboard.html
# 进化历史：http://localhost:8000/evolution.html
```

---

## 📁 项目结构

```
agentsart/
├── main.py                     # FastAPI 主入口
├── prompts.py                  # AI 分析 Prompt 模板
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── index.html                  # 主界面
├── sensor.html                 # 摄像头捕捉页面
├── dashboard.html              # 分析结果仪表盘
├── evolution.html              # 迭代历史展示
├── iterations/                 # 生成的艺术作品存储
├── assets/
│   └── mediapipe-tasks/        # MediaPipe 模型文件
├── js/
│   ├── vision_capture.js       # 视觉捕捉逻辑
│   └── api_handler.js          # API 调用封装
└── workflow/
    ├── orchestrator_full.py    # 多智能体编排器
    └── evolution_template.html # 进化历史模板
```

---

## 🎭 核心模块

### 1. 面部分析引擎 (`prompts.py`)

使用 Gemini AI 进行「福尔摩斯式」面部侧写：
- **硬件规格审计** - 年龄感、性别呈现、气色粗判
- **神经回路侧写** - 微表情、气质倾向
- **美学熵值** - 妆发、光线、构图品味
- **社会协议** - 阶层/角色联想
- **实时情绪缓存** - 当下情绪状态读取

### 2. 多智能体编排器 (`workflow/orchestrator_full.py`)

基于 CrewAI 的智能体协作系统：
- **分析智能体** - 解析面部特征
- **艺术智能体** - 生成艺术概念
- **执行智能体** - 驱动渲染引擎

### 3. 视觉捕捉 (`js/vision_capture.js`)

MediaPipe 全身姿态追踪：
- 面部地标 (Face Landmark)
- 手部地标 (Hand Landmark)
- 虹膜追踪 (Iris Tracking)
- 全身姿态 (Pose Detection)

---

## 📊 API 接口

### `POST /api/analyze`

分析上传的人脸图像

**请求:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**响应:**
```json
{
  "subject_present": true,
  "summary": "AI 侧写文本...",
  "director_influence": "导演影响说明",
  "creative_keywords": ["cyberpunk", "neon", "chrome"],
  "metrics": {
    "aesthetic_entropy": 64,
    "cyborgization_pct": 38,
    "ai_survival_pct": 71
  }
}
```

### `GET /api/iterations`

获取所有迭代历史

### `POST /api/generate`

触发新的艺术生成迭代

---

## 🎨 作品展示

运行系统后，访问以下页面查看生成结果：

| 页面 | URL | 描述 |
|------|-----|------|
| 主界面 | `/` | 系统入口 |
| 传感器 | `/sensor.html` | 实时摄像头捕捉 |
| 仪表盘 | `/dashboard.html` | AI 分析结果可视化 |
| 进化史 | `/evolution.html` | 迭代作品时间线 |

---

## 🔧 开发

### 调试模式

```bash
# 启用详细日志
export DEBUG=true
python main.py
```

### 测试

```bash
# 运行测试套件
pytest tests/
```

---

## 📝 更新日志

- **2026-03-28** - 初始公开版本
  - Gemini AI 面部分析
  - CrewAI 多智能体编排
  - MediaPipe 视觉捕捉
  - FastAPI 后端服务

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [Google Gemini AI](https://ai.google.dev) - AI 分析引擎
- [MediaPipe](https://mediapipe.dev) - 视觉追踪
- [CrewAI](https://crewai.com) - 智能体编排
- [FastAPI](https://fastapi.tiangolo.com) - Web 框架

---

<div align="center">

**AgentsArt** — 探索碳基与硅基的美学边界 🌌

*Built with 🖤 by @loserqing*

</div>
