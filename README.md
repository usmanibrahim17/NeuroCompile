# 🧠 NeuroCompile  
### *Python → C++ Translator & Performance Analyzer*


> Translate Python code into optimized C++, compile & run both versions, and compare performance,AI explain tab — all inside a beautiful Gradio app.  
> NeuroCompile bridges **AI reasoning**, **compiler engineering**, and **systems optimization** in one project.

---

## 🚀 Features

- ⚙️ **Python → C++ Translation** using OpenAI GPT or Anthropic Claude  
- 🧩 **Real-time compilation & execution** (requires `g++`)  
- 📊 **Performance comparison** with execution time tracking  
- 💬 **AI Explain tab** — chat with an LLM to understand optimizations  
- 🌐 **Multi-tab Gradio UI** for clean user interaction  
- 🧠 Includes example programs (Pi, Fibonacci, Matrix Multiplication)  
- 🔐 API key validation & error-safe runtime environment  

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.10+ |
| **Frontend** | Gradio |
| **LLM Backends** | OpenAI API / Anthropic API |
| **Compiler** | g++ |
| **Core Libs** | `os`, `subprocess`, `shutil`, `time`, `json`, `requests`, `io` |

---
🧭 Tabs Overview:

Settings → Select provider (OpenAI/Anthropic) and paste your API key

Translate → Paste Python code → Get optimized C++

Run & Compare → Execute both versions, view outputs + timing

Explain → Ask “why is this faster?” and get LLM-powered insights

💡
🌟 Project Highlights

Combines AI, systems engineering, and compiler logic

Clean LLM integration with user-controlled API keys

Displays execution metrics & model transparency

Practical demo of AI-assisted code generation and analysis
🔒 Notes

The app executes real code — use only trusted input.

API keys are not stored or logged.

Compilation requires a working C++ compiler (g++ recommended).

Internet required for API access.

🧑‍💻 Author

Usman
AI & Computer Science student passionate about LLM engineering, compiler systems, and performance optimization.
Built NeuroCompile as a portfolio and research showcase project — integrating AI reasoning with low-level performance insight.

