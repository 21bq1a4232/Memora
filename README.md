# Memora: Fine-tuning LLM Agents **without** Fine-tuning LLMs

> **First framework to run fully on consumer hardware (M4 MacBook) without requiring OpenAI APIs.** Memory-based continual learning with seamless local/cloud LLM integration.

<p align="center">
  <b>🏠 Runs Locally</b> • <b>🧠 Memory-Based Learning</b> • <b>🔄 Multi-Backend</b> • <b>📊 Research-Ready</b>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Agent-on-the-Fly/Memento/blob/main/notebooks/memento_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="#quick-start">
    <img src="https://img.shields.io/badge/setup-5%20minutes-brightgreen" alt="Quick Setup"/>
  </a>
  <a href="docs/Research.md">
    <img src="https://img.shields.io/badge/paper-ready-blue" alt="Publication Ready"/>
  </a>
</p>

---

<table>
  <tr>
    <td align="center" width="50%">
      <img src="Figure/f1_val_test.jpg" width="90%"/>
      <br/>
      <sub><b>Memora vs. Baselines on GAIA validation and test sets.</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="Figure/f1_tasks.jpg" width="90%"/>
      <br/>
      <sub><b>Ablation study of Memora across benchmarks.</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="Figure/f1_iteration.jpg" width="90%"/>
      <br/>
      <sub><b>Continual learning curves across memory designs.</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="Figure/f1_ood.jpg" width="90%"/>
      <br/>
      <sub><b>Memora’s accuracy improvement on OOD datasets.</b></sub>
    </td>
  </tr>
</table>

## ✨ What Makes Memora Special

🏠 **Runs 100% Locally** - No OpenAI API required! Uses Ollama with consumer hardware  
🧠 **Memory-Based Learning** - Agents improve from experience without model fine-tuning  
🔄 **Multi-Backend** - Seamlessly switch between local (Ollama) and cloud (OpenAI) models  
📊 **Research-Grade** - Comprehensive logging and reproducible benchmarks  

## 🚀 One-Line Demo

```bash
$ python client/agent.py --question "Design a browser built on autonomous agents"

🧠 META-PLANNER (Cycle 1): Creating 5-step plan...
⚡ EXECUTOR: Executing tasks...
✅ RESULT: [Comprehensive autonomous browser architecture with agent communication protocols]
``` 

## 💡 Memory-Based Learning Demo

**Without Memory** (Traditional Agent):
```
Q: "What's the capital of France?"
A: "Paris"
Q: "What about that place we discussed?"
A: "I don't have context about what place you're referring to."
```

**With Memora** (Memory-Augmented):
```
Q: "What's the capital of France?" 
A: "Paris" [Stored in memory: France → Paris]
Q: "What about that place we discussed?"
A: "You're referring to France. Its capital is Paris." [Retrieved from memory]
```

## 🏗️ Core Architecture

- **🧠 Memory-Augmented MDP**: No LLM fine-tuning needed - agents learn from experience
- **🔄 Planner-Executor Loop**: CBR-driven planning with tool-orchestrated execution  
- **🛠️ MCP Tool Ecosystem**: Web search, documents, code, images, video analysis
- **📈 Strong Benchmarks**: GAIA 87.88%, DeepResearcher 66.6% F1, SimpleQA 95.0%

## 📚 Examples & Use Cases

### 🌐 [Agent-Based Browser Design](examples/browser_design.md)
```python
# Real example from our demo
question = "Design a browser built on autonomous agents"
# → 5-step plan: Research → Requirements → Architecture → Tools → Prototype
# → Comprehensive design with agent communication protocols
```

### 🔬 [Research Applications](examples/research_demo.ipynb)
- **Cost-effective experimentation** with local models
- **Cross-model comparison** studies  
- **Agent behavior analysis** with detailed logs

### 💻 [Hardware Optimization](examples/hardware_guide.md)
- **M4 MacBook**: Qwen2.5-14B (near GPT-4 performance)
- **16GB RAM**: Llama3.1-8B (4-5 tokens/sec)
- **8GB RAM**: Qwen2.5-7B (lightweight option)

---

## 🚀 5-Minute Setup

### 🏠 Local Setup (Recommended - No API costs!)

```bash
# 1. Install Ollama + optimal model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:14b  # For 16GB+ RAM

# 2. Clone & run
git clone https://github.com/21bq1a4232/Memora.git
cd Memora && pip install -r requirements.txt
python client/agent.py  # Zero config - auto-detects Ollama!
```

### ☁️ Cloud Setup (OpenAI)

```bash
export OPENAI_API_KEY="your-key-here"
python client/agent.py  # Auto-detects OpenAI backend
```

### 🔬 Research Setup

```bash
# Reproducible research environment
git clone https://github.com/21bq1a4232/Memora.git
cd Memora && pip install -r requirements.txt
python -c "import random; random.seed(42)"  # Seed control
python client/agent.py --question "Your research question"
```

## 🔧 Advanced Usage

```bash
# Custom models
python client/agent.py --meta_model "llama3.1:8b" --exec_model "qwen2.5:7b"

# With specific question
python client/agent.py --question "Design a microservice architecture"

# Research mode with detailed logs
python client/agent.py --question "Compare agent architectures" > research_log.txt
```


## ⚠️ Limitations & Future Work

### Current Limitations
- **Long-horizon tasks**: Complex multi-step reasoning still challenging
- **Tool dependency**: Some advanced features require external services  
- **Memory persistence**: Case bank currently session-based

### 🔮 Future Research Directions
- [ ] **Persistent memory systems** for long-term learning
- [ ] **Multi-modal agent integration** (vision, audio, etc.)
- [ ] **Distributed agent coordination** across multiple instances
- [ ] **Advanced reasoning patterns** beyond planner-executor

See [docs/Research.md](docs/Research.md) for detailed technical analysis and benchmarks.

---

## 📚 Citation

If Memora helps your work, please cite:

```bibtex
@software{memora2025,
  title={Memora: Adaptive AI Agents with Memory},
  author={Danda, Pranav Krishna},
  year={2025},
  url={https://github.com/<your-username>/Memora}
}

@article{zhou2025agentfly,
  title={AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs},
  author={Zhou, Huichi and others},
  journal={arXiv preprint arXiv:2508.16153},
  year={2025}
}
```

For a broader overview, please check out our survey: [Github](https://github.com/ai-agents-2030/awesome-deep-research-agent)

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Tool and interpreter extensions

---

