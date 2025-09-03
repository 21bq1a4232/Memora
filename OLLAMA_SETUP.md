# Running Memento with Ollama

This guide shows you how to run the Memento project using Ollama instead of OpenAI.

## ✅ What's Already Working

- ✅ Ollama is installed and running
- ✅ You have the `deepseek-coder:6.7b-instruct` model available
- ✅ Dependencies are installed
- ✅ Environment is configured

## 🚀 Quick Start

### 1. Start Ollama (if not already running)
```bash
ollama serve
```

### 2. Load Environment Variables
```bash
source ollama_config.env
```

### 3. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 4. Run the Agent

#### Option A: Simple Question
```bash
python client/agent.py --question "What is 2+2?" --meta_model "deepseek-coder:6.7b-instruct" --exec_model "deepseek-coder:6.7b-instruct"
```

#### Option B: Interactive Mode
```bash
python client/agent.py --meta_model "deepseek-coder:6.7b-instruct" --exec_model "deepseek-coder:6.7b-instruct"
```

## ⚠️ Important Limitations

### Current Issues with Ollama
1. **No Function Calling Support**: Ollama doesn't support the `tools` parameter that Memento requires
2. **Limited Tool Integration**: The MCP tool system won't work properly
3. **JSON Parsing Issues**: The planner expects specific JSON format that Ollama may not provide

### What This Means
- The agent will run but may encounter errors when trying to use tools
- You'll get basic Q&A functionality but not the full Memento experience
- Some features like web search, document processing, etc. won't work

## 🔧 Alternative Solutions

### Option 1: Use Ollama with OpenAI-Compatible API (Current Setup)
- ✅ Pros: Easy setup, local models, no API costs
- ❌ Cons: Limited functionality, no tool support

### Option 2: Use LM Studio
- LM Studio provides better OpenAI API compatibility
- May support function calling
- Download from: https://lmstudio.ai/

### Option 3: Use OpenAI API
- Full functionality with all tools
- Requires API key and costs
- Set `OPENAI_BASE_URL=https://api.openai.com/v1` in `.env`

## 📝 Environment Configuration

Your `ollama_config.env` file contains:
```bash
export OPENAI_API_KEY=dummy_key_for_ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
```

## 🧪 Testing

To test if Ollama is working:
```bash
curl http://localhost:11434/v1/models
```

Should return:
```json
{"object":"list","data":[{"id":"deepseek-coder:6.7b-instruct","object":"model","created":1752308986,"owned_by":"library"}]}
```

## 🚨 Troubleshooting

### Ollama Not Running
```bash
ollama serve
```

### Model Not Found
```bash
ollama pull deepseek-coder:6.7b-instruct
```

### Permission Issues
```bash
chmod +x client/agent.py
```

### Virtual Environment Issues
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔮 Future Improvements

To make Memento work better with Ollama, you could:

1. **Modify the agent code** to handle missing tool support gracefully
2. **Implement a tool fallback system** for when function calling isn't available
3. **Use a different model** that better supports structured output
4. **Wait for Ollama to add function calling support**

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Memento Project](https://github.com/Agent-on-the-Fly/Memento)
- [OpenAI API Compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md)

## 🎯 Recommendation

For now, use this setup for:
- Basic Q&A tasks
- Code generation
- Simple reasoning tasks

For full Memento functionality, consider:
- Using OpenAI API
- Waiting for Ollama function calling support
- Using LM Studio as an alternative
