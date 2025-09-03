from __future__ import annotations
import asyncio
import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
import json
import tiktoken

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Constants & templates (same as original)
# ---------------------------------------------------------------------------
META_SYSTEM_PROMPT = (
    "You are the META‑PLANNER in a hierarchical AI system. A user will ask a\n"
    "high‑level question. **First**: try to break the problem into a *minimal sequence*\n"
    "of executable tasks. Reply in JSON with the schema:\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} … ] }\n\n"
    "If you cannot create a plan or if the question is simple enough to answer directly, provide the answer with:\n"
    "FINAL ANSWER: <answer>\n\n"
    "Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
    "Please ensure that the final answer strictly follows the question requirements, without any additional analysis.\n"
    "If the final answer is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."
)

EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task. If tools are available,\n"
    "you can use them via function calling. Otherwise, provide a direct answer.\n"
    "Always think step by step but reply with the minimal content needed for the meta-planner.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

MAX_CTX = 175000
EXE_MODEL = "o3"

# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIBackend(ChatBackend):
    def __init__(self, model: str):
        self.model = model
        self.is_ollama = self._detect_ollama()
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    
    def _detect_ollama(self) -> bool:
        """Detect if we're using Ollama based on base URL"""
        base_url = os.getenv("OPENAI_BASE_URL", "")
        return "localhost" in base_url or "127.0.0.1" in base_url or ":11434" in base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

# ---------------------------------------------------------------------------
#   Hierarchical client (trimmed: only essentials kept)
# ---------------------------------------------------------------------------
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\\s\\S]*}", text)
    return m.group(0) if m else text

def _count_tokens(msg: Dict[str, str], enc) -> int:
    role_tokens = 4
    content = msg.get("content") or ""
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
    """Return a tokenizer; fall back to cl100k_base if model is unknown."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
    
def trim_messages(messages: List[Dict[str, str]], max_tokens: int, model="gpt-3.5-turbo"):

    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    if total <= max_tokens:
        return messages
    system_msg = messages[0]
    kept: List[Dict[str, str]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)
        total += t
    return kept

class HierarchicalClient:
    MAX_CYCLES = 3

    def __init__(self, meta_model: str, exec_model: str):
        self.meta_llm = OpenAIBackend(meta_model)
        self.exec_llm = OpenAIBackend(exec_model)
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, str]] = []

    # ---------- Tool management ----------
    async def connect_to_servers(self, scripts: List[str]):
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        for script in scripts:
            path = Path(script)
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
        print("Connected tools:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result

    # ---------- Helper methods for response processing ----------
    def _is_direct_answer(self, content: str) -> bool:
        """Check if the content appears to be a direct answer rather than a plan"""
        content_lower = content.lower()
        
        # Check for greeting patterns
        greeting_patterns = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(pattern in content_lower for pattern in greeting_patterns):
            return True
        
        # Check for direct answer patterns
        answer_patterns = ["answer", "result", "solution", "is", "equals", "=", "the answer is", 
                          "i can", "i am", "as an ai", "assist", "help", "provide"]
        if any(pattern in content_lower for pattern in answer_patterns):
            return True
        
        # Check if it's a short response (likely direct answer)
        if len(content.strip()) < 200 and not content.strip().startswith("{"):
            return True
        
        return False
    
    def _clean_response(self, content: str) -> str:
        """Clean up response content"""
        cleaned = content.strip()
        
        # Remove duplicate "FINAL ANSWER:" prefixes
        while cleaned.startswith("FINAL ANSWER:"):
            cleaned = cleaned[len("FINAL ANSWER:"):].strip()
        
        # Remove task-related prefixes that might appear
        task_prefixes = ["Task 1 result:", "Task result:", "Result:"]
        for prefix in task_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned

    # ---------- Main processing ----------
    async def process_query(self, query: str, file: str, task_id: str = "interactive") -> str:
        tools_schema = await self._tools_schema()
        self.shared_history = []
        self.shared_history.append({"role": "user", "content": f"{query}\ntask_id: {task_id}\nfile_path: {file}\n"})
        planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history

        for cycle in range(self.MAX_CYCLES):
            print(f"\n🧠 META-PLANNER (Cycle {cycle + 1}):")
            print("=" * 50)
            
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            print(f"📝 Planner Response: {meta_content}")
            self.shared_history.append({"role": "assistant", "content": meta_content})

            if meta_content.startswith("FINAL ANSWER:"):
                final_answer = meta_content[len("FINAL ANSWER:"):].strip()
                print(f"\n✅ FINAL ANSWER REACHED: {final_answer}")
                return final_answer

            # Try to parse JSON plan, but fallback to direct answer if it fails
            try:
                tasks = json.loads(_strip_fences(meta_content))["plan"]
                print(f"\n📋 PLAN CREATED: {len(tasks)} tasks")
                for i, task in enumerate(tasks, 1):
                    print(f"   Task {i}: {task['description']}")
            except Exception as e:
                print(f"⚠️  JSON parsing failed: {e}")
                # Dynamic handling based on backend type
                if hasattr(self.meta_llm, 'is_ollama') and self.meta_llm.is_ollama:
                    # For Ollama models, be more flexible with response parsing
                    if self._is_direct_answer(meta_content):
                        cleaned_response = self._clean_response(meta_content)
                        print(f"✅ DIRECT ANSWER DETECTED: {cleaned_response}")
                        return cleaned_response
                    else:
                        # If no clear answer, create a simple task for the executor
                        tasks = [{"id": 1, "description": query}]
                        print(f"🔄 FALLBACK: Creating single task for executor")
                else:
                    # For OpenAI models, be more strict
                    if "FINAL ANSWER:" in meta_content:
                        answer = meta_content[meta_content.find("FINAL ANSWER:") + len("FINAL ANSWER:"):].strip()
                        print(f"✅ FINAL ANSWER FOUND: {answer}")
                        return answer
                    else:
                        error_msg = f"[planner error] {e}: {meta_content}"
                        print(f"❌ PLANNER ERROR: {error_msg}")
                        return error_msg

            print(f"\n⚡ EXECUTOR: Starting task execution")
            print("=" * 50)
            
            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                print(f"\n🎯 EXECUTING TASK {task['id']}: {task['description']}")
                
                # For Ollama models without tools, execute tasks directly
                if hasattr(self.exec_llm, 'is_ollama') and self.exec_llm.is_ollama and not self.sessions:
                    # Direct execution without tools
                    print("   🔧 Mode: Direct execution (no tools)")
                    exec_msgs = [
                        {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question directly and comprehensively."},
                        {"role": "user", "content": task['description']}
                    ]
                    exec_reply = await self.exec_llm.chat(exec_msgs)
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        print(f"   ✅ Task {task['id']} Result: {result_text[:100]}{'...' if len(result_text) > 100 else ''}")
                        self.shared_history.append({"role": "assistant", "content": f"Task {task['id']} result: {result_text}"})
                else:
                    # Original tool-based execution
                    print("   🔧 Mode: Tool-based execution")
                    exec_msgs = (
                        [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] +
                        self.shared_history +
                        [{"role": "user", "content": task_desc}]
                    )
                    while True:
                        exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
                        exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                        if exec_reply["content"]:
                            result_text = str(exec_reply["content"])
                            print(f"   ✅ Task {task['id']} Result: {result_text[:100]}{'...' if len(result_text) > 100 else ''}")
                            self.shared_history.append({"role": "assistant", "content": f"Task {task['id']} result: {result_text}"})
                            break
                        
                        # Check if we have tool calls and tools available
                        tool_calls = exec_reply.get("tool_calls") or []
                        if tool_calls and self.sessions:
                            for call in tool_calls:
                                t_name = call["function"]["name"]
                                if t_name in self.sessions:
                                    t_args = json.loads(call["function"].get("arguments") or "{}")
                                    session = self.sessions[t_name]
                                    result_msg = await session.call_tool(t_name, t_args)
                                    result_text = str(result_msg.content)
                                    exec_msgs.extend([
                                        {"role": "assistant", "content": None, "tool_calls": [call]},
                                        {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                                    ])
                                else:
                                    # Tool not available, treat as content
                                    result_text = f"Tool '{t_name}' not available. Continuing without it."
                                    exec_msgs.append({"role": "assistant", "content": result_text})
                                    break
                        else:
                            # No tool calls or no tools available, break the loop
                            break
            # After executing all tasks, check if we should return results or continue planning
            print(f"\n🔄 CYCLE {cycle + 1} COMPLETE: Checking results...")
            if self.shared_history:
                # Extract the final results from executed tasks
                task_results = []
                for msg in self.shared_history:
                    if msg["role"] == "assistant" and "result:" in msg["content"]:
                        result = msg["content"].split("result:", 1)[1].strip()
                        task_results.append(result)
                
                if task_results:
                    print(f"📊 COMBINING {len(task_results)} task results into final answer")
                    # Combine all task results into a final answer
                    if len(task_results) == 1:
                        final_result = task_results[0]
                        print(f"✅ SINGLE RESULT: {final_result[:100]}{'...' if len(final_result) > 100 else ''}")
                        return final_result
                    else:
                        combined_result = "\n\n".join(task_results)
                        print(f"✅ COMBINED RESULTS: {combined_result[:100]}{'...' if len(combined_result) > 100 else ''}")
                        return combined_result
            
            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        
        # If we've exhausted cycles, return the best available result
        if self.shared_history:
            task_results = []
            for msg in self.shared_history:
                if msg["role"] == "assistant" and "result:" in msg["content"]:
                    result = msg["content"].split("result:", 1)[1].strip()
                    task_results.append(result)
            
            if task_results:
                if len(task_results) == 1:
                    return task_results[0]
                else:
                    return "\n\n".join(task_results)
        
        return meta_content.strip()

    async def cleanup(self):
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   Command‑line & main routine
# ---------------------------------------------------------------------------

def get_default_model():
    """Get the appropriate default model based on the backend"""
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "localhost" in base_url or "127.0.0.1" in base_url or ":11434" in base_url:
        # Ollama - optimized for M4 MacBook Pro 16GB
        return "qwen2.5:14b"  # Best quality model that fits 16GB RAM
    else:
        # OpenAI
        return "gpt-4o"  # Best performance for research

def parse_args():
    default_model = get_default_model()
    parser = argparse.ArgumentParser(description="AgentFly – interactive version")
    parser.add_argument("-q", "--question", type=str, help="Your question")
    parser.add_argument("-f", "--file", type=str, default="", help="Optional file path")
    parser.add_argument("-m", "--meta_model", type=str, default=default_model, help="Meta‑planner model")
    parser.add_argument("-e", "--exec_model", type=str, default=default_model, help="Executor model")
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[], help="Paths of tool server scripts")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\nFINAL ANSWER:", answer)

async def main_async(args):
    load_dotenv()
    client = HierarchicalClient(args.meta_model, args.exec_model)
    
    # Only connect to servers if they are provided
    if args.servers:
        try:
            await client.connect_to_servers(args.servers)
        except Exception as e:
            print(f"Warning: Could not connect to some tools: {e}")
            print("Continuing without tools...")
    else:
        print("No tools specified. Running in basic mode.")

    try:
        if args.question:
            await run_single_query(client, args.question, args.file)
        else:
            print("Enter 'exit' to quit.")
            while True:
                q = input("\nQuestion: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("File path (optional): ").strip()
                await run_single_query(client, q, f)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))
