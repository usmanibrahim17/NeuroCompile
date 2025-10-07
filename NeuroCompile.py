import gradio as gr
import openai
import anthropic
import os
import sys
import io
import time
import subprocess
import shutil
from typing import Dict, List, Tuple

# --- Constants, Prompts, and Examples ---

TRANSLATE_SYSTEM_PROMPT = """You are an expert at translating Python code to highly optimized C++. Focus on:
Using efficient data types, applying compiler optimizations, minimizing memory allocations, and using appropriate C++ standard library features.
Generate only the C++ code with necessary includes, no explanations."""

EXPLAIN_SYSTEM_PROMPT_TEMPLATE = """You are a helpful coding mentor. You have access to the original Python code and the translated C++ code. Your task is to answer user questions about the code, optimizations, performance, or C++ concepts. Be clear, friendly, and use code snippets in your explanations when helpful.

**Original Python Code:**
```python
{python_code}
```

**Translated C++ Code:**
```cpp
{cpp_code}
```
"""

DEFAULT_PYTHON_CODE = {
    "Calculate Pi": """import time

def calculate_pi(iterations):
    result = 1.0
    for i in range(1, iterations + 1):
        j = i * 4 - 1
        result -= (1.0 / j)
        j = i * 4 + 1
        result += (1.0 / j)
    return result * 4

start = time.time()
pi = calculate_pi(10_000_000)
end = time.time()

print(f"Pi: {pi:.10f}")
print(f"Time: {end - start:.6f}s")""",
    "Fibonacci": """import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

start = time.time()
result = fibonacci(35)
end = time.time()

print(f"Fibonacci(35): {result}")
print(f"Time: {end - start:.6f}s")""",
    "Matrix Multiplication": """import time

def matrix_multiply(size):
    A = [[i + j for j in range(size)] for i in range(size)]
    B = [[i - j for j in range(size)] for i in range(size)]
    C = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    return C

start = time.time()
result = matrix_multiply(200)
end = time.time()

print(f"Matrix 200x200 multiplied")
print(f"Time: {end - start:.6f}s")"""
}

# --- Core Logic ---

def update_api_settings(provider: str, key: str, settings: Dict) -> Tuple[Dict, gr.Button, gr.Button, gr.Button]:
    if not key:
        raise gr.Error("Please enter your API key in the Settings tab.")
    if provider == "OpenAI" and not key.startswith("sk-"):
        raise gr.Error("Invalid OpenAI key. It should start with 'sk-'.")
    
    settings["provider"] = provider
    settings["key"] = key
    gr.Info(f"{provider} API key updated successfully!")
    return settings, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=True)

def clean_llm_cpp_output(code: str) -> str:
    """Strips markdown and other LLM artifacts from C++ code and ensures common headers."""
    # Strip markdown blocks
    if code.strip().startswith("```cpp:disable-run
        code = code.strip()[5:]
    if code.strip().startswith("```"):
        code = code.strip()[3:]
    if code.strip().endswith("```"):
        code = code.strip()[:-3]
    
    code = code.strip()
    
    # Ensure common headers are present if their features are used
    if "std::cout" in code and "#include <iostream>" not in code:
        code = "#include <iostream>\n" + code
    if "std::chrono" in code and "#include <chrono>" not in code:
        code = "#include <chrono>\n" + code
    if "std::vector" in code and "#include <vector>" not in code:
        code = "#include <vector>\n" + code
    if "std::string" in code and "#include <string>" not in code:
        code = "#include <string>\n" + code
        
    return code

def translate_code(python_code: str, settings: Dict, progress=gr.Progress(track_tqdm=True)) -> str:
    provider, api_key = settings.get("provider"), settings.get("key")
    if not api_key:
        raise gr.Error("API key is not set. Please go to the ⚙️ Settings tab to enter it.")
    if not python_code.strip():
        raise gr.Error("Please enter some Python code to translate.")

    progress(0, desc="Connecting to API...")
    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            progress(0.5, desc="Translating with GPT-4...")
            response = client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": TRANSLATE_SYSTEM_PROMPT}, {"role": "user", "content": python_code}], temperature=0.1)
            raw_code = response.choices[0].message.content
        else: # Claude
            client = anthropic.Anthropic(api_key=api_key)
            progress(0.5, desc="Translating with Claude 3.5 Sonnet...")
            response = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=4096, system=TRANSLATE_SYSTEM_PROMPT, messages=[{"role": "user", "content": python_code}])
            raw_code = response.content[0].text
        
        return clean_llm_cpp_output(raw_code)

    except openai.AuthenticationError:
        raise gr.Error("OpenAI authentication failed. Please check your API key.")
    except anthropic.AuthenticationError:
        raise gr.Error("Claude authentication failed. Please check your API key.")
    except Exception as e:
        raise gr.Error(f"An unexpected API error occurred: {e}")

def run_python_code(code: str) -> Tuple[str, float]:
    output_capture = io.StringIO()
    original_stdout, sys.stdout = sys.stdout, output_capture
    start_time = time.perf_counter()
    try:
        compile(code, "<string>", "exec")
        exec(code, {})
    except SyntaxError as e:
        sys.stdout = original_stdout
        return f"Python Syntax Error:\n{e}", 0.0
    except Exception as e:
        sys.stdout = original_stdout
        return f"Error executing Python code:\n{e}", 0.0
    finally:
        sys.stdout = original_stdout
    end_time = time.perf_counter()
    return output_capture.getvalue(), end_time - start_time

def run_cpp_code(code: str) -> Tuple[str, float]:
    if not shutil.which("g++"):
        raise gr.Error("C++ compiler (g++) not found. Please install MinGW and add it to your system's PATH.")
    
    source_path, exe_path = "optimized.cpp", "optimized.exe"
    # Clean the code one last time before writing
    cleaned_code = clean_llm_cpp_output(code)
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(cleaned_code)

    compile_cmd = ["g++", "-Ofast", "-std=c++17", "-march=native", "-o", exe_path, source_path]
    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=15)
    
    if compile_proc.returncode != 0:
        error_message = f"Compilation failed.\n\n{compile_proc.stderr}"
        if os.path.exists(source_path):
            os.remove(source_path)
        return error_message, 0.0

    start_time = time.perf_counter()
    try:
        run_proc = subprocess.run([os.path.abspath(exe_path)], capture_output=True, text=True, timeout=10, check=True)
        output = run_proc.stdout
    except subprocess.TimeoutExpired:
        return "Execution timed out after 10 seconds.", 0.0
    except subprocess.CalledProcessError as e:
        return f"Execution failed with return code {e.returncode}:\n{e.stderr}", 0.0
    finally:
        end_time = time.perf_counter()
        if os.path.exists(source_path): os.remove(source_path)
        if os.path.exists(exe_path): os.remove(exe_path)
        
    return output, end_time - start_time

def run_and_compare(python_code: str, cpp_code: str, progress=gr.Progress(track_tqdm=True)) -> str:
    if not python_code.strip() or not cpp_code.strip():
        raise gr.Error("Both Python and C++ code must be present to run a comparison.")
    
    progress(0, desc="Running Python code...")
    py_out, py_time = run_python_code(python_code)
    
    progress(0.5, desc="Compiling & Running C++ code...")
    cpp_out, cpp_time = run_cpp_code(cpp_code)

    py_result = f"PYTHON OUTPUT:\n{py_out.strip()}\n\nExecution Time: {py_time:.6f}s"
    cpp_result = f"C++ OUTPUT:\n{cpp_out.strip()}\n\nExecution Time: {cpp_time:.6f}s"
    
    divider = "═" * 50
    speedup_msg = "Could not determine speedup due to errors or zero execution time."
    if py_time > 0 and cpp_time > 0 and "Error" not in cpp_out and "failed" not in cpp_out:
        speedup = py_time / cpp_time
        speedup_msg = f"SPEEDUP: C++ was {speedup:.2f}x faster"

    return f"{divider}\n{py_result}\n{divider}\n{cpp_result}\n{divider}\n{speedup_msg}"

def explain_code(chat_history: List[Dict[str, str]], user_question: str, settings: Dict, py_code: str, cpp_code: str, progress=gr.Progress(track_tqdm=True)):
    provider, api_key = settings.get("provider"), settings.get("key")
    if not api_key:
        raise gr.Error("API key is not set. Please configure it in the ⚙️ Settings tab.")
    if not user_question.strip():
        return chat_history, ""
    if not py_code.strip() or not cpp_code.strip():
        raise gr.Error("Please translate the code first before asking questions.")

    progress(0, desc="Thinking...")
    
    chat_history.append({"role": "user", "content": user_question})
    system_prompt = EXPLAIN_SYSTEM_PROMPT_TEMPLATE.format(python_code=py_code, cpp_code=cpp_code)

    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history
            response = client.chat.completions.create(model="gpt-4", messages=messages_for_api, temperature=0.5)
            bot_message = response.choices[0].message.content
        else: # Claude
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=4096, system=system_prompt, messages=chat_history)
            bot_message = response.content[0].text
            
        chat_history.append({"role": "assistant", "content": bot_message})
        return chat_history, ""
    except Exception as e:
        chat_history.pop() # Remove the user's question if the API call fails
        raise gr.Error(f"An API error occurred: {e}")


def load_example(example_name: str) -> str:
    return DEFAULT_PYTHON_CODE.get(example_name, "")

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Code Translator") as demo:
    app_settings = gr.State({"provider": "OpenAI", "key": None})
    python_code_state = gr.State(DEFAULT_PYTHON_CODE["Calculate Pi"])
    cpp_code_state = gr.State("")

    gr.Markdown("# 🤖 Python to C++ Code Translator & Performance Analyzer")
    gr.Markdown("A portfolio-ready tool to translate Python code to optimized C++, execute both versions, compare their performance, and explain the optimizations.")
    
    with gr.Accordion("Expand for Instructions", open=False):
        gr.Markdown("""
        ### How to Use This Tool
        1.  **⚙️ Settings:** Choose your AI provider (OpenAI or Claude) and enter your API key. The buttons on other tabs will be disabled until you save your key.
        2.  **🔄 Translate:**
            *   Load a pre-built example from the dropdown or write your own Python code. The 'Translate' button is disabled if the code box is empty.
            *   Click **Translate**. The optimized C++ code will appear on the right.
        3.  **▶️ Run & Compare:**
            *   After a successful translation, click **Run Both Versions**.
            *   The tool will execute the Python code, compile and run the C++ code, and show you the output and execution times for both.
        4.  **💬 Explain:**
            *   Ask the AI assistant questions about the code, the translation, or the performance results.
        
        **System Requirements:**
        *   Python 3.10+
        *   `g++` compiler (MinGW on Windows) installed and accessible in your system's PATH.
        """)

    with gr.Tabs():
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("## API Configuration")
            gr.Markdown("Get your API keys from [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/settings/keys). The tool securely handles your key and does not store it.")
            provider_radio = gr.Radio(["OpenAI", "Claude"], label="Select API Provider", value="OpenAI")
            api_key_textbox = gr.Textbox(label="API Key", type="password", placeholder="Enter your API key (e.g., sk-...)")
            update_button = gr.Button("Save Settings")

        with gr.Tab("🔄 Translate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Python Input")
                    example_dropdown = gr.Dropdown(label="Load Example", choices=list(DEFAULT_PYTHON_CODE.keys()), value="Calculate Pi")
                    python_input = gr.Code(value=DEFAULT_PYTHON_CODE["Calculate Pi"], label="Python", language="python", lines=20)
                with gr.Column(scale=1):
                    gr.Markdown("### Optimized C++ Output")
                    cpp_output = gr.Code(label="C++", language="cpp", lines=20, interactive=False)
            translate_button = gr.Button("Translate", variant="primary", interactive=False)
            
        with gr.Tab("▶️ Run & Compare"):
            gr.Markdown("## Performance Comparison")
            run_button = gr.Button("Run Both Versions", variant="primary", interactive=False)
            results_display = gr.Textbox(label="📊 Results", lines=22, interactive=False, placeholder="Execution results and speed comparison will appear here...")

        with gr.Tab("💬 Explain"):
            gr.Markdown("## Code Explanation Chatbot")
            chatbot = gr.Chatbot(label="Code Mentor", height=550, type="messages")
            with gr.Row():
                question_box = gr.Textbox(label="Your Question", scale=4, placeholder="e.g., Why is the C++ version faster? Explain the memory allocation.")
                ask_button = gr.Button("Ask", scale=1, variant="primary", interactive=False)
            clear_button = gr.Button("Clear Chat History")

    gr.Markdown("---")
    gr.Markdown("for code check out,https://github.com/usmanibrahim17/NeuroCompile")

    # --- Event Handlers ---
    
    update_button.click(
        update_api_settings, 
        inputs=[provider_radio, api_key_textbox, app_settings], 
        outputs=[app_settings, translate_button, run_button, ask_button], 
        queue=False
    )

    def translate_and_update_state(python_code, settings, progress=gr.Progress(track_tqdm=True)):
        translated_code = translate_code(python_code, settings, progress)
        return translated_code, python_code, translated_code

    translate_button.click(
        translate_and_update_state,
        inputs=[python_input, app_settings],
        outputs=[cpp_output, python_code_state, cpp_code_state]
    )
    
    def on_py_input_change(code):
        # Only enable the button if the API key is also set (i.e., the button is already interactive)
        return gr.Button(interactive=bool(code.strip()))

    python_input.change(
        on_py_input_change,
        inputs=[python_input],
        outputs=[translate_button],
        queue=False
    )
    
    example_dropdown.change(load_example, inputs=[example_dropdown], outputs=[python_input], queue=False)
    
    def run_and_compare_wrapper(py_code, cpp_code, progress=gr.Progress(track_tqdm=True)):
        try:
            return run_and_compare(py_code, cpp_code, progress)
        except Exception as e:
            return f"An unexpected application error occurred:\n{type(e).__name__}: {e}"
    
    run_button.click(run_and_compare_wrapper, inputs=[python_code_state, cpp_code_state], outputs=[results_display])
    
    def explain_code_wrapper(chat_history, question, settings, py_code, cpp_code, progress=gr.Progress(track_tqdm=True)):
        try:
            return explain_code(chat_history, question, settings, py_code, cpp_code, progress)
        except Exception as e:
            # Add error as a temporary bot message
            chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
            return chat_history, ""

    ask_button.click(explain_code_wrapper, inputs=[chatbot, question_box, app_settings, python_code_state, cpp_code_state], outputs=[chatbot, question_box])
    
    clear_button.click(lambda: ([], None), outputs=[chatbot, cpp_code_state], queue=False)

if __name__ == "__main__":
    demo.launch(share=False)
```
