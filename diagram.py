import base64
import sys
import json
import zlib
from IPython.display import SVG, Image
import requests

mermaid_code = '''
sequenceDiagram
    participant User
    participant GradioUI as Gradio Interface
    participant ChatBot as chat_with_bot()
    participant Callback as ThinkingCallback
    participant Agent as AgentExecutor
    participant LLM as GPT-4 (OpenAI)
    participant Calculator as Calculator Tool
    participant Hotlines as SuicideHotlines Tool

    User->>GradioUI: Types message & hits Enter/Send
    GradioUI->>ChatBot: Call chat_with_bot(message, history)

    ChatBot->>Callback: Initialize ThinkingCallback()
    ChatBot->>Agent: agent_executor.invoke(input, callbacks)

    Agent->>LLM: Process user input with prompt template
    LLM->>LLM: Analyze if tools are needed

    alt Tool Usage Required
        LLM->>Agent: Decide to use tool
        Agent->>Callback: on_tool_start() - Log tool usage

        alt Math Expression Detected
            Agent->>Calculator: simple_calculator(query)
            Calculator->>Calculator: eval(expression)
            Calculator->>Agent: Return calculation result
        else Crisis/Help Request Detected
            Agent->>Hotlines: list_suicide_hotlines()
            Hotlines->>Agent: Return hotline numbers
        end

        Agent->>Callback: on_tool_end() - Log tool result
        Agent->>LLM: Send tool results back
        LLM->>LLM: Process tool results
    end

    LLM->>Agent: Generate final response
    Agent->>ChatBot: Return {output, intermediate_steps}

    ChatBot->>Callback: Get thinking_process()
    Callback->>ChatBot: Return formatted thinking steps

    ChatBot->>ChatBot: Combine thinking_process + bot_response
    ChatBot->>GradioUI: Return ("", updated_history)

    GradioUI->>User: Display response with expandable thinking process

    Note over User,Hotlines: The thinking process shows:<br/>🤔 Step 1: Using Calculator with 2*3+90<br/>✅ Result: 96

'''

def js_btoa(data):
    return base64.b64encode(data)

def pako_deflate(data):
    compress = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    compressed_data = compress.compress(data)
    compressed_data += compress.flush()
    return compressed_data

def genPakoLink(graphMarkdown: str):
    jGraph = {"code": graphMarkdown, "mermaid": {"theme": "standard"}}
    byteStr = json.dumps(jGraph).encode('utf-8')
    deflated = pako_deflate(byteStr)
    dEncode = js_btoa(deflated)
    link_code = dEncode.decode('ascii')
    return link_code

mermaid_link = genPakoLink(mermaid_code)
print("mermaid.live link:")
print('http://mermaid.live/edit#pako:' + mermaid_link)
