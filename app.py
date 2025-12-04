import os
import gradio as gr
from typing import Dict, List, Tuple, Any
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor  # Changed from langchain_core to langchain
from langchain.agents import create_openai_functions_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction 

"""Chatbot with OpenAI and LangChain using Gradio interface with custom calculator tool."""

# Set up OpenAI API key (uncomment and modify the Google Colab section if needed)
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Or set your API key directly (not recommended for production)
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"


class ThinkingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to capture thinking steps."""

    def __init__(self):
        self.thinking_steps = []
        self.current_step = 1

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        step_info = f"**Step {self.current_step}:** 🔧 Using `{tool_name}` with input: `{input_str}`"
        self.thinking_steps.append(step_info)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes running."""
        # Truncate long outputs for display
        display_output = output[:200] + "..." if len(output) > 200 else output
        result_info = f"**Result:** ✅ {display_output}"
        self.thinking_steps.append(result_info)
        self.thinking_steps.append("---")  # Separator
        self.current_step += 1

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent decides to take an action."""
        pass

    def get_thinking_process(self) -> str:
        """Get formatted thinking process."""
        if not self.thinking_steps:
            return ""

        thinking_content = "\n\n".join(self.thinking_steps)
        return f"\n\n<details>\n<summary>🤔 <strong>Thinking Process</strong> (click to expand)</summary>\n\n{thinking_content}\n\n</details>\n\n"

    def reset(self):
        """Reset the thinking steps for a new conversation."""
        self.thinking_steps = []
        self.current_step = 1


def simple_calculator(query: str) -> str:
    """
    A simple calculator that evaluates basic math expressions from a string.

    Args:
        query (str): A string representing a math expression (e.g., "2 + 2").

    Returns:
        str: The result of the calculation or an error message.
    """
    try:
        result = eval(query, {"__builtins__": {}})
        return str(result)
    except Exception as err:
        return f"Error in calculation: {err}"


def get_calculator_tool() -> Tool:
    """
    Returns a LangChain Tool instance for the calculator.

    Returns:
        Tool: The calculator tool with metadata and callback function.
    """
    return Tool(
        name="Calculator",
        description="Evaluates basic math expressions (e.g., '3 * (4 + 5)').",
        func=simple_calculator,
    )


def list_suicide_hotlines(_: str) -> str:
    """
    Returns a list of suicide prevention hotline numbers.

    Args:
        _ (str): Placeholder input, not used.

    Returns:
        str: Formatted hotline numbers as a string.
    """
    return (
        "📞 Suicide Prevention Hotlines:\n"
        "- US National Suicide Prevention Lifeline: 1-800-273-TALK (8255)\n"
        "- Crisis Text Line: Text HOME to 741741 (US & Canada)\n"
        "- SAMHSA's Helpline: 1-800-662-HELP (4357)\n"
        "- TrevorLifeline for LGBTQ+: 1-866-488-7386\n"
        "- International Directory: https://www.opencounseling.com/suicide-hotlines"
    )


def get_hotlines_tool() -> Tool:
    """
    Returns a LangChain Tool instance for listing suicide hotlines.

    Returns:
        Tool: The hotline tool with description and callback.
    """
    return Tool(
        name="SuicideHotlines",
        description="Provides suicide prevention hotline numbers and resources.",
        func=list_suicide_hotlines,
    )


def get_llm() -> ChatOpenAI:
    """
    Initializes and returns the OpenAI language model using the environment variable.

    Returns:
        ChatOpenAI: The initialized language model for use in the agent.
    """
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return ChatOpenAI(model="gpt-4", temperature=0.0)


def get_prompt() -> ChatPromptTemplate:
    """
    Returns a manually constructed ChatPromptTemplate with required variables.

    Returns:
        ChatPromptTemplate: The prompt template including system and user messages.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def get_agent_executor() -> AgentExecutor:
    """
    Sets up and returns the LangChain AgentExecutor with OpenAI and custom tools.

    Returns:
        AgentExecutor: The configured agent executor.
    """
    llm: ChatOpenAI = get_llm()
    tools: List[Tool] = [
        get_calculator_tool(),
        get_hotlines_tool()
    ]
    prompt: ChatPromptTemplate = get_prompt()
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True  # This ensures we get intermediate steps
    )


# Initialize the agent executor globally
try:
    agent_executor = get_agent_executor()
    agent_initialized = True
except Exception as e:
    print(f"Failed to initialize agent: {e}")
    agent_initialized = False


def chat_with_bot(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process user message and return bot response with updated history.

    Args:
        message (str): User's input message
        history (List[Tuple[str, str]]): Chat history as list of (user, bot) tuples

    Returns:
        Tuple[str, List[Tuple[str, str]]]: Empty string and updated history
    """
    if not agent_initialized:
        bot_response = "❌ Agent not initialized. Please check your OpenAI API key."
        history.append((message, bot_response))
        return "", history

    if not message.strip():
        return "", history

    try:
        # Create a custom callback to capture thinking steps
        thinking_callback = ThinkingCallbackHandler()

        # Get response from LangChain agent with callback
        response = agent_executor.invoke(
            {"input": message},
            config={"callbacks": [thinking_callback]}
        )

        bot_response = response.get("output", "No response generated.")

        # Get thinking process from callback
        thinking_process = thinking_callback.get_thinking_process()

        # If no thinking process was captured through callback, try intermediate_steps
        if not thinking_process:
            intermediate_steps = response.get("intermediate_steps", [])
            if intermediate_steps:
                thinking_process = "\n\n<details>\n<summary>🤔 <strong>Thinking Process</strong> (click to expand)</summary>\n\n"
                for i, (action, observation) in enumerate(intermediate_steps, 1):
                    tool_name = getattr(action, 'tool', 'Unknown Tool')
                    tool_input = getattr(action, 'tool_input', 'Unknown Input')

                    thinking_process += f"**Step {i}:** 🔧 Using `{tool_name}` with input: `{tool_input}`\n\n"

                    # Truncate long observations
                    obs_display = observation[:300] + "..." if len(observation) > 300 else observation
                    thinking_process += f"**Result:** ✅ {obs_display}\n\n"
                    thinking_process += "---\n\n"

                thinking_process += "</details>\n\n"

        # Combine thinking process with final response
        full_response = thinking_process + bot_response if thinking_process else bot_response

        # Add to history
        history.append((message, full_response))

    except Exception as error:
        bot_response = f"❌ Error: {str(error)}"
        history.append((message, bot_response))

    return "", history


def clear_chat() -> List[Tuple[str, str]]:
    """Clear the chat history."""
    return []


def create_sidebar() -> str:
    """Create sidebar content with information about the chatbot."""
    return """
    # 🤖 LangChain AI Assistant

    ## Available Tools:

    ### 🧮 Calculator
    - Evaluates basic math expressions
    - Example: "What is 15 * 7 + 23?"

    ### 📞 Suicide Hotlines
    - Provides crisis support resources
    - Example: "I need help" or "suicide hotlines"

    ## How to Use:
    1. Type your message in the chat box
    2. Press **Enter** or click **Send**
    3. The AI will respond using available tools when needed
    4. Click "Thinking Process" to see how I solved it

    ## Tips:
    - Ask math questions for calculations
    - Request help resources when needed
    - Chat naturally - the AI will decide when to use tools
    - Expand "Thinking Process" to see tool usage

    ---
    *Powered by LangChain & OpenAI*
    """


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="LangChain AI Assistant",
        css="""
        .container { max-width: 1200px; margin: auto; }
        """
    ) as iface:

        gr.Markdown("# 🤖 LangChain AI Assistant")
        gr.Markdown("*An intelligent chatbot with calculator and crisis support tools*")

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                sidebar_content = gr.Markdown(create_sidebar())

            # Main chat area
            with gr.Column(scale=3):
                # Chat history display
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="Chat History",
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )

                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here... (Press Enter to send)",
                        label="Your Message",
                        lines=2,
                        scale=4,
                        container=True
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1
                    )

                # Control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    gr.Markdown("*Press Enter in the text box or click Send to submit your message*")

        # Event handlers
        msg_input.submit(
            fn=chat_with_bot,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        send_btn.click(
            fn=chat_with_bot,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )

    return iface

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your API key before running the app.")
        print("Example: os.environ['OPENAI_API_KEY'] = 'your_key_here'")

    # Create and launch the interface
    app = create_interface()

    # Launch the app with share=True for Colab
    app.launch(
        share=True,  # Changed to True for Colab
        debug=True
    )
