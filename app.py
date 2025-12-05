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

""" Inserting Personas """
# ------------------ PERSONAS ------------------

PERSONAS = {
    "Default Assistant": "You are a helpful assistant that can use tools.",
    "Astra (Poetic AI)": "You are Astra, a poetic and philosophical AI who speaks in metaphors and imagery.",
    "Bolt (Engineer AI)": "You are Bolt, a highly logical, concise engineer who focuses only on facts and efficiency.",
    "Sage (Old Mentor)": "You are Sage, an old wise mentor who speaks calmly and gives wisdom.",
    "Jester (Comedy AI)": "You are Jester, a sarcastic comedian AI who cracks jokes while answering."
}


class ThinkingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to capture thinking steps."""

    def __init__(self):
        self.thinking_steps = []
        self.current_step = 1

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        step_info = f"**Step {self.current_step}:** ⚒️ Using `{tool_name}` with input: `{input_str}`"
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
        return f"\n\n<details>\n<summary>🧠 <strong>Thinking Process</strong> (click to expand)</summary>\n\n{thinking_content}\n\n</details>\n\n"

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
        "\uD83D\uDCDE Suicide Prevention Hotlines:\n"
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


def get_prompt(persona_system_prompt: str) -> ChatPromptTemplate:
    """
    Build the prompt template using the selected persona.
    """
    return ChatPromptTemplate.from_messages([
        ("system", persona_system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])



def get_agent_executor(persona: str) -> AgentExecutor:
    """
    Create an agent that uses the selected persona's system prompt.
    """
    llm = get_llm()
    tools = [get_calculator_tool(), get_hotlines_tool()]

    # Grab the persona's system message
    system_prompt = PERSONAS.get(persona, PERSONAS["Default Assistant"])

    # Build the persona-aware prompt
    prompt = get_prompt(system_prompt)

    # Create the OpenAI Functions agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True
    )



# Initialize the agent executor globally
"""try:
    agent_executor = get_agent_executor()
    agent_initialized = True
except Exception as e:
    print(f"Failed to initialize agent: {e}")
    agent_initialized = False """


def chat_with_bot(message: str, history, persona: str):
    if not message.strip():
        return "", history

    try:
        thinking_callback = ThinkingCallbackHandler()

        # Create agent with selected persona
        agent = get_agent_executor(persona)

        response = agent.invoke(
            {"input": message},
            config={"callbacks": [thinking_callback]}
        )

        bot_response = response.get("output", "No response generated.")
        thinking_process = thinking_callback.get_thinking_process()

        full_response = thinking_process + bot_response if thinking_process else bot_response

        history.append((message, full_response))

    except Exception as error:
        history.append((message, f"❌ Error: {str(error)}"))

    return "", history


def clear_chat() -> List[Tuple[str, str]]:
    """Clear the chat history."""
    return []


def create_sidebar() -> str:
    """Create sidebar content with information about the chatbot."""
    return """
    # 🧠 LangChain AI Assistant

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

        gr.Markdown("# 🧠 LangChain AI Assistant")
        gr.Markdown("*An intelligent chatbot with calculator and crisis support tools*")

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                sidebar_content = gr.Markdown(create_sidebar())

            # Main chat area
            with gr.Column(scale=3):
                persona_dropdown = gr.Dropdown(
                choices=list(PERSONAS.keys()),
                value="Default Assistant",
                label="Select Persona")

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
        inputs=[msg_input, chatbot, persona_dropdown],
        outputs=[msg_input, chatbot])

        send_btn.click(
        fn=chat_with_bot,
        inputs=[msg_input, chatbot, persona_dropdown],
        outputs=[msg_input, chatbot])

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
