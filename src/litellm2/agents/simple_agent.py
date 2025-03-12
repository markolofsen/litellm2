"""
Simple Agent - A lightweight implementation of an AI agent with tools
"""

import logging
from typing import Callable, List
from smolagents import LiteLLMModel, ToolCallingAgent, GradioUI as OriginalGradioUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom GradioUI class to handle None values
class GradioUI(OriginalGradioUI):
    """A wrapper around smolagents GradioUI that handles None values properly"""

    def __init__(self, agent, file_upload_folder=None):
        # Make sure name and description are strings, not None
        if not hasattr(agent, 'name') or agent.name is None:
            agent.name = "Agent"
        if not hasattr(agent, 'description') or agent.description is None:
            agent.description = ""

        # Call the parent constructor
        super().__init__(agent, file_upload_folder)


class SimpleAgent:
    """
    Simple agent that executes tools based on user queries
    """

    def __init__(self, model_id: str = None):
        """
        Initialize a SimpleAgent without any tools

        Args:
            api_base: API base URL (default: OpenRouter)
            model_id: Model ID to use
        """
        self.api_base = 'https://openrouter.ai/api/v1'
        self.model_id = model_id or "openrouter/openai/gpt-4o-mini"
        self.tools = []
        self.llm_agent = None
        # Add name and description attributes for Gradio UI
        self.name = "Simple Agent"
        self.description = "A lightweight AI agent that can execute various tools based on user queries."

    def add_tools(self, tools: List[Callable]) -> None:
        """
        Add multiple tools to the agent

        Args:
            tools: List of tool functions decorated with @tool
        """
        for tool in tools:
            self.tools.append(tool)
        self._init_llm_agent()  # Reinitialize once after adding all tools

    def _init_llm_agent(self):
        """
        Initialize or reinitialize the LLM agent with current tools
        """
        try:
            # Create LiteLLM model
            model = LiteLLMModel(
                model_id=self.model_id,
                api_base=self.api_base
            )

            # Create agent with tools and model
            self.llm_agent = ToolCallingAgent(
                tools=self.tools,
                model=model,
                max_steps=5,
                verbosity_level=1
            )
            logger.info(f"Agent initialized with {len(self.tools)} tools")
        except Exception as e:
            logger.error(f"Error initializing LLM agent: {e}")
            self.llm_agent = None

    def run(self, query: str) -> str:
        """
        Run a query using the LLM agent

        Args:
            query: User query in natural language
        """
        if not self.llm_agent:
            if len(self.tools) == 0:
                return "Error: No tools added to agent. Add tools with add_tool() first."
            else:
                self._init_llm_agent()  # Try to initialize again

            if not self.llm_agent:
                return "Error: Failed to initialize LLM agent."

        try:
            # Process the query with ToolCallingAgent
            response = self.llm_agent(query)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"

    def launch_gradio_ui(self, server_name="0.0.0.0", server_port=7860):
        """
        Launch a Gradio UI for interacting with the agent

        Args:
            server_name: Server hostname
            server_port: Server port
        """
        if not self.llm_agent:
            if len(self.tools) == 0:
                print("Error: No tools added to agent. Add tools with add_tool() first.")
                return
            else:
                self._init_llm_agent()  # Try to initialize again

            if not self.llm_agent:
                print("Error: Failed to initialize LLM agent.")
                return

        try:
            print(f"Launching Gradio UI with {len(self.tools)} tools")
            # Use our custom GradioUI class that handles None values
            GradioUI(self.llm_agent).launch(
                server_name=server_name,
                server_port=server_port,
                share=False,
                inbrowser=True
            )
        except Exception as e:
            logger.error(f"Error launching Gradio UI: {e}")
            print(f"Failed to launch UI: {e}")
