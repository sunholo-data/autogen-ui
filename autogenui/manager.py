# a manager class that can
# load an autogen flow run an autogen flow and return the response to the client


from typing import Dict
import autogen
from .utils import parse_token_usage
import time

import os
import tempfile
import datetime

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.code_utils import create_virtual_env

from sunholo.utils import load_config_key
from sunholo.langfuse.prompts import load_prompt_from_yaml



from .tools import calculator

venv_dir = ".venv2"
venv_context = create_virtual_env(venv_dir)

# Create a temporary directory to store the code files.
temp_dir = tempfile.TemporaryDirectory()

class Manager(object):
    def __init__(self) -> None:

        pass

    def run_flow(self, prompt: str, vector_name: str, chat_history = [], stream=False) -> None:

        backup_yaml = os.path.join(os.path.dirname(__file__), 'prompt_template.yaml')
        print("backup prompt yaml: {backup_yaml}")
        try:
            code_writer_system_message = load_prompt_from_yaml("system", prefix=vector_name, file_path=backup_yaml)
        except Exception as err:
             print("No promptConfig.{vector_name}.system set - loading default promptConfig.autogen.system")
             code_writer_system_message = load_prompt_from_yaml("system", prefix="autogen", file_path=backup_yaml)

        model = load_config_key("model", vector_name=vector_name, kind="vacConfig")
        if not stream:
            config_list = {"config_list": [{"model": model or "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}
        else:
            config_list = {"config_list": [{"model": model or "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "stream": True,}

        code_writer_agent = ConversableAgent(
            "code_writer_agent",
            system_message=code_writer_system_message,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            llm_config=config_list,
            code_execution_config=False,  # Turn off code execution for this agent.
        )

        # Create a local command line code executor.
        executor = LocalCommandLineCodeExecutor(
            timeout=60,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
            virtual_env_context=venv_context
        )
        # Create an agent with code executor configuration.
        code_executor_agent = ConversableAgent(
            "code_executor_agent",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # never take human input for this agent for safety.
        )
        start_time = time.time()

        # 4. Define Agent-specific Functions
        def weather_forecast(city: str) -> str:
            return f"The weather forecast for {city} at {datetime.now()} is sunny."

        autogen.register_function(
            weather_forecast, caller=code_writer_agent, executor=code_executor_agent, description="Weather forecast for a city"
        )
        autogen.register_function(
            calculator, caller=code_writer_agent, executor=code_executor_agent, description="A simple calculator"
        )

        print(
            f" - on_connect(): Initiating chat with agent {code_writer_agent} using message '{prompt}'",
            flush=True,
        )
        chat_result = code_executor_agent.initiate_chat(
            code_writer_agent,
            message=prompt,
            summary_method="reflection_with_llm",
            max_turns=10
        )

        print(chat_result)

        print("SUMMARY")
        print(chat_result.summary)

        response = {
            "messages": chat_result.chat_history,
            "summary": chat_result.summary,
            "usage": "", #parse_token_usage(logged_history),
            "duration": time.time() - start_time,
        }

        return response
