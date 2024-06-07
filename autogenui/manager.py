# a manager class that can
# load an autogen flow run an autogen flow and return the response to the client


from typing import Dict
import autogen
from .utils import parse_token_usage
import time

import os
import tempfile

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

def get_merged_terminate_messages(chat_result) -> str:
    terminate_messages = []

    # Iterate over chat history to find messages containing "TERMINATE"
    for message in chat_result.chat_history:
        if 'content' in message:
            content = message['content']
            if content and 'TERMINATE' in content:
                    # Remove "TERMINATE" string from the content
                    cleaned_content = content.replace('TERMINATE', '').strip()
                    terminate_messages.append(cleaned_content)

    # Merge the cleaned messages into one string
    merged_message = ' '.join(terminate_messages)
    
    return merged_message

class Manager(object):
    def __init__(self) -> None:

        pass

    def run_flow(self, prompt: str, vector_name: str, chat_history = []) -> None:

        backup_yaml = os.path.join(os.path.dirname(__file__), 'prompt_template.yaml')
        print("backup prompt yaml: {backup_yaml}")
        try:
            code_writer_system_message = load_prompt_from_yaml("system", prefix=vector_name, file_path=backup_yaml)
        except Exception as err:
             print("No promptConfig.{vector_name}.system set - loading default promptConfig.autogen.system")
             code_writer_system_message = load_prompt_from_yaml("system", prefix="autogen", file_path=backup_yaml)

        model = load_config_key("model", vector_name=vector_name, kind="vacConfig")

        code_writer_agent = ConversableAgent(
            "code_writer_agent",
            system_message=code_writer_system_message,
            llm_config={"config_list": [{"model": model or "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
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
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # never take human input for this agent for safety.
        )
        start_time = time.time()

        # Register the tool signature with the assistant agent.
        code_writer_agent.register_for_llm(name="calculator", description="A simple calculator")(calculator)

        # Register the tool function with the user proxy agent.
        code_executor_agent.register_for_execution(name="calculator")(calculator)

        chat_result = code_executor_agent.initiate_chat(
            code_writer_agent,
            message=prompt,
            summary_method="reflection_with_llm",
            max_turns=10
        )

        print(chat_result)


        filtered = get_merged_terminate_messages(chat_result)
        print("FILTERED")
        print(filtered)

        #chat_result.chat_history.append({'content': filtered, 'role': 'assistant'})

        response = {
            "messages": chat_result.chat_history,
            "usage": "", #parse_token_usage(logged_history),
            "duration": time.time() - start_time,
        }

        return response
