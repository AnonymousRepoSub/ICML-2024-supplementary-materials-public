from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.code_utils import extract_code
import autogen
from autogen.agentchat import Agent
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple
from utils import is_termination_msg_gaia, get_answer_gaia, normalize_answer, get_env_variables
import tiktoken
from datetime import datetime
import copy
from openai import (
    BadRequestError,
)

def trim_string(string: str, limit=5000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(string)
    if len(tokens) <= limit:
        return string
    else:
        return encoding.decode(tokens[:limit//2]) + f"\n...The log is too long, you should print less than {limit} tokens...\n" + encoding.decode(tokens[-limit//2:])

class GaiaUserProxyAgent(UserProxyAgent):
    MAX_CONSECUTIVE_AUTO_REPLY = 20
    DEFAULT_REPLY = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, format it as FINAL ANSWER: [YOUR FINAL ANSWER]), if you think the problem is unsolvable, reply TERMINATE.)"
    PROMPTS = ("You are a helpful AI assistant, and today's date is "
    + datetime.now().date().isoformat()
    + """.
The user will ask you a question. Answer this question using your coding and language skills.
In the following cases, suggest python code (presented in a coding block beginning with```python) or shell script (presented in a coding block beginning with ```sh) for the user to execute:
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Answer the question step by step. You should devise a plan before solving the problems.
Note that when you use python, you should always use the 'print' function to output. The python code working directory is {code_dir}, all the files are contained in this directory, you should always use **absolute path** to access any related files. 
The user cannot provide any other feedback or perform any other action beyond executing the code appearing in the code block. The user can't modify your code, so do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user. Don't include multiple code blocks in one response. Do not ask users to copy and paste code or results. Instead, use the 'print' function for the output when relevant. Check the execution result reported by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Question:""".strip())

    PROMPTS_NOPY = """Let's solve a math problem.
Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through functions you know in the context of this conversation.
3. Wait for me to give the results or wait for the executed results of the function call.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem: 
"""

    def __init__(
            self,
            name: Optional[str] = "GaiaChatAgent",
            is_termination_msg: Optional[
                Callable[[Dict], bool]
            ] = is_termination_msg_gaia,
            human_input_mode: Optional[str] = "NEVER",
            default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_REPLY,
            use_py = True,
            **kwargs,
    ):
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            default_auto_reply=default_auto_reply,
            **kwargs,
        )
        del self._reply_func_list[3]
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=GaiaUserProxyAgent.generate_function_call_reply, position=3)
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=GaiaUserProxyAgent._check_final_result, position=0)
        del self._reply_func_list[7]
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=GaiaUserProxyAgent.generate_code_execution_reply, position=7)

        self.max_function_call_trial = 3 
        self.query = None
        self.answer = None
        self.use_py = use_py
        
        self.function_statistic = None
    
    def generate_code_execution_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Union[Dict, Literal[False]]] = None,
    ):
        code_execution_config = config if config is not None else self._code_execution_config
        if code_execution_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        last_n_messages = code_execution_config.pop("last_n_messages", 1)

        messages_to_scan = last_n_messages
        if last_n_messages == "auto":
            messages_to_scan = 0
            for i in range(len(messages)):
                message = messages[-(i + 1)]
                if "role" not in message:
                    break
                elif message["role"] != "user":
                    break
                else:
                    messages_to_scan += 1
        for i in range(min(len(messages), messages_to_scan)):
            message = messages[-(i + 1)]
            if not message["content"]:
                continue
            code_blocks = extract_code(message["content"])
            if len(code_blocks) == 1 and code_blocks[0][0] == "unknown":
                continue

            exitcode, logs = self.execute_code_blocks(code_blocks)
            code_execution_config["last_n_messages"] = last_n_messages
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            logs = trim_string(logs)
            return True, f"exitcode: {exitcode} ({exitcode2str})\nCode output: {logs}"

        code_execution_config["last_n_messages"] = last_n_messages

        return False, None
            

    def generate_function_call_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.ConversableAgent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[Dict, None]]:
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message:
            is_exec_success, func_return = self.execute_function(
                message["function_call"])
            func_name = message["function_call"].get("name", "")
            if self.function_statistic is None:
                self.function_statistic = {}
            if func_name not in self.function_statistic.keys():
                self.function_statistic[func_name] = False
            if is_exec_success:
                func_return["content"] = trim_string(func_return["content"])
                self.max_function_call_trial = 3
                self.function_statistic[func_name] = True
                return True, func_return
            else:
                if self.max_function_call_trial == 0:
                    error_message = func_return["content"]
                    self.logs["is_correct"] = 0
                    self.max_function_call_trial = 3
                    return True, "The func is executed failed many times. " + error_message + ". Please directly reply me with TERMINATE. We need to terminate the conversation."
                else:
                    revise_prompt = "You may make a wrong function call (It may due the arguments you provided doesn't fit the function arguments like missing required positional argument). \
                    If you think this error occurs due to you make a wrong function arguments input and you could make it success, please try to call this function again using the correct arguments. \
                    Otherwise, the error may be caused by the function itself. If this is the case, please directly reply me with TERMINATE. We need to terminate the conversation. "
                    error_message = func_return["content"]
                    return True, "The func is executed failed." + error_message + revise_prompt
        return False, None

    def initiate_chat(
            self,
            recipient,
            query: None,
            answer: None,
            file: None,
            code_dir: None,
            silent: Optional[bool] = False,            
            **context,
    ):
        self.query = query
        self._answer = normalize_answer(answer)
        self.logs = {}
        self._prepare_chat(recipient, True)
        envs = get_env_variables()

        chat_history = []
        error_message = None

        try:
            if self.use_py:
                prompt = self.PROMPTS.format(code_dir=code_dir) + " " + context["problem"]
            else:
                prompt = self.PROMPTS_NOPY + context["problem"]
            if file:
                prompt += "The path to the file you need is: " + file
            if len(list(envs.keys())) != 0:
                prompt += " The following environment variables are available and can be accessed directly through os.environ: " + str(list(envs.keys()))
            self.send(prompt, recipient, silent=silent)
        except BadRequestError as e:
            error_message = str(e)
            self.logs["is_correct"] = 0
            print("error information: {}".format(error_message))

        key = list(self.chat_messages.keys())[0]
        chat_messages = self.chat_messages[key]
        for item in chat_messages:
            chat_history.append(item)
        if error_message is not None:
            chat_history.append(error_message)
        recipient.reset()
        logs_return = copy.deepcopy(self.logs)
        self._reset()
        return logs_return, chat_history

    def _check_final_result(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.Agent] = None,
            config: Optional[Any] = None):
        messages = messages[-1]

        if isinstance(messages, dict):
            messages = messages.get("content")
            if messages is None:
                return False, None

        cb = extract_code(messages)
        contain_code = False
        for c in cb:
            if c[0] == "python" or c[0] == "wolfram":
                contain_code = True
                break
        if (
                not contain_code
                and get_answer_gaia(messages) is not None
                and get_answer_gaia(messages) != ""
        ):
            if get_answer_gaia(messages) == self._answer:
                self.logs["is_correct"] = 1
                return True, "The result is Correct. Please reply me with TERMINATE."
            elif "TERMINATE" in messages:
                self.logs["is_correct"] = 0
                return False, None
            else:
                self.logs["is_correct"] = 0
                return False, None
        else:
            return False, None

    def _reset(self):
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ""
        self.last_reply = None

        self.query = None
        self.answer = None
        self.logs = {}
        self.max_function_call_trial = 3
        
    def clear_function_statistic(self):
        self.function_statistic = None