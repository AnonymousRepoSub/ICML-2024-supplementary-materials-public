from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.code_utils import extract_code
from autogen.math_utils import get_answer
from autogen.agentchat import Agent
import autogen
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from utils_gpt import is_termination_msg_mathchat
import copy
from openai import (
    BadRequestError,
)

class MathUserProxyAgent(MathUserProxyAgent):
    MAX_CONSECUTIVE_AUTO_REPLY = 15
    DEFAULT_REPLY = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"
    PROMPTS = """Let's solve a math problem.
Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```
If some packages are missing, you could also suggest a code to install the corresponding package.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) and functions you know in the context of this conversation.

Please 
(1) do not mix suggested Python codes and function calls in one step.
(2) You MUST remember that you don’t have a function named "python" available.

You must follow the formats below to write your Python code:
```python
# your code
```

3. Wait for me to give the results or wait for the executed results of the function call.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem: 
"""

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
            name: Optional[str] = "MathChatAgent",
            is_termination_msg: Optional[
                Callable[[Dict], bool]
            ] = is_termination_msg_mathchat,
            human_input_mode: Optional[str] = "NEVER",
            default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_REPLY,
            max_invalid_q_per_step=3,
            use_py = True,
            **kwargs,
    ):
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            default_auto_reply=default_auto_reply,
            max_invalid_q_per_step=max_invalid_q_per_step,
            **kwargs,
        )
        del self._reply_func_list[2]
        self.register_reply([Agent, None], MathUserProxyAgent._generate_math_reply, position=4)
        del self._reply_func_list[3]
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=MathUserProxyAgent.generate_function_call_reply, position=3)
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=MathUserProxyAgent._check_final_result, position=0)

        self.max_function_call_trial = 3 
        self.query = None
        self.answer = None
        self.use_py = use_py
        
        self.function_statistic = None

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
                    Otherwise, the error may be caused by the function itself. Please directly reply me with TERMINATE. We need to terminate the conversation. "
                    error_message = func_return["content"]
                    return True, "The func is executed failed." + error_message + revise_prompt
        return False, None

    def initiate_chat(
            self,
            recipient,
            query: None,
            answer: None,
            silent: Optional[bool] = False,
            **context,
    ):
        self.query = query
        if not isinstance(answer, str):
            answer = str(answer)
            if answer.endswith('.0'):
                answer = answer[:-2]
            self._answer = answer
        else:
            self._answer = answer
        self.logs = {}
        self._prepare_chat(recipient, True)

        chat_history = []
        error_message = None

        try:
            if self.use_py:
                prompt = self.PROMPTS + context["problem"]
            else:
                prompt = self.PROMPTS_NOPY + context["problem"]
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
                and get_answer(messages) is not None
                and get_answer(messages) != ""
        ):
            if get_answer(messages) == self._answer:
                self.logs["is_correct"] = 1
                return True, "The result is Correct. Please reply me with TERMINATE."
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