from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.code_utils import extract_code, UNKNOWN, infer_lang
from autogen.math_utils import get_answer
import autogen
from autogen.agentchat import Agent
from typing import Any, Callable, Dict, List, Optional, Union
from utils import is_termination_msg_mathchat
import copy
from openai import (
    BadRequestError,
)

class TabMWPAgentProxy(MathUserProxyAgent):
    MAX_CONSECUTIVE_AUTO_REPLY = 12

    DEFAULT_REPLY = """Continue, keep solving the problem. If you get to the "final" answer, put only the final answer in \\boxed{} (no approximation, no explanation, no other words).
If the result indicates there is an error, fix the error and output again. If the error can't be fixed or if the task is not solved even after the code or function is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
If this message repeat multiple times, your answer may be wrong. Please check your answer and try again."""

    PROMPTS = """<<SYS>>Let's solve a problem with a table that gives you some necessary information.
Please follow this process to solve the problem:
1. Think about the problem carefully.
2. Solve the problem step by step (do not over-divide the steps).
3. Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) or functions you know in the context of this conversation.
4. Wait for the user to give the results or wait for the executed results of the function call.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

The following instructions are important for you to derive the correct answer without making mistakes:
# Describe your motivation and listed all argument you want to input before you suggest to make a function call.
# Do not make a function call until you complete the analysis of the problem.
# Do NOT mix suggested Python codes and function calls in one step.
# You don’t have a function named "python" available.
# You don't need to execute the code block by yourself. The user will execute it for you.
# Do not think about the final result before the user provides you with the execution result of the Python code or function call. Never trust your calculation and code execution ability.
# You must follow the formats below to write your code:
```python
    your code (do not use input() or something else that need user input. User will not input anything.)
    （use only one this block in your reply)
```
# Once you make a wrong function call, check the error carefully, and correct the error based on the error message before you make a new function call.
# If the result of your code or function call from user is not as your expectation, believe the user's result.
# If you reach to the "final" answer, put only the final answer in \\boxed{YOUR ANSWER} (DO NOT APPROXIMATE).

If you are asked to choose an answer from some choices, make sure the answer in the box is completely the same as the choice you choose.<<SYS>>

Problem: 
"""

    PROMPTS_NOPY = """Let's solve a problem with a table that gives you some necessary information.
Query requirements:
(1) You MUST remember that you don’t have a function named "python" available.
(2) Describe your motivation and listed all argument you want to input before you suggest to make a function call.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through functions you know in the context of this conversation.
3. Wait for me to give the results or wait for the executed results of the function call.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

If you are asked to choose an answer from some choices, make sure the answer in the box is completely the same as the choice you choose.

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
            use_py = True,
            max_invalid_q_per_step=1,
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
        self.register_reply([Agent, None], TabMWPAgentProxy._generate_math_reply, position=4)
        del self._reply_func_list[3]
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=TabMWPAgentProxy.generate_function_call_reply, position=3)
        self.register_reply(trigger=autogen.ConversableAgent,
                            reply_func=TabMWPAgentProxy._check_final_result, position=1)

        self.max_function_call_trial = 3
        self.query = None
        self.answer = None
        self.use_py = use_py
        
        self.function_statistic = None

    def _generate_math_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ):
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        message = message.get("content", "")
        code_blocks = extract_code(message)

        if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
            return True, self._default_auto_reply
        is_success, all_success = True, True
        reply = ""
        for code_block in code_blocks:
            lang, code = code_block
            if not lang:
                lang = infer_lang(code)
            if lang == "python":
                output, is_success = self.execute_one_python_code(code)
            elif lang == "wolfram":
                output, is_success = self.execute_one_wolfram_query(code)
            else:
                output = "Error: Unknown language."
                is_success = False

            reply += output + "\n"
            if not is_success:
                all_success = False
                self._valid_q_count -= 1  

        reply = f"Your Python code execution result is: {reply.strip()}"

        return True, reply

    def generate_function_call_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.ConversableAgent] = None,
            config: Optional[Any] = None,
    ) -> tuple[bool, dict[str, str]] | tuple[bool, str] | tuple[bool, None]:
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message:
            is_exec_success, func_return = self.execute_function(message["function_call"])
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
            query: dict = None,
            silent: Optional[bool] = False,
            **context,
    ):
        self.query = query
        table_title = query['table_title'] if query['table_title'] is not None else ""
        problem = f"{query['question']}\nHere is the table that provides you with some necessary information:\n{table_title}\n{query['table']}\nPlease solve the problem step by step (do not over-divide the steps)."
        if query['choices'] != "null":
            problem += f"\nChoose your answer from the following choices:\n{query['choices']}"

        answer = query['answer']
        if not isinstance(answer, str):
            answer = str(answer)
        if "." in answer:
            answer = str(float(answer))
        if answer.endswith('.0') or answer.endswith('.00'):
            answer = answer[:-2]
        if "/" in answer:
            answer = answer.split('/')
            answer = str(int(answer[0]) / int(answer[1]))
            answer = str(float(answer))
        answer = answer.replace(',', '')
        self._answer = answer
        self.logs = {}
        self._prepare_chat(recipient, True)

        chat_history = []
        error_message = None

        try:
            if self.use_py:
                prompt = self.PROMPTS + problem
            else:
                prompt = self.PROMPTS_NOPY + problem
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
        chat_history.append({
            "correct_answer": answer,
            "is_correct": self.logs.get("is_correct", 0)
        })
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
            answer = get_answer(messages)
            if not isinstance(answer, str):
                answer = str(answer)
            if "." in answer:
                try:
                    answer = str(float(answer))
                except Exception:
                    answer = str(answer)
            if answer.endswith('.0'):
                answer = answer[:-2]
            if "/" in answer:
                answer = answer.split('/')
                try:
                    answer = str(int(answer[0]) / int(answer[1]))
                    answer = str(float(answer))
                    if answer.endswith('.0'):
                        answer = answer[:-2]
                except Exception:
                    answer = str(get_answer(messages))
            answer = answer.replace(',', '')
            answer = answer.replace('"', '')
            if "frac" in answer:
                answer = answer.replace("}{", "/")
                answer = answer.replace("\\frac", "")
                answer = answer.replace("{", "")
                answer = answer.replace("}", "")
                answer = answer.split('/')
                try:
                    answer = str(int(answer[0]) / int(answer[1]))
                except Exception:
                    pass
            if answer == self._answer:
                self.logs["is_correct"] = 1
                print('Correct Answer. (This message is unseen by the assistant)')
                return True, 'We got your answer, please reply me with "TERMINATE" (all in upper case).'
            else:
                self.logs["is_correct"] = 0
                print(f'Wrong Answer, correct answer is {self._answer}. (This message is unseen by the assistant)')
                return True, 'We got your answer, please reply me with "TERMINATE" (all in upper case).'
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