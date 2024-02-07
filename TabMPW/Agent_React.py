import openai
import autogen
import json
import re
import traceback
from functools import partial
from typing import Dict, Callable
from autogen.code_utils import execute_code, extract_code
from utils import retry_with_exponential_backoff, normalize_answer, extract_code, trim_string

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


def parse_text(response):
    text = response.choices[0].message.content

    if "terminate" in text.lower():
        return {"final_answer": "I don't know", "thought": "TERMINATE"}

    if not text.startswith("Thought:"):
        text = "Thought: " + text

    pattern1 = r"Thought\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input:[\s]*(.*)"
    pattern2 = r"(.*?)\s*FINAL ANSWER:\s*(\S+)"

    match2 = re.findall(pattern2, text, re.IGNORECASE | re.MULTILINE)
    if match2:
        return {
            "thought": match2[0][0].strip(),
            "final_answer": match2[0][1].strip()
        }

    match1 = re.search(pattern1, text, re.DOTALL)
    if match1:
        return {
            "thought": match1.group(1).strip(),
            "action": match1.group(2).strip(),
            "input": match1.group(3).strip()
        }

    return None


def extract_and_execute_code(code, lang, use_docker, work_dir):
    _, code = extract_code(code)[0]
    return execute_code(code, lang=lang, use_docker=use_docker, work_dir=work_dir)


class ReActAgent:
    TASK_PROMPT = """[INST]<<SYS>>Your mission is to answer a question by using the following tools.
Below is a list of the tools you can use and their detailed descriptions:
{tool_desc}

You should always follow the below template. You should reply with one <Action, Action Input> doublet and wait for the result before proceeding to the next round.
TEMPLATE:
The reason why you suggested the following action.
Action: (Should be exactly one of the tools in [{tool_names}])
Action Input: (the arguments passed to action or the code block you want to execute. The format is described in the tool description)
TEMPLATE END
Requirement:
# Do not leave a placeholder. For example, do not say "FINAL ANSWER: (Replace this with the number received from the Python action)".
# Do not execute the tool or assume or modify the "Observation" yourself. If the observation is unexpected, analyze it and try to fix it by modifying your previous action with a new action.
# Only take ONE action. Solve the problem step by step.
# DO NOT assume the result in the next step and say "after receiving the result".
# If you suggest an action, wait for the result before proceeding to the next round without saying anything.

If you derive to the final answer, you should reply the "FINAL ANSWER" with the following template:
TEMPLATE:
Thought: I now know the final answer.
FINAL ANSWER: (Put your final answer here by yourself. It should be only the number or choice, no explanations, no other words after final answer. Do not approximate the decimal point, reply with the exact number)
TEMPLATE END<<SYS>>
"""

    def __init__(self, config_list, use_docker=False, work_dir="coding_react"):
        self._client = autogen.OpenAIWrapper(
            config_list=config_list, temperature=0, max_tokens=400, top_p=0.95)
        self.use_docker = use_docker
        self.functions = []
        self.unsupported_tools = ["video"]
        self.work_dir = work_dir
        self._function_map = {}
        self.function_statistic = None
        self.max_function_call_trial = 3
        self.logs = {}
        self.base_tools = {
            "Python": partial(extract_and_execute_code, lang="python", work_dir=work_dir, use_docker=use_docker),
        }

    def register_function(self, function_map: Dict[str, Callable]):
        self._function_map.update(function_map)

    @retry_with_exponential_backoff
    def completion_with_backoff(self, **kwargs):
        try:
            return self._client.create(cache_seed=None, **kwargs)
        except openai.BadRequestError as e:
            traceback.print_exc()
            return {}

    def not_capable(self, task):
        for tool in self.unsupported_tools:
            if tool in task['Annotator Metadata']['Tools'].lower():
                return True
        return False

    def update_function_signature(self, func_sig, is_remove):
        if is_remove:
            self.functions = [
                func for func in self.functions if func["name"] != func_sig
            ]
        else:
            self.functions = [func for func in self.functions if func.get(
                "name") != func_sig["name"]] + [func_sig]

    def get_tool_descriptions(self):
        tool_descs = f"(1) Python. Executes the exact python code you provide. You MUST always use 'print' function for the output."
        tool_descs += "\nInput: Python code block, should follow ```python\n YOUR CODE\n ```"

        for i, function in enumerate(self.functions):
            desc = f"({i + 2}) {function['name']}. {function['description']}\n"
            args_desc = json.dumps(function["parameters"]["properties"])
            desc += f"Input: {args_desc}"
            tool_descs += desc
            tool_descs += "\n"
        return tool_descs

    def get_tool_names(self):
        tool_names = list(self.base_tools.keys())
        for function in self.functions:
            tool_names.append(function["name"])
        return tool_names

    def execute_function(self, func_call):
        func = self._function_map.get(func_call["name"], None)
        pattern = r"\{[\s\S]*?\}"

        is_exec_success = False
        error = None
        content = None
        if func is not None:
            try:
                arguments = re.findall(pattern, func_call["arguments"])[0]
            except Exception as e:
                error = e
                arguments = None

            if arguments is not None:
                print(
                    colored(
                        f"\n>>>>>>>> EXECUTING FUNCTION {func_call['name']}...", "magenta"),
                    flush=True,
                )
                try:
                    content = func(json.loads(arguments))
                    is_exec_success = True
                    content = "Execution succeeded. Tool output:\n" + content
                except Exception as e:
                    content = f" Error: {e}. "
        else:
            if error:
                content = f"Error: {error}"
            else:
                content = f"Error: Tool {func_call['name']} not found."

        return is_exec_success, {
            "name": func_call["name"],
            "role": "function",
            "content": str(content),
        }

    def clear_function_statistic(self):
        self.function_statistic = None

    def generate_tool_response(self, response):
        tool_name, args = response["action"], response["input"]
        tool_call = {"name": tool_name, "arguments": args}
        if tool_name in self.base_tools.keys():
            exitcode, logs, _ = self.base_tools[tool_name](args)
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            if len(trim_string(logs)) == 0:
                logs = "No output. Do you forget to use the 'print'?"
            func_return = exitcode2str + " Code output:\n" + trim_string(logs)
            return func_return

        if tool_name not in self._function_map.keys():
            return "The action is not valid. It should be excatly one of the following tool names: " + ", ".join(
                self.get_tool_names())

        is_exec_success, func_return = self.execute_function(tool_call)

        if self.function_statistic is None:
            self.function_statistic = {}
        if tool_name not in self.function_statistic.keys():
            self.function_statistic[tool_name] = False
        if is_exec_success:
            self.function_statistic[tool_name] = True
            return trim_string(func_return["content"])
        else:
            if self.max_function_call_trial == 0:
                error_message = func_return["content"]
                self.logs["is_correct"] = 0
                self.max_function_call_trial = 3
                return "The tool has failed execution many times. " + error_message + ". Please directly reply me with TERMINATE. We need to terminate the conversation."
            else:
                revise_prompt = "You might have made a wrong tool call (It may due the input you provided doesn't fit the arguments). \
                If you think this error occurs due to you making a wrong action input and you can fix it, please try to call this action again using the correct arguments. \
                Otherwise, the error may be caused by the tool itself. If you think this is the case, please directly reply me with TERMINATE. We need to terminate the conversation. "
                error_message = func_return["content"]
                return "The func is executed failed." + error_message + revise_prompt

    def query(self, instance, to_print=False):
        table_title = instance['table_title'] if instance['table_title'] is not None else ""
        problem = f"{instance['question']}\nHere is the table that provides you with some necessary information:\n{table_title}\n{instance['table']}."
        if instance['choices'] != "null":
            problem += f"\nChoose your answer from the following choices:\n{instance['choices']}"

        answer = instance['answer']
        if not isinstance(answer, str):
            answer = str(answer)
        if "." in answer:
            answer = str(float(answer))
        if answer.endswith('.0'):
            answer = answer[:-2]
        if "/" in answer:
            answer = answer.split('/')
            answer = str(int(answer[0]) / int(answer[1]))
            answer = str(float(answer))
        answer = answer.replace(',', '')

        prompt = self.TASK_PROMPT.format(
            tool_names=", ".join(self.get_tool_names()),
            tool_desc=self.get_tool_descriptions(),
        )
        prompt += "\nQuestion: " + problem + \
            "\nYour Thought (with one action):[/INST]\n"

        for i in range(1, 12):
            invalid_response_counter = 0
            if to_print:
                print("Round {} prompt: {}".format(i, prompt))
                print("=====================================")
            config = {}
            response = self.completion_with_backoff(
                messages=[{"content": prompt, "role": "user"}],
                **config,
            )
            try:
                if to_print:
                    content = response.choices[0].message.content
                    tool_call = response.choices[0].message.tool_calls
                    print("Response from gpt:\n", content)
                    print("Suggested tool call:\n", tool_call)
                    print("-------------------------------------")
                response = parse_text(response)
            except Exception as e:
                print(e)
                response = None

            if response is None:
                invalid_response_counter += 1
                if invalid_response_counter > 3:
                    response = {"final_answer": "I don't know",
                                "thought": "Too many invalid responses."}
                    break
                continue

            if "final_answer" in response:
                prompt += "Thought: " + \
                    response["thought"] + "\nFINAL ANSWER: " + \
                    response["final_answer"] + "\n"
                break
            else:
                exec_result = self.generate_tool_response(response)
                if to_print:
                    print("Execution result: ", exec_result)
                    print("-------------------------------------")
            prompt += response["thought"] + f"\nAction {i}: " + response["action"] + f"\nAction {i} Input: " + response[
                "input"] + f"\nObservation {i}: " + exec_result + "\nYour Thought (with one action):"

        final_answer = None
        if response is not None:
            final_answer = normalize_answer(response.get(
                "final_answer", "Out of iteration limit."))
            if not isinstance(final_answer, str):
                final_answer = str(final_answer)
            if "." in final_answer:
                try:
                    final_answer = str(float(final_answer))
                except Exception:
                    final_answer = str(final_answer)
            if final_answer.endswith('.0'):
                final_answer = final_answer[:-2]
            if "/" in final_answer:
                final_answer = final_answer.split('/')
                try:
                    final_answer = str(
                        int(final_answer[0]) / int(final_answer[1]))
                    final_answer = str(float(final_answer))
                    if final_answer.endswith('.0'):
                        final_answer = final_answer[:-2]
                except Exception:
                    final_answer = normalize_answer(response.get(
                        "final_answer", "Out of iteration limit."))
            final_answer = final_answer.replace(',', '')
            final_answer = final_answer.replace('"', '')
            if "frac" in final_answer:
                final_answer = final_answer.replace("}{", "/")
                final_answer = final_answer.replace("\\frac", "")
                final_answer = final_answer.replace("{", "")
                final_answer = final_answer.replace("}", "")
                final_answer = final_answer.split('/')
                try:
                    final_answer = str(int(answer[0]) / int(answer[1]))
                except Exception:
                    pass

        correct = int(final_answer == normalize_answer(answer))
        print("-" * 25 + "Final Answer" + "-" * 25)
        print("Final answer: ", final_answer,
              "\nExpected answer: ", answer, "\nCorrect: ", correct)
        print("-" * (50 + len("Final Answer")))
        return {"is_correct": correct}, prompt
