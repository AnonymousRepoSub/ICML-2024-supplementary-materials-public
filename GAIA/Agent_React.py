from utils import retry_with_exponential_backoff, normalize_answer, get_env_variables, extract_code, trim_string
from openai import AzureOpenAI
from functools import partial
from autogen.code_utils import execute_code, extract_code
import os
from typing import Dict, Callable
import shutil
import re

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

def parse_text(text: str):
    if "terminate" in text.lower():
        return {"final_answer": "I don't know", "thought": "TERMINATE"}

    if not text.startswith("Thought:"):
        text = "Thought: " + text
    obs_match =  re.search(r"Observation\s*\d*\s*:", text)
    if obs_match:
        text = text[:obs_match.start()]

    pattern1 = r"Thought\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input:[\s]*(.*)"
    pattern2 = r"FINAL ANSWER:(.*)"

    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        return {
            "thought": match2.group(1).strip(),
            "final_answer": match2.group(1).strip()
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
    TASK_PROMPT = """Answer the following question using your coding skills. Below is a list of the tools you can use and their detailed descriptions:
{tool_desc}
You should always follow the below template, when you respond you should provide one <Thought, Action, Action Input> triplet and wait for observation before proceeding to the next round, unless you have reached a FINAL ANSWER.

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

TEMPLATE:
Question: the input question you must answer
Thought: your reasoning about the current situation
Action 1: the action to take, should be one of [{tool_names}]
Action 1 Input: the arguments passed to action 1
Observation 1: the result of action 1
Action 2: the action to take, should be one of [{tool_names}]
Action 2 Input: the input to action 2
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
FINAL ANSWER: the final answer to the original input question
"""
    
    def __init__(self, azure_key, azure_endpoint, use_docker=False, work_dir="coding_react"):
        self.client = AzureOpenAI(
            api_key=azure_key,  
            api_version="2023-08-01-preview",
            azure_endpoint=azure_endpoint
        )
        self.use_docker = use_docker
        self.functions = []
        self.unsupported_tools = ["video"]
        self.work_dir = work_dir
        self._function_map = {}
        self.function_statistic = None
        self.max_function_call_trial = 3
        self.logs = {}
        self.base_tools = {
            "Python": partial(extract_and_execute_code,  lang="python", work_dir=work_dir, use_docker=use_docker),
            "Shell": partial(extract_and_execute_code, lang="shell", work_dir=work_dir, use_docker=use_docker)
        }
    
    def register_function(self, function_map: Dict[str, Callable]):
        self._function_map.update(function_map)

    @retry_with_exponential_backoff
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

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
            self.functions = [
                func for func in self.functions if func.get("name") != func_sig["name"]
            ] + [func_sig]

    def get_tool_descriptions(self):
        envs = get_env_variables()
        tool_descs = f"(1) Python. Executes the exact python code you provide. You should always use 'print' function for the output.\nThe python code working directory is {os.path.abspath(self.work_dir)}, all the files (if any) are contained in this directory, you should always use **absolute path** to access any related files."
        if envs:
            tool_descs += " The following environment variables are available and can be accessed directly through os.environ: " + str(list(envs.keys()))
        tool_descs += "\nInput: Python code block, should begin with ```python\n"        
        tool_descs += "(2) Shell. Executes the exact shell command you provide.\nInput: Shell command block, should begin with ```shell\n"

        for i, function in enumerate(self.functions):
            desc = f"({i+3}) {function['name']}. {function['description']}\n"
            args_desc = list(function["parameters"]["properties"].values())[0]['description']
            desc += f"Input: {args_desc} It should be double-quoted all the time."
            tool_descs += desc
            tool_descs += "\n"
        return tool_descs

    def get_tool_names(self):
        tool_names = list(self.base_tools.keys())
        for function in self.functions:
            tool_names.append(function["name"])
        return tool_names

    def execute_function(self, func_call):
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)
        
        is_exec_success = False
        if func is not None:
            arguments = func_call.get("arguments", None)

            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                try:
                    content = func(arguments)
                    is_exec_success = True
                    content = "Execution succeeded. Tool output:\n" + content
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Tool {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def clear_function_statistic(self):
        self.function_statistic = None

    def generate_tool_response(self, tool_name, args):
        
        if tool_name in self.base_tools.keys():
            exitcode, logs, _ = self.base_tools[tool_name](args)
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            func_return = exitcode2str + " Code output:\n" + trim_string(logs)
            return func_return

        if tool_name not in self._function_map.keys():
            return "The action is not valid. It should be exactly one of the following tool names: " + ", ".join(self.get_tool_names())
        is_exec_success, func_return = self.execute_function({"name": tool_name, "arguments": args})

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

    def query(self, task, to_print=False):
        if self.not_capable(task):
            return {"is_correct": 0}, "Task not supported"
        filename = task['file_name']
        if len(filename) > 0:
            shutil.copy(os.path.join("GAIA/2023/validation", filename), self.work_dir)
            question = task['Question'] + " The path to the file you need is: " + os.path.abspath(os.path.join(self.work_dir, filename))
        else:
            question = task['Question']

        prompt = self.TASK_PROMPT.format(
            tool_names=", ".join(self.get_tool_names()),
            tool_desc=self.get_tool_descriptions(),
        )
        prompt += "\nQuestion: " + question + "\nThought: "

        for i in range(1, 21):
            invalid_response_counter = 0
            if to_print:
                print("Round {} prompt: {}".format(i, prompt))
                print("=====================================")
            response = self.completion_with_backoff(
                model="gpt-4-1106",
                messages=[{"content": prompt, "role": "user"}],
                top_p=0.95,
            ).choices[0].message.content
            if to_print:
                print("Response from gpt:\n", response)
                print("-------------------------------------")
            response = parse_text(response)

            if response is None:
                invalid_response_counter += 1
                if invalid_response_counter > 3:
                    response = {"final_answer": "I don't know", "thought": "Too many invalid responses."}
                    break
                continue

            if "final_answer" in response:
                prompt += "Thought: " + response["thought"] + "\nFINAL ANSWER: " + response["final_answer"] + "\n"
                break
            else:
                exec_result = self.generate_tool_response(response["action"], response["input"])
                if to_print:
                    print("Execution result: ", exec_result)
                    print("-------------------------------------")
            prompt += response["thought"] + f"\nAction {i}: " + response["action"] + f"\nAction {i} Input: " + response["input"] + f"\nObservation {i}: " + exec_result + "\nThought: "

        final_answer = normalize_answer(response.get("final_answer", "Out of iteration limit."))
        correct = int(final_answer == normalize_answer(task['Final answer']))

        print("-"*25 + "Final Answer" + "-"*25)
        print("Final answer: ", final_answer, "\nExpected answer: ", task['Final answer'], "\nCorrect: ", correct)
        print("-" * (50 + len("Final Answer")))
        return {"is_correct": correct}, prompt
    
    def resume(self, task, prompt, to_print=False):

        idx = prompt.rfind("Thought: \nFINAL ANSWER: \n")
        prompt = prompt[:idx]
        turn = find_largest_n(prompt)

        for i in range(turn+1, 21):
            invalid_response_counter = 0
            if to_print:
                print("Round {} prompt: {}".format(i, prompt))
                print("=====================================")
            response = self.completion_with_backoff(
                model="gpt-4-1106",
                messages=[{"content": prompt, "role": "user"}],
                top_p=0.95,
            ).choices[0].message.content
            if to_print:
                print("Response from gpt:\n", response)
                print("-------------------------------------")
            response = parse_text(response)

            if response is None:
                invalid_response_counter += 1
                if invalid_response_counter > 3:
                    response = {"final_answer": "I don't know", "thought": "Too many invalid responses."}
                    break
                continue

            if "final_answer" in response:
                prompt += "Thought: " + response["thought"] + "\nFINAL ANSWER: " + response["final_answer"] + "\n"
                break
            else:
                exec_result = self.generate_tool_response(response["action"], response["input"])
                if to_print:
                    print("Execution result: ", exec_result)
                    print("-------------------------------------")
            prompt += response["thought"] + f"\nAction {i}: " + response["action"] + f"\nAction {i} Input: " + response["input"] + f"\nObservation {i}: " + exec_result + "\nThought: "

        final_answer = normalize_answer(response.get("final_answer", "Out of iteration limit."))
        correct = int(final_answer == normalize_answer(task['Final answer']))

        print("-"*25 + "Final Answer" + "-"*25)
        print("Final answer: ", final_answer, "\nExpected answer: ", task['Final answer'], "\nCorrect: ", correct)
        print("-" * (50 + len("Final Answer")))
        return {"is_correct": correct}, prompt

def find_largest_n(s):
    string = s[s.rfind("Question:"): ]
    pattern = r'Observation (\d+)'

    numbers = [int(match) for match in re.findall(pattern, string)]

    return max(numbers) if numbers else 0