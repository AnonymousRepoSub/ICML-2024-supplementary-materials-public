import autogen
from autogen.code_utils import execute_code
from utils import retry_with_exponential_backoff
import json
import copy


class AgentOptimizer:
    OPT_PROMPT = """You are a function optimizer. Your task is to maintain a list of functions for the assistant according to the existing function list and conversation history that happens between the assistant and the user. 
You can perform one of the following four actions to manipulate the function list using the functions you have: 
1. Revise one existing function (using revise_function). 
2. Remove one existing function (using remove_function).
3. Add one new function (using add_function).
4. Directly return "TERMINATE" to me if no more actions are needed for the current function list.

Below are the principles that you need to follow for taking these four actions.
(1) Revise one existing tool function:
1. Pay more attention to the failed tasks and corresponding error information, and optimize the function used in these tasks according to the conversation history if needed.
2. A failed function call can occur due to incorrect input arguments (missing arguments) or an incorrect function code implementation. You should focus more on the function code implementation and make it easy to get success function call.
3. Do not revise the function that you think works well and plays a critical role in solving the problems according to the conversation history. Only making revisions if needed.
4. Sometimes, a NameError may occur. To fix this error, you can either revise the name of the function in the code implementation or revise the name of the function call to make these two names consistent.

(2) Remove one existing tool function:
1. Only remove the function that you think is not needed anymore in future tasks. 

(3) Add one new tool function:
1. The added function should be general and easy enough to be used in future tasks, with no complex parameters or high level education requirement.
2. The added new function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable.
3. You must provide an input example of each argument in the argument's description.
4. Do not use dict, object, or array as the argument type. The argument type should be string, integer, or boolean.
5. The function name should be as sample as possible, with a clear description.
6. IMPORTANT: Prevent complex input. For example, you should not let user provide a whole table as input for the function.
For example, here is a function that can solve some math problems:
{{
    "name": "PerformArithmeticOperations",
    "description": "Perform various arithmetic operations on a list of numbers.",
    "arguments": {{
        "numbers": {{
            "type": "string",
            "description": "The list of numbers represented as a comma-separated string. INPUT EXAMPLE: '1,2,3,4,5'"
        }},
        "operation": {{
            "type": "string",
            "description": "The type of arithmetic operation to perform like 'sum', 'product', 'mean', 'max', 'min', 'range'. INPUT EXAMPLE: 'mean'",
            "enum": [
                "sum",
                "product",
                "mean",
                "max",
                "min",
                "range"
            ]
        }}
    }},
    "packages": "statistics",
    "code": "import statistics\n\ndef perform_arithmetic_operations(numbers, operation):\n    numbers_list = [float(num.strip()) for num in numbers.split(',')]\n    result = {{'sum': sum(numbers_list)}}\n    result['product'] = 1\n    for num in numbers_list:\n        result['product'] *= num\n    result['mean'] = statistics.mean(numbers_list)\n    result['max'] = max(numbers_list)\n    result['min'] = min(numbers_list)\n    if len(numbers_list) > 1:\n        result['range'] = max(numbers_list) - min(numbers_list)\n    else:\n        result['range'] = 0\n    return result.get(operation, 'Invalid operation')"
}}

(4) Directly return "TERMINATE":
If you think there is no need to perform any other actions for the current function list since the current list is optimal more actions will harm the performance in future tasks. Please directly reply to me with "TERMINATE".

One function signature includes the following five elements:
1. Function name 
2. Function description 
3. JSON schema of arguments encoded as a string
4. A list of package names imported by the function packages 
5. The code implementation

Below are the signatures of the current functions:
List A: {signiture}. 
The success rate (performance) with this function list is {success_rate}.

The following list are the function signatures that you have after taking {actions_num} actions in our previous conversations:
List B: {after_signiture}. 

The following list are the functions that failed in previous conversations:
(you cannot remove the function below, they are history records)
{historical_fail_functions}

Here are {conversation_num} conversation histories of solving {conversation_num} tasks. 
You should analysis and summarize the error made by assistant and function call, think about their common ground, and think about how to minimize the error and maximize the efficiency of the "assistant".
Sometimes even the answer is correct, the function call may fail many times. You should also think about how to make the function call success more easily.
You should also notice the relation between the problem and the function made by assistant, which is also the key to improve the function.
History:
{history}
The following table shows the statistical information for solving each task in each conversation and indicates whether each task was successfully solved. 
is_correct: 1 represents correct. is_correct: 0 represents wrong.
statistics: {statistic}

According to the above information, please take one of four actions to manipulate list B using the functions you know to minimize the effort of the assistant. The new functions should help help assistant to generate the patch and pass tests more accurately and efficiently.
Instead of returning TERMINATE directly or taking no action, you should try your best to optimize the function list. Only take no action if you really think the current list is optimal, as more actions will harm performance in future tasks. 
Even adding a general function that can substitute the assistant’s repeated suggestions of Python code with the same functionality could also be helpful.
"""

    ADD_FUNC = {
        "type": "function",
        "function": {
            "name": "add_function",
            "description": "Add a function in the context of the conversation. Necessary Python packages must be declared. The name of the function MUST be the same with the function name in the code you generated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function in the code implementation."
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the function."
                    },
                    "arguments": {
                        "type": "string",
                        "description": "JSON schema of arguments encoded as a string. Please note that the JSON schema only supports specific types including string, integer, boolean. (do not have float type) For example: { \"url\": { \"type\": \"string\", \"description\": \"The URL\", }}."
                    },
                    "packages": {
                        "type": "string",
                        "description": "A list of package names imported by the function, and that need to be installed with pip prior to invoking the function. This solves ModuleNotFoundError. It should be string, not list."
                    },
                    "required": {
                        "type": "string",
                        "description": "A list of required arguments. For example: [\"numbers\", \"operation\"]."
                    },
                    "code": {
                        "type": "string",
                        "description": "The implementation in Python."
                    }
                },
                "required": ["name", "description", "arguments", "packages", "required", "code"]
            }
        }
    }

    REVISE_FUNC = {
        "type": "function",
        "function": {
            "name": "revise_function",
            "description": "Revise a function in the context of the conversation. Necessary Python packages must be declared. The name of the function MUST be the same with the function name in the code you generated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function in the code implementation."
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the function."
                    },
                    "arguments": {
                        "type": "string",
                        "description": "JSON schema of arguments encoded as a string. Please note that the JSON schema only supports specific types including string, integer, boolean. (do not have float type) For example: { \"url\": { \"type\": \"string\", \"description\": \"The URL\", }}."
                    },
                    "packages": {
                        "type": "string",
                        "description": "A list of package names imported by the function, and that need to be installed with pip prior to invoking the function. This solves ModuleNotFoundError. It should be string, not list."
                    },
                    "required": {
                        "type": "string",
                        "description": "A list of required arguments. For example: [\"numbers\", \"operation\"]."
                    },
                    "code": {
                        "type": "string",
                        "description": "The implementation in Python. Do not include the function declaration."
                    }
                },
                "required": ["name", "description", "arguments", "packages", "required", "code"]
            }
        }
    }

    REMOVE_FUNC = {
        "type": "function",
        "function": {
            "name": "remove_function",
            "description": "Remove one function in the context of the conversation. Once remove one function, the assistant will not use this function in future conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function in the code implementation."
                    }
                },
                "required": ["name"]
            }
        }
    }

    def __init__(self, config_list, action_num=3, each_action_max_trials=10):
        self._action_num = action_num
        self._each_action_max_trials = each_action_max_trials
        self._client = autogen.OpenAIWrapper(config_list=config_list)
        self.model = "gpt-4-1106"

        self.checkpoint_mathproxyagent = None
        self.checkpoint_assistant = None
        self.checkpoint_history_list = None
        self.checkpoint_statistic_list = None
        self.checkpoint_function_json = None

        self.historical_fails = []

    def save_fail_function(self, cur_performance, fail_func):
        self.historical_fails.append({"performance": cur_performance, "fail_func": fail_func})
        self.historical_fails = sorted(self.historical_fails, key=lambda x: x["performance"])

    def clear_fail_function(self):
        self.historical_fails = []

    def _val_json(self, actions):
        logs = ""
        if actions is None:
            return True, logs
        else:
            for action in actions:
                function_args = action.function.arguments
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(f'{function_args}')
                        if 'arguments' in function_args.keys():
                            json.loads(f'{function_args.get("arguments")}')
                    except Exception as e:
                        print("JSON is invalid:", e)
                        logs = f"JSON is invalid: {e}"
                        return False, logs
                else:
                    if 'arguments' in function_args.keys():
                        json.loads(function_args.get("arguments"))
        return True, logs

    def _add_historical_functions(self):

        historical_prompt = "We also provide more examples for different functions and their corresponding success rates.\n The following function signatures are arranged in are arranged in ascending order based on their success rates, where higher success rates indicate better quality."
        historical_prompt += "\n\n"
        if len(self.historical_fails) > 0:
            for item in self.historical_fails:
                historical_prompt += "Function: \n" + str(item["fail_func"]) + "\n"
                historical_prompt += "Performance: \n" + str(item["performance"]) + "\n"
            historical_prompt += "\n\n"
            return historical_prompt
        else:
            return None

    def _val_remove(self, actions, after_signiture):
        logs = ""
        if actions is None:
            return True, logs
        else:
            for action in actions:
                action_name = action.function.name
                if action_name == "remove_function":
                    function_args = json.loads(action.function.arguments.strip('"'))
                    if function_args.get("name") not in [item["name"] for item in after_signiture]:
                        print("The function you want to remove does not exist.")
                        logs = f"Action Error: the function you want to remove does not exist. Function info: {actions}"
                        return False, logs
            return True, logs

    def _val_syntax(self, actions):
        logs = ""
        if actions is None:
            return True, logs
        else:
            for action in actions:
                if action.function.name != "remove_function":
                    try:
                        function_args = json.loads(action.function.arguments.strip('"'))
                    except Exception as e:
                        return False, f"JSON is invalid: {e}"
                    code = function_args.get("code")
                    try:
                        compile(code, '', 'exec')
                        print("successfully compiled")
                    except Exception as e:
                        print("Syntax is invalid:", e)
                        logs = f" Your previous function is {action}. Syntax is invalid: {e}."
                        return False, logs
            return True, logs

    def _val_argument(self, actions):
        for action in actions:
            if action.function.name != "remove_function":
                function_args = json.loads(action.function.arguments.strip('"'))
                arguments = json.loads(function_args['arguments'])
                for value in arguments.values():
                    if value["type"] != "string":
                        print("The argument must be string.")
                        return False
        return True

    def _format_actions(self, actions):
        ans = []
        for action in actions:
            func = json.loads(action.function.arguments.strip('"'))
            func["action_name"] = action.function.name

            if func.get("action_name") == "remove_function":
                item = {
                    "action_name": func.get("action_name"),
                    "name": func.get("name"),
                }
            else:
                item = {
                    "action_name": func.get("action_name"),
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "arguments": json.loads(func.get("arguments").strip('"')),
                    "packages": func.get("packages"),
                    "code": func.get("code"),
                }
            ans.append(item)
        return ans

    def _get_success_rate(self, statistic):
        sum = 0
        for key, value in statistic.items():
            if "is_correct" not in value.keys():
                statistic[key]["is_correct"] = 0
        for key, value in statistic.items():
            sum += value["is_correct"]
        if len(statistic.keys()) != 0:
            success_rate = sum / len(statistic.keys())
        else:
            success_rate = None
        return success_rate, statistic

    def _modify_function_signiture(self, cur_functions, action_json):
        for action in action_json:
            action_name = action.get("action_name")
            if action_name != "remove_function":
                cur_functions = [
                    item for item in cur_functions if item["name"] != action.get("name")]
                cur_functions.append(
                    {"name": action.get("name"),
                     "description": action.get("description"),
                     "arguments": action.get("arguments"),
                     "packages": action.get("packages"),
                     "code": action.get("code")}
                )
            else:
                cur_functions = [
                    item for item in cur_functions if item["name"] != action.get("name")]
        return cur_functions

    def make_checkpoint(self, history_list, statistic_list, function_json):

        self.checkpoint_history_list = copy.deepcopy(history_list)
        self.checkpoint_statistic_list = copy.deepcopy(statistic_list)
        self.checkpoint_function_json = copy.deepcopy(function_json)

    def roll_back(self, mathproxyagent):

        self.update_function_call_forall(self.checkpoint_function_json, mathproxyagent)
        return self.checkpoint_history_list, self.checkpoint_statistic_list, self.checkpoint_function_json

    def update_function_call_forall(self, function_json, mathproxyagent):

        for name in list(mathproxyagent._function_map.keys()):
            del mathproxyagent._function_map[name]
        mathproxyagent.functions = []
        for func_json in function_json:
            action = {
                "action_name": "add_function",
                "name": func_json.get("name"),
                "description": func_json.get("description"),
                "arguments": func_json.get("arguments"),
                "packages": func_json.get("packages"),
                "code": func_json.get("code"),
            }
            self.update_function_call(action, mathproxyagent)

    def update_function_call(self, action, agent):
        def execute_func(name, packages, code, args):
            if "," in packages:
                packages = packages.replace(",", "\", \"")
            pip_install = (
                f"""print("Installing package: {packages}")\nsubprocess.run(["pip", "-qq", "install", "{packages}"])"""
                if packages
                else ""
            )
            str = f"""
import subprocess
{pip_install}
print("Result of {name} function execution:")
{code}
args={args}
result={name}(**args)
if result is not None: print(result)
"""
            print(f"execute_code:\n{str}")
            result = execute_code(str, use_docker=False)  
            if result[0] != 0:
                raise Exception("Error in executing function:" + result[1])
            print(f"Result: {result[1]}")
            return result[1]

        name, description, arguments, packages, code, action_name = action.get("name"), action.get(
            "description"), action.get("arguments"), action.get("packages"), action.get("code"), action.get(
            "action_name")

        if name in agent._function_map.keys():
            del agent._function_map[name]
        if action_name != "remove_function":
            function_config = {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": arguments},
            }
            agent.register_function(function_map={name: lambda x: execute_func(name, packages, code, x)})
            agent.update_function_signature(function_config, is_remove=False)
        else:
            agent.update_function_signature(name, is_remove=True)

    @retry_with_exponential_backoff
    def completion_with_backoff(self, **kwargs):
        return self._client.create(**kwargs, cache_seed=None)

    def step(self, history, statistic, func_signiture):
        action_return = []
        origin_signiture = func_signiture
        modified_signiture = origin_signiture

        success_rate, statistic = self._get_success_rate(statistic)  
        for action_index in range(self._action_num):
            historical_fail_functions = self._add_historical_functions()
            if historical_fail_functions is None:
                historical_fail_functions = ""
            prompt = self.OPT_PROMPT.format(
                conversation_num=len(history),
                statistic=statistic,
                signiture=origin_signiture,
                history=history,
                success_rate=success_rate,
                actions_num=action_index,
                after_signiture=modified_signiture,
                historical_fail_functions=historical_fail_functions,

            )
            messages = [{"role": "user", "content": prompt}]
            for _ in range(self._each_action_max_trials):
                response = self.completion_with_backoff(
                    model=self.model,
                    messages=messages,
                    tools=[self.ADD_FUNC, self.REVISE_FUNC, self.REMOVE_FUNC],
                    tool_choice="auto"
                )
                actions = response.choices[0].message.tool_calls
                is_json_valid, logs_json_valid = self._val_json(actions)
                is_syntax_valid, logs_syntax_valid = self._val_syntax(actions)
                is_remove_valid, logs_remove_valid = self._val_remove(actions, modified_signiture)
                if is_json_valid and is_syntax_valid and is_remove_valid:
                    break
            if actions is not None:
                action_result = self._format_actions(actions)
                action_return = action_return + action_result
                modified_signiture = self._modify_function_signiture(modified_signiture, action_result)
        return action_return, modified_signiture
