from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional, Tuple, Union
from autogen.math_utils import get_answer
from hashlib import md5
import re
import os
import pathlib
import subprocess
import sys
import shutil
import time
import tiktoken
import openai
import random
import logging
import json


try:
    import docker
except ImportError:
    docker = None

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600
WIN32 = sys.platform == "win32"
WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extensions")
TIMEOUT_MSG = "Timeout"
PATH_SEPARATOR = WIN32 and "\\" or "/"
UNSUPPORTED_TOOLS = ["video"]

def normalize_answer(a: str):
    ans = re.sub(r"[\.\!\?]+$", "", re.sub(r"\s+", " ", a.strip())).strip()
    return ans

def _cmd(lang):
    if lang.startswith("python") or lang in ["bash", "sh", "powershell"]:
        return lang
    if lang in ["shell"]:
        return "sh"
    if lang in ["ps1"]:
        return "powershell"
    raise NotImplementedError(f"{lang} not recognized in code execution")

def execute_code(
    code: Optional[str] = None,
    timeout: Optional[int] = None,
    filename: Optional[str] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[Union[List[str], str, bool]] = None,
    lang: Optional[str] = "python",
) -> Tuple[int, str, str]:
    if all((code is None, filename is None)):
        error_msg = f"Either {code=} or {filename=} must be provided."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    if use_docker is None:
        if docker is None:
            use_docker = False
            logger.warning(
                "execute_code was called without specifying a value for use_docker. Since the python docker package is not available, code will be run natively. Note: this fallback behavior is subject to change"
            )
        else:
            use_docker = True

    timeout = timeout or DEFAULT_TIMEOUT
    original_filename = filename
    if WIN32 and lang in ["sh", "shell"] and (not use_docker):
        lang = "ps1"
    if filename is None:
        code_hash = md5(code.encode()).hexdigest()
        filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
    if work_dir is None:
        work_dir = WORKING_DIR
    filepath = os.path.join(work_dir, filename)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)
    if code is not None:
        with open(filepath, "w", encoding="utf-8") as fout:
            fout.write(code)
    in_docker_container = os.path.exists("/.dockerenv")
    if not use_docker or in_docker_container:
        cmd = [
            sys.executable if lang.startswith("python") else _cmd(lang),
            f".\\{filename}" if WIN32 else filename,
        ]
        if WIN32:
            logger.warning("SIGALRM is not supported on Windows. No timeout will be enforced.")
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    subprocess.run,
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                try:
                    result = future.result(timeout=timeout)
                except TimeoutError:
                    if original_filename is None:
                        os.remove(filepath)
                    return 1, TIMEOUT_MSG, None
        if original_filename is None:
            os.remove(filepath)
        if result.returncode:
            logs = result.stderr
            if original_filename is None:
                abs_path = str(pathlib.Path(filepath).absolute())
                logs = logs.replace(str(abs_path), "").replace(filename, "")
            else:
                abs_path = str(pathlib.Path(work_dir).absolute()) + PATH_SEPARATOR
                logs = logs.replace(str(abs_path), "")
        else:
            logs = result.stdout
        return result.returncode, logs, None

    client = docker.from_env()
    image_list = (
        ["python:3-alpine", "python:3", "python:3-windowsservercore"]
        if use_docker is True
        else [use_docker]
        if isinstance(use_docker, str)
        else use_docker
    )
    for image in image_list:
        try:
            client.images.get(image)
            break
        except docker.errors.ImageNotFound:
            print("Pulling image", image)
            try:
                client.images.pull(image)
                break
            except docker.errors.DockerException:
                print("Failed to pull image", image)
    exit_code_str = f"exitcode{time.time()}"
    abs_path = pathlib.Path(work_dir).absolute()
    cmd = [
        "sh",
        "-c",
        f"{_cmd(lang)} {filename}; exit_code=$?; echo -n {exit_code_str}; echo -n $exit_code; echo {exit_code_str}",
    ]
    container = client.containers.run(
        image,
        command=cmd,
        working_dir="/workspace",
        detach=True,
        volumes={abs_path: {"bind": "/workspace", "mode": "rw"}},
    )
    start_time = time.time()
    while container.status != "exited" and time.time() - start_time < timeout:
        container.reload()
    if container.status != "exited":
        container.stop()
        container.remove()
        if original_filename is None:
            os.remove(filepath)
        return 1, TIMEOUT_MSG, image
    logs = container.logs().decode("utf-8").rstrip()
    tag = filename.replace("/", "")
    container.commit(repository="python", tag=tag)
    container.remove()
    exit_code = container.attrs["State"]["ExitCode"]
    if exit_code == 0:
        pattern = re.compile(f"{exit_code_str}(\\d+){exit_code_str}")
        match = pattern.search(logs)
        exit_code = 1 if match is None else int(match.group(1))
        logs = logs if match is None else pattern.sub("", logs)

    if original_filename is None:
        os.remove(filepath)
    if exit_code:
        logs = logs.replace(f"/workspace/{filename if original_filename is None else ''}", "")
    return exit_code, logs, f"python:{tag}"

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 8,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                print("<error>", e, "</error>")
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= min(exponential_base * (1 + jitter * random.random()), max_delay)

                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper

def content_str(content: Union[str, List, None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"content must be None, str, or list, but got {type(content)}")

    rst = ""
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("Wrong content format: every element should be dict if the content is a list.")
        assert "type" in item, "Wrong content format. Missing 'type' key in content's dict."
        if item["type"] == "text":
            rst += item["text"]
        elif item["type"] == "image_url":
            rst += "<image>"
        else:
            raise ValueError(f"Wrong content format: unknown type {item['type']} within the content")
    return rst


CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
def extract_code(
    text: Union[str, List], pattern: str = CODE_BLOCK_PATTERN, detect_single_line_code: bool = False
) -> List[Tuple[str, str]]: 
    text = content_str(text)
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [("unknown", text)]

    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    return extracted


def record_result(history_list, statistic_list, function_json, folder_path, fail_train=False, test_function_json = None):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if fail_train == True:
        existing_folders = [folder for folder in os.listdir(folder_path) if folder.startswith("fail_")]
        if not existing_folders:
            new_folder_name = "fail_1"
        else:
            existing_numbers = [int(folder.split("_")[1]) for folder in existing_folders]
            max_number = max(existing_numbers)
            new_folder_number = max_number + 1
            new_folder_name = f"fail_{new_folder_number}"
        folder_path = os.path.join(folder_path, new_folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    history_save_path = os.path.join(folder_path, "history.json")
    statistic_save_path = os.path.join(folder_path, "statistic.json")
    function_json_save_path = os.path.join(folder_path, "function.json")
    final_summary_save_path = os.path.join(folder_path, "summary.json")
    if test_function_json is not None:
        test_function_json_save_path = os.path.join(folder_path, "test_function.json")
    final_summary = {}
    sum = 0
    for key, value in statistic_list.items():
        if "is_correct" not in value.keys():
            statistic_list[key]["is_correct"] = 0
    for key, value in statistic_list.items():
        sum += value["is_correct"]
    if len(statistic_list.keys()) != 0:
        final_summary["correct_rate"] = sum/len(statistic_list.keys())
    else:
        final_summary["correct_rate"] = 0

    with open(history_save_path, 'w') as file:
        json.dump(history_list, file, indent=4)

    with open(statistic_save_path, 'w') as file:
        json.dump(statistic_list, file, indent=4)
    if function_json is not None:
        with open(function_json_save_path, 'w') as file:
            json.dump(function_json, file, indent=4)
    with open(final_summary_save_path, 'w') as file:
        json.dump(final_summary, file, indent=4)
    
    if test_function_json is not None:
        with open(test_function_json_save_path, 'w') as file:
            json.dump(test_function_json, file, indent=4)


def get_train_test(args):
    test_data, train_data = [], []
    with open(os.path.join("MATH/dataset", args.math_type + ".jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    with open(os.path.join("MATH/dataset/train", args.math_type + ".jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    test_data, train_data = test_data[0:80], train_data[0:20]
    return train_data, test_data

def filter_gaia_tasks(tasks):
    capable_tasks = []
    for task in tasks:
        for tool in UNSUPPORTED_TOOLS:
            if tool in task['Annotator Metadata']['Tools'].lower() or task['Level'] == 3:
                break
        else:
            capable_tasks.append(task)
    return capable_tasks

def is_termination_msg_gaia(message):
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == "python" or c[0] == "wolfram":
            contain_code = True
            break
    if message.rstrip().find("FINAL ANSWER") >= 0:
        return True
    if "TERMINATE" in message:
        return True

    return (
        not contain_code
        and get_answer_gaia(message) is not None
        and get_answer_gaia(message) != ""
    )

def get_answer_gaia(string: str):
    idx = string.rfind("FINAL ANSWER:")
    if idx < 0:
        return None
    return normalize_answer(string[idx + len("FINAL ANSWER:"):].strip())

def read_txt(data_dir):
    with open(data_dir) as f:
        return f.read()

def is_termination_msg_mathchat(message):
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == "python" or c[0] == "wolfram":
            contain_code = True
            break
    if message.rstrip().find("TERMINATE") >= 0:
        return True

    return (
        not contain_code
        and get_answer(message) is not None
        and get_answer(message) != ""
    )

def prepare_file(file, val_path="GAIA/2023/validation", work_dir="coding"):
    if isinstance(file, list):
        for f in file:
            shutil.copy(os.path.join(val_path, f), work_dir)
    elif isinstance(file, str):
        if file:
            shutil.copy(os.path.join(val_path, file), work_dir)
    else:
        raise Exception("Wrong file format")

def post_process_file(work_dir="coding"):
    for root, dirs, files in os.walk(work_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def get_env_variables(env_file="env"):
    with open(env_file, "r") as f:
        envs = json.load(f)
    for key, value in envs.items():
        os.environ[key] = value
    return envs

def trim_string(string: str, limit=5000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(string)
    if len(tokens) <= limit:
        return string
    else:
        return encoding.decode(tokens[:limit//2]) + f"\n...The log is too long, you should print less than {limit} tokens...\n" + encoding.decode(tokens[-limit//2:])