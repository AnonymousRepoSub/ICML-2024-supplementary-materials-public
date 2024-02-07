from autogen.math_utils import get_answer
from autogen.code_utils import extract_code
import json
import os
import random


def is_termination_msg_mathchat(message):
    """Check if a message is a termination message."""
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


def load_jsonl(data_dir):
    data_list = []
    with open(data_dir, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def read_txt(data_dir):
    with open(data_dir) as f:
        return f.read()


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
    final_summary["correct_rate"] = sum/len(statistic_list.keys())

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

    if args.debug:
        with open("MATH/dataset/algebra.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))
        with open("MATH/dataset/train/algebra.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))
        test_data, train_data = test_data[0:1], train_data[0:1]
        return train_data, test_data
    elif args.batch_train:
        with open("MATH/dataset/intermediate_algebra.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))
        with open("MATH/dataset/train/intermediate_algebra.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))
        test_data, train_data = test_data[0:3], train_data[0:2]
        return train_data, test_data
    else:
        if args.task == "math" and args.math_type == "all_types":
            pro_types = [pro_type for pro_type in os.listdir("MATH/dataset/train") if pro_type.endswith(".jsonl")]
            for pro_type in pro_types:
                with open(os.path.join("MATH/dataset/train", pro_type), "r", encoding="utf-8") as f:
                    tem_train = []
                    for line in f:
                        tem_train.append(json.loads(line))
                    train_data = train_data + random.sample(tem_train, 3)
                with open(os.path.join("MATH/dataset", pro_type), "r", encoding="utf-8") as f:
                    tem_test = []
                    for line in f:
                        tem_test.append(json.loads(line))
                    test_data = test_data + random.sample(tem_test, 8)
            return train_data, test_data

        else:
            with open(os.path.join("MATH/dataset", args.math_type + ".jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(json.loads(line))
            with open(os.path.join("MATH/dataset/train", args.math_type + ".jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    train_data.append(json.loads(line))
            test_data, train_data = test_data[0:80], train_data[0:20]
            return train_data, test_data

def get_ood_math(args):

    test_data, train_data = [], []
    with open(os.path.join("MATH/dataset", args.test_data + ".jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    with open(os.path.join("MATH/dataset/train", args.train_data + ".jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    test_data, train_data = test_data[0:80], train_data[0:20]
    return train_data, test_data
