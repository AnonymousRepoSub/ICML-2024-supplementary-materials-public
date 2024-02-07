from Agent_GPT import TabMWPAgentProxy
from Optimizer_GPT import AgentOptimizer
import os
import autogen
import argparse
from datasets import load_dataset
from utils import record_result


def is_termination_msg(msg):
    if msg['content'] is None:
        return False
    return "TERMINATE" in msg['content']

def train(args):
    dataset = load_dataset("json", data_files={
        'test': f"{args.dataset_name_or_path}/problems_test_selected_hard.json",
        'train': f"{args.dataset_name_or_path}/problems_train_selected_hard.json"
    })
    train_data, test_data = dataset["train"], dataset["test"]
    config_list_gpt_4 = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        file_location="",  
        filter_dict={"model": ["gpt-4-1106"]}
    )
    config_list_gpt_35 = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        file_location="",  
        filter_dict={"model": ["mistralai/Mixtral-8x7B-Instruct-v0.1"]}
    )
    if args.remove_python:
        use_py = False
    else:
        use_py = True

    user_proxy = TabMWPAgentProxy(
        name="math_problem_solving_assistant",
        code_execution_config={
            "work_dir": "_output",
            "use_docker": False,
        },
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        use_py=True
    )
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="Think your self as a math student. You need to follow the instructions to solve the problem.",
        llm_config={
            "timeout": 20,
            "seed": 42,
            "temperature": 0.05,
            "config_list": config_list_gpt_35,
            "cache_seed": None,
        }
    )

    agent_optimizer = AgentOptimizer(config_list=config_list_gpt_4, dataset_name="tabmwp", run_test=args.run_test)

    pre_performance, cur_performance, epoch, max_epoch, fail_stuck = -1, -1, 0, 10, 10
    history_list, statistic_list, function_json, actions ={}, {}, [], []
    while epoch < max_epoch:
        agent_optimizer.make_checkpoint(history_list, statistic_list, function_json)
        if args.result_dir is None:
            result_dir = "./result"
        else:
            result_dir = args.result_dir
        folder_path = os.path.join(result_dir, "epoch_{epoch}".format(epoch=str(epoch)))
        if len(history_list) != 0:
            actions, function_json = agent_optimizer.step(history_list, statistic_list, function_json)
            for action in actions:
                agent_optimizer.update_function_call(action, proxyagent=user_proxy, assistant=assistant)

        history_list, statistic_list = {}, {}
        user_proxy.clear_function_statistic()
        for index, instance in enumerate(train_data.select(range(args.training_batch_size))):
            statistic, chat_history = user_proxy.initiate_chat(
                recipient=assistant,
                cache_seed=None,
                query = instance,
            )
            history_list["conversation: {index}".format(index=index + 1)] = chat_history
            statistic_list["conversation: {index}".format(index=index + 1)] = statistic

            print("statistic", statistic)

        sum = 0
        for key, value in statistic_list.items():
            if "is_correct" not in value.keys():
                statistic_list[key]["is_correct"] = 0
        for key, value in statistic_list.items():
            sum += value["is_correct"]
        cur_performance = sum / len(statistic_list.keys())

        if cur_performance < pre_performance and len(actions) != 0:
            record_result(history_list, statistic_list, function_json, folder_path, fail_train=True)
            fail_func = function_json
            _, statistic_list, function_json = agent_optimizer.roll_back(user_proxy, assistant)
            fail_stuck -= 1
            agent_optimizer.save_fail_function(cur_performance, fail_func)
        elif cur_performance < pre_performance and len(actions) == 0:
            record_result(history_list, statistic_list, function_json, folder_path, fail_train=True)
            fail_stuck -= 1
        else:
            fail_stuck = 10
            agent_optimizer.clear_fail_function()
            epoch += 1
            pre_performance = cur_performance
            record_result(history_list, statistic_list, function_json, folder_path)

            if epoch == 1 or epoch == max_epoch:
                test_function_json = []

                for func in function_json:
                    test_function_json.append(func)

                print("test_function_json", test_function_json)
                if "functions" in assistant.llm_config.keys():
                    print(assistant.llm_config["functions"])

                agent_optimizer.update_function_call_forall(test_function_json, proxyagent=user_proxy, assistant=assistant)
                if "functions" in assistant.llm_config.keys():
                    print(assistant.llm_config["functions"])
                test_history_list, test_statistic_list = {}, {}

                for index, instance in enumerate(test_data.select(range(100))):
                    user_proxy.clear_function_statistic()
                    test_single_statistic, test_chat_history = user_proxy.initiate_chat(
                        recipient=assistant,
                        cache_seed=None,
                        query=instance,
                        answer=instance['answer'],
                    )
                    test_history_list["conversation: {index}".format(index=index + 1)] = test_chat_history
                    test_statistic_list["conversation: {index}".format(index=index + 1)] = test_single_statistic

                test_folder_path = os.path.join(folder_path, "test")
                record_result(test_history_list,
                              test_statistic_list,
                              function_json,
                              test_folder_path,
                              fail_train=False,
                              test_function_json=test_function_json)

            if "functions" in assistant.llm_config.keys():
                print(assistant.llm_config["functions"])

            agent_optimizer.update_function_call_forall(function_json, proxyagent=user_proxy, assistant=assistant)

            if "functions" in assistant.llm_config.keys():
                print(assistant.llm_config["functions"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--remove_python", type=bool, default=False)
    parser.add_argument("--code_dir", type=str, default="_output")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--run_test", type=bool, default=True)
    parser.add_argument("--training_batch_size", type=int, default=30)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="HuggingFace dataset name or local path",
        default="dataset"
    )
    args = parser.parse_args()
    train(args)

