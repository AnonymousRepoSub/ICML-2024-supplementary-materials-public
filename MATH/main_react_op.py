from Optimizer_React import AgentOptimizer
from utils_react import record_result, get_train_test
from Agent_React import ReActAgent
import argparse
import json
import os
import datetime
import shutil


def main(args):
    current_time = datetime.datetime.now()
    train_data, test_data = get_train_test(args)
    print("Train dataset length: ", len(train_data))
    print("Test dataset length: ", len(test_data))
    agent = ReActAgent(
        azure_key="Your API",
        azure_endpoint="Your endpoint",
        use_docker=False,
        work_dir=os.path.join("./work_dir/", current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    )
    OAI_config = {
        "AZURE_OPENAI_API_KEY": "Your API Key",
        "api_version": "2023-12-01-preview",
        "azure_endpoint": "Your endpoint",
        "model": "gpt-4-1106",
    }
    agent_optimizer = AgentOptimizer(OAI_config=OAI_config)

    history_list, statistic_list, function_json, actions = {}, {}, [], []
    pre_performance, cur_performance, epoch, max_epoch, fail_stuck = -1, -1, 0, 10, 10

    if args.debug:
        train_data = [train_data[0]]
        test_data = [test_data[0]]
        max_epoch = 3
        fail_stuck = 1

    if args.debug:
        result_dir = "./result/debug"
    else:
        result_dir = os.path.join("./result", "reactopt_" + args.math_type)
        
    if os.path.isdir(result_dir) and len(os.listdir(result_dir)) != 0:
        max_x = -1
        for epoch_folder_name in os.listdir(result_dir):
            if epoch_folder_name.startswith('epoch_') and os.path.isdir(os.path.join(result_dir, epoch_folder_name)):
                if 'function.json' in os.listdir(os.path.join(result_dir, epoch_folder_name)):
                    x = int(epoch_folder_name.split('_')[1])
                    max_x = max(max_x, x)

        with open(os.path.join(result_dir, "epoch_" + str(max_x), "function.json")) as f:
            function_json = json.load(f)
        with open(os.path.join(result_dir, "epoch_" + str(max_x), "statistic.json")) as f:
            statistic_list = json.load(f)
        with open(os.path.join(result_dir, "epoch_" + str(max_x), "history.json")) as f:
            history_list = json.load(f)
        with open(os.path.join(result_dir, "epoch_" + str(max_x), "summary.json")) as f:
            pre_performance_json = json.load(f)
        pre_performance = pre_performance_json["correct_rate"]

        for folder_name in os.listdir(result_dir):
            if folder_name.startswith('epoch_') and os.path.isdir(os.path.join(result_dir, folder_name)):
                try:
                    y = int(folder_name.split('_')[1])
                    if y > max_x:
                        shutil.rmtree(os.path.join(result_dir, folder_name))
                        print(f"delete: {folder_name}")
                except ValueError:
                    pass

        epoch = max_x + 1

    while epoch < max_epoch and fail_stuck > 0:
        print("Starting Epoch {} at {}".format(epoch, datetime.datetime.now().strftime("%H:%M:%S")))
        print("Time elapsed: ", (datetime.datetime.now() - start_time).seconds, "(s)")
        agent_optimizer.make_checkpoint(history_list, statistic_list, function_json)
        folder_path = os.path.join(result_dir, "epoch_{epoch}".format(epoch=str(epoch)))
        if len(history_list) != 0:
            actions, function_json = agent_optimizer.step(
                history_list, statistic_list, function_json)
            for action in actions:
                agent_optimizer.update_function_call(
                    action, agent=agent)

        history_list, statistic_list = {}, {}
        agent.clear_function_statistic()
        for index, query in enumerate(train_data):
            single_statistic, chat_history = agent.query(query, True)
            history_list["conversation: {index}".format(
                index=index + 1)] = chat_history
            statistic_list["conversation: {index}".format(
                index=index + 1)] = single_statistic
            print("single_statistic", single_statistic)
        sum = 0
        for key, value in statistic_list.items():
            if "is_correct" not in value.keys():
                statistic_list[key]["is_correct"] = 0
        for key, value in statistic_list.items():
            sum += value["is_correct"]
        cur_performance = sum/len(statistic_list.keys())
        if cur_performance < pre_performance and len(actions) != 0:
            record_result(history_list, statistic_list,
                          function_json, folder_path, fail_train=True)
            fail_func = function_json
            history_list, statistic_list, function_json = agent_optimizer.roll_back(agent)
            fail_stuck -= 1
            agent_optimizer.save_fail_function(cur_performance, fail_func)
        elif cur_performance < pre_performance and len(actions) == 0:
            record_result(history_list, statistic_list,
                          function_json, folder_path, fail_train=True)
            fail_stuck -= 1
        else:
            fail_stuck = 10
            agent_optimizer.clear_fail_function()
            epoch += 1
            pre_performance = cur_performance
            record_result(history_list, statistic_list,
                          function_json, folder_path)

    function_statistic = agent.function_statistic
    test_function_json = []
    for func in function_json:
        func_name = func["name"]
        if func_name in function_statistic.keys() and function_statistic[func_name] == True:
            test_function_json.append(func)
    agent_optimizer.update_function_call_forall(
        test_function_json, mathproxyagent=agent)
    test_history_list, test_statistic_list = {}, {}
    for index, query in enumerate(test_data):
        test_single_statistic, test_chat_history = agent.query(query, True)
        test_history_list["conversation: {index}".format(
            index=index + 1)] = test_chat_history
        test_statistic_list["conversation: {index}".format(
            index=index + 1)] = test_single_statistic
    test_folder_path = os.path.join(folder_path, "test")
    record_result(test_history_list, test_statistic_list, function_json, test_folder_path,
                  fail_train=False, test_function_json=test_function_json)
    agent_optimizer.update_function_call_forall(
        function_json, mathproxyagent=agent)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Starting at: ", start_time.strftime("%H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--remove_python", action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--math_type", type=str, default="algebra")
    args = parser.parse_args()
    main(args)
    end_time = datetime.datetime.now()
    print("End at: ", end_time.strftime("%H:%M:%S"))
    print("Total time elapsed: ", str(end_time - start_time))
