from utils import record_result
from Agent_React import ReActAgent
from datasets import load_dataset
from Optimizer_React import AgentOptimizer
import argparse
import os
import datetime
import openai
import autogen

def main(args):
    dataset = load_dataset("json", data_files={
        'test': f"{args.dataset_name_or_path}/problems_test_selected_hard.json",
        'train': f"{args.dataset_name_or_path}/problems_train_selected_hard.json"
    })
    train_data, test_data = dataset["train"], dataset["test"]
    print("Train dataset length: ", len(train_data))
    print("Test dataset length: ", len(test_data))
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
    agent = ReActAgent(
        config_list_gpt_35,
        use_docker=False,
        work_dir=args.code_dir,
    )
    agent_optimizer = AgentOptimizer(config_list_gpt_4)

    history_list, statistic_list, function_json, actions = {}, {}, [], []
    pre_performance, cur_performance, epoch, max_epoch, fail_stuck = -1, -1, 0, 2, 6

    while epoch < max_epoch:
        print("Starting Epoch {} at {}".format(epoch, datetime.datetime.now().strftime("%H:%M:%S")))
        print("Time elapsed: ", (datetime.datetime.now() - start_time).seconds, "(s)")
        print("--------------epoch, fail_stuck-----------------")
        print(epoch, fail_stuck)
        agent_optimizer.make_checkpoint(history_list, statistic_list, function_json)
        if args.result_dir is None:
            result_dir = "./result"
        else:
            result_dir = args.result_dir

        folder_path = os.path.join(result_dir, "epoch_{epoch}".format(epoch=str(epoch)))
        if len(history_list) != 0:
            actions, function_json = agent_optimizer.step(
                history_list, statistic_list, function_json)
            for action in actions:
                agent_optimizer.update_function_call(
                    action, agent=agent)
        history_list, statistic_list = {}, {}
        agent.clear_function_statistic()

        for index, query in enumerate(train_data.select(range(10))):
            single_statistic = {}
            chat_history = ""
            try:
                single_statistic, chat_history = agent.query(query, True)
            except openai.BadRequestError as e:
                error_message = str(e)
                single_statistic["is_correct"] = 0
                print("error information: {}".format(error_message))
            history_list["conversation: {index}".format(index=index + 1)] = chat_history
            statistic_list["conversation: {index}".format(index=index + 1)] = single_statistic
            print("single_statistic", single_statistic)

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
            history_list, statistic_list, function_json = agent_optimizer.roll_back(agent)
            fail_stuck -= 1
            agent_optimizer.save_fail_function(cur_performance, fail_func)

        elif cur_performance < pre_performance and len(actions) == 0:
            record_result(history_list, statistic_list, function_json, folder_path, fail_train=True)
            fail_stuck -= 1

        else:
            fail_stuck = 6
            agent_optimizer.clear_fail_function()
            epoch += 1
            pre_performance = cur_performance
            record_result(history_list, statistic_list, function_json, folder_path)

            if epoch == max_epoch:
                function_statistic = agent.function_statistic
                test_function_json = []
                for func in function_json:
                    func_name = func["name"]
                    if func_name in function_statistic.keys() and function_statistic[func_name] == True:
                        test_function_json.append(func)

                print("test_function_json", test_function_json)
                if agent.functions:
                    print(agent.functions)
                agent_optimizer.update_function_call_forall(
                    test_function_json, mathproxyagent=agent)

                if agent.functions:
                    print(agent.functions)

                test_history_list, test_statistic_list = {}, {}
                for index, query in enumerate(test_data.select(range(100))):
                    test_single_statistic, test_chat_history = agent.query(query, True)
                    test_history_list["conversation: {index}".format(
                        index=index + 1)] = test_chat_history
                    test_statistic_list["conversation: {index}".format(
                        index=index + 1)] = test_single_statistic

                test_folder_path = os.path.join(folder_path, "test")
                record_result(test_history_list, test_statistic_list, function_json, test_folder_path,
                              fail_train=False, test_function_json=test_function_json)

            if agent.functions:
                print(agent.functions)

            agent_optimizer.update_function_call_forall(function_json, mathproxyagent=agent)

            if agent.functions:
                print(agent.functions)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Starting at: ", start_time.strftime("%H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tabmwp")
    parser.add_argument("--remove_python", type=bool, default=False)
    parser.add_argument("--code_dir", type=str, default="_output_react")
    parser.add_argument("--result_dir", type=str, default='result_react_opt')
    parser.add_argument("--training_batch_size", type=int, default=10)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="HuggingFace dataset name or local path",
        default="dataset"
    )
    args = parser.parse_args()
    main(args)
    end_time = datetime.datetime.now()
    print("End at: ", end_time.strftime("%H:%M:%S"))
    print("Total time elapsed: ", str(end_time - start_time))
