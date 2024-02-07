from utils import record_result, is_termination_msg_gaia, get_train_test, prepare_file
from Optimizer_GPT import AgentOptimizer
from Agent_GPT import GaiaUserProxyAgent
import autogen
import argparse
import os
import datetime


def main(args):

    train_data, test_data = get_train_test(args)
    print("Train dataset length: ", len(train_data))
    print("Test dataset length: ", len(test_data))
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        file_location="./",
    )
    if args.remove_python:
        use_py = False
    else:
        use_py = True
    GAIAProxyAgent = GaiaUserProxyAgent(
        name="gaiaproxy",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": args.code_dir, "use_docker": False},
        is_termination_msg=is_termination_msg_gaia,
        use_py=use_py,
    )
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "seed": 41,
            "config_list": config_list,
        }
    )
    OAI_config = {
        "AZURE_OPENAI_API_KEY": "Your API Key",
        "api_version": "Your API Version",
        "azure_endpoint": "Your endpoint",
        "model": "gpt-4-1106",
    }
    agent_optimizer = AgentOptimizer(OAI_config=OAI_config)
    history_list, statistic_list, function_json, actions = {}, {}, [], []
    pre_performance, cur_performance, epoch, max_epoch, fail_stuck = -1, -1, 0, 10, 5
    if args.debug:
        print("debug")
        test_data = []
        train_data = train_data[:1]
        max_epoch = 3
        fail_stuck = 1
    while epoch < max_epoch:
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
                    action, mathproxyagent=GAIAProxyAgent, assistant=assistant)
        history_list, statistic_list = {}, {}
        GAIAProxyAgent.clear_function_statistic()
        for index, query in enumerate(train_data):
            prepare_file(query["file_name"], work_dir=args.code_dir)
            single_statistic, chat_history = GAIAProxyAgent.initiate_chat(
                recipient=assistant, answer=query["Final answer"], query=query["Question"], problem=query["Question"], file=query["file_name"], code_dir=os.path.abspath(args.code_dir))
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
            history_list, statistic_list, function_json = agent_optimizer.roll_back(
                GAIAProxyAgent, assistant)
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

            function_statistic = GAIAProxyAgent.function_statistic
            test_function_json = []
            for func in function_json:
                func_name = func["name"]
                if func_name in function_statistic.keys() and function_statistic[func_name] == True:
                    test_function_json.append(func)
            agent_optimizer.update_function_call_forall(
                test_function_json, mathproxyagent=GAIAProxyAgent, assistant=assistant)
            test_history_list, test_statistic_list = {}, {}
            for index, query in enumerate(test_data):
                prepare_file(query["file_name"], work_dir=args.code_dir)
                test_single_statistic, test_chat_history = GAIAProxyAgent.initiate_chat(
                    recipient=assistant, answer=query["Final answer"], query=["Question"], problem=query["Question"], file=query["file_name"], code_dir=os.path.abspath(args.code_dir))
                test_history_list["conversation: {index}".format(
                    index=index + 1)] = test_chat_history
                test_statistic_list["conversation: {index}".format(
                    index=index + 1)] = test_single_statistic
            test_folder_path = os.path.join(folder_path, "test")
            record_result(test_history_list, test_statistic_list, function_json, test_folder_path,
                            fail_train=False, test_function_json=test_function_json)
            agent_optimizer.update_function_call_forall(
                function_json, mathproxyagent=GAIAProxyAgent, assistant=assistant)

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Starting at: ", start_time.strftime("%H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gaia")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--remove_python", action='store_true')
    parser.add_argument("--code_dir", type=str, default="_output")
    parser.add_argument("--result_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
    end_time = datetime.datetime.now()
    print("End at: ", end_time.strftime("%H:%M:%S"))
    print("Total time elapsed: ", str(end_time - start_time))