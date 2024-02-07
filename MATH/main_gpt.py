import autogen
from utils_gpt import record_result, is_termination_msg_mathchat, get_train_test
from Optimizer_GPT import AgentOptimizer
from Agent_GPT import MathUserProxyAgent
import argparse
import os

def main(args):

    train_data, test_data = get_train_test(args)
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
    )
    if args.remove_python:
        use_py = False
    else:
        use_py = True
    mathproxyagent = MathUserProxyAgent(
        name="mathproxyagent",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": args.code_dir, "use_docker": False},
        is_termination_msg=is_termination_msg_mathchat,
        use_py=use_py,
    )
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "seed": 42,
            "config_list": config_list,
        }
    )
    OAI_config = {
        "AZURE_OPENAI_API_KEY": "Your API Key",
        "api_version": "2023-12-01-preview",
        "azure_endpoint": "Your endpoint",
        "model": "Your model",
    }
    agent_optimizer = AgentOptimizer(OAI_config=OAI_config)
    history_list, statistic_list, function_json, actions = {}, {}, [], []
    pre_performance, cur_performance, epoch, max_epoch, fail_stuck = -1, -1, 0, args.epoch_num, 10
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
                    action, mathproxyagent=mathproxyagent, assistant=assistant)
        history_list, statistic_list = {}, {}
        mathproxyagent.clear_function_statistic()
        for index, query in enumerate(train_data):
            single_statistic, chat_history = mathproxyagent.initiate_chat(
                recipient=assistant, answer=query["answer"], query=query["question"], problem=query["question"])
            history_list["conversation: {index}".format(
                index=index + 1)] = chat_history
            statistic_list["conversation: {index}".format(
                index=index + 1)] = single_statistic
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
                mathproxyagent, assistant)
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
            
    function_statistic = mathproxyagent.function_statistic
    test_function_json = []
    for func in function_json:
        func_name = func["name"]
        if func_name in function_statistic.keys() and function_statistic[func_name] == True:
            test_function_json.append(func)
    agent_optimizer.update_function_call_forall(
        test_function_json, mathproxyagent=mathproxyagent, assistant=assistant)
    test_history_list, test_statistic_list = {}, {}
    for index, query in enumerate(test_data):
        test_single_statistic, test_chat_history = mathproxyagent.initiate_chat(
            recipient=assistant, answer=query["answer"], query=["question"], problem=query["question"])
        test_history_list["conversation: {index}".format(
            index=index + 1)] = test_chat_history
        test_statistic_list["conversation: {index}".format(
            index=index + 1)] = test_single_statistic
    test_folder_path = os.path.join(folder_path, "test")
    record_result(test_history_list, test_statistic_list, function_json, test_folder_path,
                    fail_train=False, test_function_json=test_function_json)
    agent_optimizer.update_function_call_forall(
        function_json, mathproxyagent=mathproxyagent, assistant=assistant)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--remove_python", action='store_true')
    parser.add_argument("--math_type", type=str, default="algebra")
    parser.add_argument("--code_dir", type=str, default="_output")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--epoch_num", type=int, default=10)
    args = parser.parse_args()
    main(args)
