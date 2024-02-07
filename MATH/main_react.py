import os
import argparse
from Agent_React import ReActAgent
from utils_react import get_train_test
import json
import datetime

def evaluate_tasks(agent: ReActAgent, validation_tasks, data_name):
    sum = len(validation_tasks)
    cor_num = 0
    for level, task in enumerate(validation_tasks):
        corr, logs = agent.query(task, True)
        cor_num += corr['is_correct']
        print("Correct: ", corr)
    final_summary = cor_num / sum
    print(final_summary)
    save_path = os.path.join("./result", "react_" + args.math_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "summary.json")
    with open(save_path, 'w') as file:
        json.dump(final_summary, file, indent=4)


if __name__ == "__main__":
    azure_key = "your_key_here"
    azure_endpoint = "your_endpoint_here"
    current_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--math_type", type=str, default="algebra")
    args = parser.parse_args()
    train_data, test_data = get_train_test(args)
    agent = ReActAgent(azure_key,
                       azure_endpoint,
                       use_docker=False,
                       work_dir=os.path.join("./work_dir/", current_time.strftime("%Y-%m-%d_%H-%M-%S")))
    evaluate_tasks(agent, test_data, args.math_type)