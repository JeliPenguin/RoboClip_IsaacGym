import hydra
import numpy as np 
import logging 
import matplotlib.pyplot as plt
import os
import subprocess
from pathlib import Path
import shutil
from utils.misc import * 
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{EUREKA_ROOT_DIR}/.."

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    task_rew_file = f'{ROOT_DIR}/{env_name}/{cfg.env.reward_file}'

    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    # task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.env.reward_output_file}"

    video_file = f"{workspace_dir}/video/dog.mp4"

    # # Loading all text prompts
    # prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'
    # initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    # code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    # code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    # initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    # reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
    # policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    # execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    # initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    # initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)

    DUMMY_FAILURE = -10000
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response

        response_id = 0

        logging.info(f"Iteration {iter}:")
                
        # Add the Eureka Reward Signature to the environment code
        # cur_task_rew_code_string = task_rew_code_string.replace("# INSERT EUREKA REWARD HERE", code_string)

        cur_task_rew_code_string = task_rew_code_string

        # Save the new environment code when the output contains valid code string!
        with open(output_file, 'w') as file:
            file.writelines(cur_task_rew_code_string + '\n')

        # Copy the generated environment code to hydra output directory for bookkeeping
        shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

        # Find the freest GPU to run GPU-accelerated RL
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"env_iter{iter}_response{response_id}.txt"
        with open(rl_filepath, 'w') as f:
            command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --dr-config off --reward-config eureka"
            command = command.split(" ")
            if not cfg.use_wandb:
                command.append("--no-wandb")
            if cfg.debug:
                process = subprocess.run(command)
            else:
                process = subprocess.run(command, stdout=f, stderr=f)
        block_until_training(rl_filepath, success_keyword=cfg.env.success_keyword, failure_keyword=cfg.env.failure_keyword,
                                 log_status=True, iter_num=iter, response_id=response_id)

        # Gather RL training results and construct reward reflection
        successes = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False 

        rl_filepath = f"env_iter{iter}_response{response_id}.txt"
        code_paths.append(f"env_iter{iter}_response{response_id}.py")
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 

        traceback_msg = filter_traceback(stdout_str)

        if traceback_msg == '':
            # If RL execution has no error, provide policy statistics feedback
            exec_success = True
            run_log = construct_run_log(stdout_str)
            
            # Compute Correlation between Human-Engineered and GPT Rewards
            if "gt_reward" in run_log and "gpt_reward" in run_log:
                gt_reward = np.array(run_log["gt_reward"])
                gpt_reward = np.array(run_log["gpt_reward"])
                reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                reward_correlations.append(reward_correlation)

            # Add reward components log to the feedback
            for metric in sorted(run_log.keys()):
                if "/" not in metric:
                    metric_cur_max = max(run_log[metric])
                    if "consecutive_successes" == metric:
                        successes.append(metric_cur_max)
                  
        else:
            # Otherwise, provide execution traceback error feedback
            successes.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
            
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) 

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)
    
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")

    best_reward = file_to_string(max_reward_code_path)
    with open(output_file, 'w') as file:
        file.writelines(best_reward + '\n')
    
    # Get run directory of best-performing policy
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Dashboard: "):
            run_dir = line.split(": ")[1].strip()
            run_dir = run_dir.replace("http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/")
            logging.info("Best policy run directory: " + run_dir)

if __name__ == "__main__":
    main()