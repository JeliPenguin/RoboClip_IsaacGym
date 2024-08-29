If you're running into download issues with the S3D weights (last 2 commands), the two files can also be obtained from our google drive: https://drive.google.com/file/d/1DN8t1feS0t5hYYTfXQfZd596Pu9yXJIr/view?usp=sharing https://drive.google.com/file/d/1p_CMGppAQk3B0_Z0i1r2AN_D1FM3kUFW/view?usp=sharing

## Installation

This repository contains code for DrEureka's reward generation, RAPP, and domain randomization generation pipelines as well as the forward locomotion and globe walking environments. The two environments are modified from [Rapid Locomotion](https://github.com/Improbable-AI/rapid-locomotion-rl) and [Dribblebot](https://github.com/Improbable-AI/dribblebot), respectively.

The following instructions will install everything under one Conda environment. We have tested on Ubuntu 20.04.

1. Create a new Conda environment with:
   ```
   conda create -n roboclip python=3.8
   conda activate roboclip
   ```
2. Install Pytorch with CUDA:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install IsaacGym, the simulator for forward locomotion and globe walking:
   1. Download and install IsaacGym from NVIDIA: https://developer.nvidia.com/isaac-gym.
   2. Unzip the file:
      ```
      tar -xf IsaacGym_Preview_4_Package.tar.gz
      ```
   3. Install the python package:
      ```
      cd isaacgym/python
      pip install -e .
      ```

## Usage

We'll use forward locomotion (`forward_locomotion`) as an example. The following steps can also be done for globe walking (`globe_walking`).

First, run reward generation (Eureka):

```
cd ../eureka
python eureka.py env=forward_locomotion
```

At the end, the final best reward will be saved in `forward_locomotion/go1_gym/rewards/eureka_reward.py` and used for subsequent training runs. The Eureka logs will be stored in `eureka/outputs/[TIMESTAMP]`, and the run directory of the best-performing policy will be printed to terminal.

Second, copy the run directory and run RAPP:

```
cd ../dr_eureka
python rapp.py env=forward_locomotion run_path=[YOUR_RUN_DIRECTORY]
```

This will update the prompt in `dr_eureka/prompts/initial_users/forward_locomotion.txt` with the computed RAPP bounds.

Third, run run DR generation with the new reward and RAPP bounds:

```
python dr_eureka.py env=forward_locomotion
```

The trained policies are ready for deployment, see the section below.

## Deployment

Our deployment infrastructure is based on [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways). We'll use forward locomotion as an example, though the deployment setup for both environments are essentially the same.

1. Add the (relative) path to your checkpoint to `forward_locomotion/go1_gym_deploy/scripts/deploy_policy.py`. Note that you can have multiple policies at once and switch between them.
2. Start up the Go1, and connect to it on your machine via Ethernet. Make sure you can ssh onto the NX (`192.168.123.15`).
3. Put the robot into damping mode with the controller: L2+A, L2+B, L1+L2+START. The robot should be lying on the ground afterwards.
4. Run the following to send the checkpoint and code to the Go1:
   ```
   cd forward_locomotion/go1_gym_deploy/scripts
   ./send_to_unitree.sh
   ```
5. Now, ssh onto the Go1 and run the following:
   ```
   chmod +x installer/install_deployment_code.sh
   cd ~/go1_gym/go1_gym_deploy/scripts
   sudo ../installer/install_deployment_code.sh
   ```
6. Make sure your Go1 is in a safe location and hung up. Start up two prompts in the Go1. In the first, run:
   ```
   cd ~/go1_gym/go1_gym_deploy/autostart
   ./start_unitree_sdk.sh
   ```
7. In the second, run:
   ```
   cd ~/go1_gym/go1_gym_deploy/docker
   sudo make autostart && sudo docker exec -it foxy_controller bash
   ```
8. The previous command should enter a Docker image. Within it, run:
   ```
   cd /home/isaac/go1_gym && rm -r build && python3 setup.py install && cd go1_gym_deploy/scripts && python3 deploy_policy.py
   ```
9. Now, you can press R2 on the controller, and the robot should extend its legs (calibrate).
10. Pressing R2 again will start the policy.
11. To switch policies, press L1 or R1 to switch between policies in the list in `deploy_policy.py`.

## Code Structure

DrEureka manipulates pre-defined environments by inserting generated reward functions and domain randomization configurations. To do so, we have designed the environment code to be modular and easily configurable. Below, we explain how the components of our code interact with each other, using forward locomotion as an example:

`eureka/eureka.py` runs the reward generation process. It uses:

1. **Environment source code** as input to the LLM, which is at `eureka/envs/forward_locomotion.py`. This is a shortened version of the actual environment code to save token usage.
2. **Reward signature definition** as input to the LLM, which is at `eureka/prompts/reward_signatures/forward_locomotion.txt`. This file should contain a simple format for the LLM to follow. It may also contain additional instructions or explanations for the format, if necessary.
3. **Location of training script**, which is defined as `train_script: scripts/train.py` in `eureka/cfg/env/forward_locomotion.yaml`.
4. **Location of the reward template and output files**, which are defined as `reward_template_file: go1_gym/rewards/eureka_reward_template.py` and `reward_output_file: go1_gym/rewards/eureka_reward.py` in `eureka/cfg/env/forward_locomotion.yaml`. Eureka reads the template file's boilerplate code, fills in the reward function, and writes to the output file for use during training.
5. **Function to extract training metrics**, which is defined in `eureka/utils/misc.py` as `construct_run_log(stdout_str)`. This function parses the training script's standard output into a dictionary. Alternatively, it can be used to load a file containing metrics saved during training (for example, tensorboard logs).

`dr_eureka/rapp.py` computes the RAPP bounds. It uses:

1. **Location of the play (evaluation) script**, which is defined as `play_script: scripts/play.py` in `dr_eureka/cfg/env/forward_locomotion.yaml`.
2. **Location of the DR template and output files**, which are defined as `dr_template_file: go1_gym/envs/base/legged_robot_config_template.py` and `dr_output_file: go1_gym/envs/base/legged_robot_config.py` in `dr_eureka/cfg/env/forward_locomotion.yaml`. Like the reward template/output setup, DrEureka fills in the boilerplate code and writes to the output file for use during evaluation.
3. **List of randomizable DR parameters**, defined in the variable `parameter_test_vals` in `dr_eureka/rapp.py`.
4. **Simple success criteria** for the task, defined as the function `forward_locomotion_success()` in `dr_eureka/rapp.py`.

`dr_eureka/dr_eureka.py` runs the DR generation process. It uses:

1. **RAPP bounds** as input to the LLM, defined in `dr_eureka/prompts/initial_users/forward_locomotion.txt`. This uses the direct output of `dr_eureka/rapp.py`.
2. **Best reward function**, the output of reward generation. This should be in the file defined in `reward_output_file: go1_gym/rewards/eureka_reward.py`.
3. **Location of the training script**, same as reward generation. This is defined in `dr_eureka/cfg/env/forward_locomotion.yaml`.
4. **Location of the DR template and output files**, same as RAPP.
5. **Function to extract training metrics**, same as reward generation. Note that this is used only for a general idea of the policy's performance in simulation, and unlike reward generation, is not used for iterative feedback.

## Acknowledgements

We thank the following open-sourced projects:

- Our simulation runs in [IsaacGym](https://developer.nvidia.com/isaac-gym).
- Our LLM-generation algorithm builds on [Eureka](https://github.com/eureka-research/Eureka).
- Our environments are adapted from [Rapid Locomotion](https://github.com/Improbable-AI/rapid-locomotion-rl) and [Dribblebot](https://github.com/Improbable-AI/dribblebot).
- The environment structure and training code build on [Legged Gym](https://github.com/leggedrobotics/legged_gym) and [RSL_RL](https://github.com/leggedrobotics/rsl_rl).

## License

This codebase is released under [MIT License](LICENSE).

## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{ma2024dreureka,
    title   = {DrEureka: Language Model Guided Sim-To-Real Transfer},
    author  = {Yecheng Jason Ma and William Liang and Hungju Wang and Sam Wang and Yuke Zhu and Linxi Fan and Osbert Bastani and Dinesh Jayaraman}
    year    = {2024},
}
```
