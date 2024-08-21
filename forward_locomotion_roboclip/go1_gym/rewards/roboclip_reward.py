import torch
import numpy as np
from forward_locomotion_roboclip.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from VLMs.vlm_tests import S3D_Method
from VLMs.utils import compute_similarity,read_video
import os
from ml_logger import logger

VIDEO_STORE_ROOT_PATH = os.path.join(logger.root,logger.prefix,"videos")

class EurekaReward():
    def __init__(self, env):
        self.env = env
        self.model = S3D_Method()
        self.video_resolution = 32
        self.env_num = len(self.env.envs)
        demo_frames,_ = read_video(os.path.join(logger.root,"robodog.mp4"),self.video_resolution)
        self.demo_embedding = self.model.embed_video(demo_frames)
        self.initialised = False

    def load_env(self, env):
        self.env = env
        self.env_num = len(self.env.envs)

    def similarity_score(self):
        score = torch.zeros_like(self.env.reset_buf).float()
        for env_id in range(self.env_num):
            if self.env.reset_buf[env_id]:
                video_path = os.path.join(VIDEO_STORE_ROOT_PATH,str(env_id),"new.mp4")
                frames,total_frames = read_video(video_path,resolution=self.video_resolution)
                #print("Total_frames: ",total_frames)
                try:
                    action_embedding = self.model.embed_video(frames)
                    sim_score = compute_similarity(self.demo_embedding,action_embedding)
                    score[env_id] = sim_score.item()
                except:
                    pass
                    # print(f"Env {env_id} too short")
                score[env_id] += total_frames/100
                # print(f"Env {env_id} Score: ",score[env_id])

        return score
    
    def compute_reward(self):
        # print("In reward func: ",self.env.reset_buf)

        if self.initialised:
            # Reset Buffer indicates termination or robot requres reset, see legged_robot for more detail
            # Sparse reward as indicated in RoboCLIP, reward only given at the end of episode
            #similarity_reward = self.env.reset_buf.clone().detach().float() * self.similarity_score()
            similarity_reward = self.similarity_score()

            # print(similarity_reward)

            total_reward = similarity_reward
            # print(total_reward)
    
        else:
            self.initialised = True
            total_reward = 0
        reward_components = {
            'similarity_reward': total_reward,
        }

        return total_reward, reward_components
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

