import os
import copy
import time
import numpy as np
import gym
from tqdm import tqdm
from collections import deque
import hydra

import torch
gym.logger.set_level(40)

import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model_vlm_rating import RewardModel
from vlm_query_metaworld import construct_gemini_keys


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        if cfg.vlm_feedback:
            construct_gemini_keys(num_rollout_workers=cfg.n_processes_query)

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_wandb=cfg.log_save_wandb,
            log_frequency=cfg.log_frequency,
            agent=cfg.algo_name,
            cfg=cfg
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg, img_width=cfg.image_size, img_height=cfg.image_size)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        utils.set_seed_everywhere(cfg.seed)     # Set seed again to avoid messy randomness during env making
        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # Region for image-based reward model
        image_height = image_width = cfg.image_size
        if 'metaworld' in cfg.env:
            image_height = image_width = 224
        self.image_height = image_height
        self.image_width = image_width

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity) if not self.cfg.image_reward else 200000, # we cannot afford to store too many images in the replay buffer.
            self.device,
            store_image=self.cfg.image_reward,
            image_size=image_height)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            teacher_num_ratings=cfg.num_ratings,
            env_name=cfg.env,
            capacity=cfg.max_feedback,

            ### image-based reward model parameters
            vlm_feedback=cfg.vlm_feedback,
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
            softmax_temp=cfg.softmax_temp,
            n_processes_query=cfg.n_processes_query,
            reward_loss=cfg.reward_loss,
            weighting_loss=cfg.weighting_loss,
            batch_stratify=cfg.batch_stratify,
            use_cached=cfg.use_cached,
            query_cached=cfg.query_cached,
        )

        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, self.cfg.model_load_step)

        if self.cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, self.cfg.model_load_step)

    def evaluate(self, n_episodes_save_gif=0):
        print(f"Log: {self.work_dir}")
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)
        
        for episode in range(self.cfg.num_eval_episodes):
            images = []
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)

                if "metaworld" in self.cfg.env and episode < n_episodes_save_gif:
                    rgb_image = self.env.render()
                    if "drawer" not in self.cfg.env:
                        rgb_image = rgb_image[::-1, :, :]
                    images.append(rgb_image)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            if "metaworld" in self.cfg.env and episode < n_episodes_save_gif:
                save_gif_path = os.path.join(
                    save_gif_dir,
                    'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2))
                )
                utils.save_numpy_as_gif(np.array(images), save_gif_path)

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        total_acc = None
        # get feedbacks
        labeled_queries = self.reward_model.uniform_sampling(self.logger)

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            print(f"Labeled feedback: {self.labeled_feedback}")
            # update reward
            for epoch in tqdm(range(self.cfg.reward_update), desc="Reward training"):
                self.reward_model.train()
                train_acc, class_counts = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97 and epoch > self.cfg.least_reward_update:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))
        vlm_metric = copy.deepcopy(self.reward_model.vlm_label_acc)
        return total_acc, class_counts, vlm_metric

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 

        start_time = time.time()
        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    for key, value in extra.items():
                        self.logger.log('train/' + key, value, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                ep_info = []
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                print("finished unsupervised exploration!!")
                # first learn reward
                rew_acc, class_counts, vlm_metric = self.learn_reward(first_flag=1)
                if rew_acc is not None:
                    self.logger._try_sw_log('train/reward_acc', rew_acc, self.step)
                    for i in range(len(class_counts)):
                        self.logger._try_sw_log(f"debug/count_rate_{i}", class_counts[i], self.step)
                    for key, val in vlm_metric.items():
                        self.logger._try_sw_log(f"debug/vlm_{key}", val, self.step)
                
                # relabel buffer
                self.reward_model.eval()
                with torch.no_grad():
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                self.reward_model.train()
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:

                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        rew_acc, class_counts, vlm_metric = self.learn_reward()
                        if rew_acc is not None:
                            self.logger._try_sw_log('train/reward_acc', rew_acc, self.step)
                            for i in range(len(class_counts)):
                                self.logger._try_sw_log(f"debug/count_rate_{i}", class_counts[i], self.step)
                            for key, val in vlm_metric.items():
                                self.logger._try_sw_log(f"debug/vlm_{key}", val, self.step)
                        self.reward_model.eval()
                        with torch.no_grad():
                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                        self.reward_model.train()
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                if self.step % 1000 == 0:
                    print("unsupervised exploration!!")
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,  gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            ep_info.append(extra)
            if self.cfg.image_reward:
                if 'metaworld' in self.cfg.env:
                    rgb_image = self.env.render()
                    if "drawer" not in self.cfg.env:
                        rgb_image = rgb_image[::-1, :, :] # swap top-bottom
                else:
                    raise NotImplementedError
            else:
                rgb_image = None

            self.reward_model.eval()
            if self.cfg.image_reward:
                image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0     # convert to CHW and normalize
                image = image.reshape(1, 3, image.shape[1], image.shape[2])         # extend batch dim
                with torch.no_grad():
                    reward_hat = self.reward_model.r_hat(image)
            else:
                with torch.no_grad():
                    reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            self.reward_model.train()

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done, img=rgb_image, extra=extra)
            self.replay_buffer.add(obs, action, reward_hat, next_obs, done, done_no_max, image=rgb_image)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.step % 100000 == 0 and self.step > 0:
                self.agent.save(self.work_dir, self.step)
                self.reward_model.save(self.work_dir, self.step)
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path='config', config_name='train_PEBBLE_VLM.yaml')
def main(cfg):
    workspace = Workspace(cfg)

    if cfg.mode == "eval":
        workspace.evaluate(n_episodes_save_gif=cfg.n_episodes_save_gif)
        exit()
    workspace.run()


if __name__ == '__main__':
    main()