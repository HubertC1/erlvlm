import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import os
import time
import pickle

from conv_net import CNN, fanin_init
from vlm_query_metaworld import vlm_reasoning_rating_metaworld
import multiprocessing as mp
from sklearn.metrics import confusion_matrix

device = 'cuda'


def extract_query_index(path):
    filename = os.path.basename(path)
    filename = filename.split('.')[0]  # remove extension
    index_str = filename[filename.find('_total'):]
    index = int(index_str.replace('_total', ''))
    return index

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def gen_image_net(image_height, image_width,
                  conv_kernel_sizes=[5, 3, 3 ,3],
                  conv_n_channels=[16, 32, 64, 128],
                  conv_strides=[3, 2, 2, 2]):
    conv_args=dict( # conv layers
        kernel_sizes=conv_kernel_sizes, # for sweep into, cartpole, drawer open.
        n_channels=conv_n_channels,
        strides=conv_strides,
        output_size=1,
    )
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    return CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=image_height,
        input_width=image_width,
        input_channels=3,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

def gen_image_net2():
    from torchvision.models.resnet import ResNet
    from torchvision.models.resnet import BasicBlock

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model


class RewardModel:
    def __init__(
            self, ds, da,
            ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
            env_maker=None, max_size=100, activation='tanh', capacity=5e5,
            large_batch=1, label_margin=0.0,
            teacher_beta=-1, teacher_gamma=1,
            teacher_eps_mistake=0,
            teacher_eps_skip=0,
            teacher_eps_equal=0,
            teacher_num_ratings=2,
            env_name=None,

            # image based reward
            vlm_feedback=True,
            image_reward=True,
            image_height=128,
            image_width=128,
            resnet=False,
            conv_kernel_sizes=[5, 3, 3, 3],
            conv_strides=[3, 2, 2, 2],
            conv_n_channels=[16, 32, 64, 128],
            reward_model_layers=3,
            reward_model_H=256,
            softmax_temp=30,
            n_processes_query=1,
            reward_loss="ce",
            weighting_loss=False,
            batch_stratify=False,
            use_cached=False,
            query_cached=None,
    ):

        # train data is trajectories, must process to sa and s..
        self.env_name = env_name
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.vlm_feedback = vlm_feedback
        self.image_reward = image_reward
        self.image_height = image_height
        self.image_width = image_width
        self.resnet = resnet
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_n_channels = conv_n_channels
        self.reward_model_layers = reward_model_layers
        self.reward_model_H = reward_model_H
        self.softmax_temp = softmax_temp
        self.n_processes_query = n_processes_query

        self.capacity = int(capacity)
        if self.image_reward:
            assert self.size_segment == 1, f"We use segment size of 1 in MetaWorld."
            self.buffer_seg1 = np.empty((self.capacity, size_segment, image_height, image_width, 3), dtype=np.uint8)
            self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        else:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
            self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.extra_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        if self.image_reward:
            if self.resnet:
                self.train_batch_size = 32
            else:
                self.train_batch_size = 512
        else:
            self.train_batch_size = 128
        self.reward_loss = reward_loss
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.num_ratings = teacher_num_ratings
        self.weighting_loss = weighting_loss
        self.batch_stratify = batch_stratify

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - self.num_ratings * self.label_margin

        self.num_timesteps = 0
        self.member_1_pred_reward = []
        self.member_2_pred_reward = []
        self.member_3_pred_reward = []
        self.real_rewards = []
        self.eps_norm = 1e-6
        self.vlm_label_acc = {'accuracy': 0, 'precision': 0, 'recall': 0, 'fpr': 0}
        self.total_feedback_until_now = 0
        self.use_cached = use_cached
        self.query_cached = query_cached
        self.cached_loaded_cnt = 0
        self.all_cached_queries = None


    def eval(self,):
        for i in range(self.de):
            self.ensemble[i].eval()

    def train(self,):
        for i in range(self.de):
            self.ensemble[i].train()

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]

    def mae_loss(self, input_logit, target, batch_weights=None):
        predict = F.softmax(input_logit, dim=1)
        if batch_weights is None:
            loss = F.l1_loss(predict, target)
        else:
            raw_loss = F.l1_loss(predict, target, reduction="none")
            weighted_loss = raw_loss.sum(dim=1) * batch_weights.squeeze(1)
            loss = weighted_loss.sum() / batch_weights.sum()
        return loss

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def construct_ensemble(self):
        for i in range(self.de):
            if self.image_reward:
                if self.resnet:
                    model = gen_image_net2().float().to(device)
                else:
                    model = gen_image_net(self.image_height, self.image_width, self.conv_kernel_sizes,
                                          self.conv_n_channels, self.conv_strides).float().to(device)
            else:
                model = nn.Sequential(*gen_net(in_size=self.ds+self.da,
                                               out_size=1, H=self.reward_model_H, n_layers=self.reward_model_layers,
                                               activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)

    def add_data(self, obs, act, rew, done, img=None, extra=None):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew

        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        if img is not None:
            flat_img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            if img is not None:
                self.img_inputs.append(flat_img)
                self.extra_inputs.append([copy.deepcopy(extra)])
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if img is not None:
                self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                self.extra_inputs[-1].append(copy.deepcopy(extra))
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                if img is not None:
                    self.img_inputs = self.img_inputs[1:]
                    self.extra_inputs = self.extra_inputs[1:]
            self.inputs.append([])
            self.targets.append([])
            if img is not None:
                self.img_inputs.append([])
                self.extra_inputs.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                if img is not None:
                    self.img_inputs[-1] = flat_img
                    self.extra_inputs[-1].append(copy.deepcopy(extra))
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                if img is not None:
                    self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], flat_img], axis=0)
                    self.extra_inputs[-1].append(copy.deepcopy(extra))

    def r_hat_member(self, x, member=-1):
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member))

    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member)))

    def get_queries(self, mb_size=100):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1 = None

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        if self.image_reward:
            train_images = np.array(self.img_inputs[:max_len])
            train_extras = np.array(copy.deepcopy(self.extra_inputs[:max_len]), dtype=object)

        replace = True if max_len < mb_size else False
        # Sampling trajectories from buffer of reward model
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=replace)
        sa_t_1 = train_inputs[batch_index_1]
        r_t_1 = train_targets[batch_index_1]
        if self.image_reward:
            img_t_1 = train_images[batch_index_1]
            extra_t_1 = train_extras[batch_index_1]

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
        if self.image_reward:
            img_t_1 = img_t_1.reshape(-1, img_t_1.shape[2], img_t_1.shape[3], img_t_1.shape[4])
            extra_t_1 = extra_t_1.reshape(-1, 1)

        time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size)])

        # Sampling timestep within a selected trajectory
        range_steps = list(range(len_traj - self.size_segment))
        random_idx_1 = np.random.choice(range_steps, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + random_idx_1

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        if self.image_reward:
            img_t_1 = np.take(img_t_1, time_index_1, axis=0)
            extra_t_1 = np.take(extra_t_1, time_index_1, axis=0)


        return sa_t_1, r_t_1, img_t_1, extra_t_1

    def put_queries(self, sa_t_1, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def get_label(self, sa_t_1, r_t_1, img_t_1):
        sum_r_t_1 = np.sum(r_t_1, axis=1)

        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            r_t_1 = r_t_1[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)

        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
        temp_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_1 = np.zeros_like(temp_r_t_1)

        rewards = np.sum(r_t_1, axis=1)

        if "sweep-into-v2" in self.env_name:        # per state: min_r=0, max_r=10 => seg_size=50, min_r=0, max_r=500
            if self.num_ratings == 2:
                sum_r_t_1[(temp_r_t_1 < 250)] = 0
                sum_r_t_1[(temp_r_t_1 >= 250.0001)] = 1
            elif self.num_ratings == 3:
                sum_r_t_1[(temp_r_t_1 < 166.67)] = 0
                sum_r_t_1[(temp_r_t_1 >= 166.6701) & (temp_r_t_1 < 333.33)] = 1
                sum_r_t_1[(temp_r_t_1 >= 333.3301)] = 2
            elif self.num_ratings == 4:
                sum_r_t_1[(temp_r_t_1 < 125.0)] = 0
                sum_r_t_1[(temp_r_t_1 >= 125.001) & (temp_r_t_1 < 250)] = 1
                sum_r_t_1[(temp_r_t_1 >= 250.0001) & (temp_r_t_1 < 375.0)] = 2
                sum_r_t_1[(temp_r_t_1 >= 375.0001)] = 3
            elif self.num_ratings == 5:
                sum_r_t_1[(temp_r_t_1 < 100)] = 0
                sum_r_t_1[(temp_r_t_1 >= 100.0001) & (temp_r_t_1 < 200)] = 1
                sum_r_t_1[(temp_r_t_1 >= 200.0001) & (temp_r_t_1 < 300)] = 2
                sum_r_t_1[(temp_r_t_1 >= 300.0001) & (temp_r_t_1 < 400)] = 3
                sum_r_t_1[(temp_r_t_1 >= 400.0001)] = 4
            else:
                sum_r_t_1[temp_r_t_1 < 83.3] = 0
                sum_r_t_1[(temp_r_t_1 >= 83.3001) & (temp_r_t_1 < 166.67)] = 1
                sum_r_t_1[(temp_r_t_1 >= 166.6701) & (temp_r_t_1 < 250)] = 2
                sum_r_t_1[(temp_r_t_1 >= 250.0001) & (temp_r_t_1 < 333.33)] = 3
                sum_r_t_1[(temp_r_t_1 >= 333.3301) & (temp_r_t_1 < 416.66)] = 4
                sum_r_t_1[(temp_r_t_1 >= 416.6601)] = 5

        elif "drawer-open-v2" in self.env_name: # per state: min_r=0.6, max_r=5.04 => seg_size=50, min_r=30, max_r=252
            if self.num_ratings == 2:
                sum_r_t_1[(temp_r_t_1 < 141)] = 0
                sum_r_t_1[(temp_r_t_1 >= 141.0001)] = 1
            elif self.num_ratings == 3:
                sum_r_t_1[(temp_r_t_1 < 104)] = 0
                sum_r_t_1[(temp_r_t_1 >= 104.0001) & (temp_r_t_1 < 178.0)] = 1
                sum_r_t_1[(temp_r_t_1 >= 178.0001)] = 2
            elif self.num_ratings == 4:
                sum_r_t_1[(temp_r_t_1 < 85.5)] = 0
                sum_r_t_1[(temp_r_t_1 >= 85.5001) & (temp_r_t_1 < 141)] = 1
                sum_r_t_1[(temp_r_t_1 >= 141.0001) & (temp_r_t_1 < 196.5)] = 2
                sum_r_t_1[(temp_r_t_1 >= 196.5001)] = 3
            elif self.num_ratings == 5:
                sum_r_t_1[(temp_r_t_1 < 74.4)] = 0
                sum_r_t_1[(temp_r_t_1 >= 74.4001) & (temp_r_t_1 < 118.8)] = 1
                sum_r_t_1[(temp_r_t_1 >= 118.8001) & (temp_r_t_1 < 163.2)] = 2
                sum_r_t_1[(temp_r_t_1 >= 163.2001) & (temp_r_t_1 < 207.6)] = 3
                sum_r_t_1[(temp_r_t_1 >= 207.6001)] = 4
            else:
                sum_r_t_1[temp_r_t_1 < 67] = 0
                sum_r_t_1[(temp_r_t_1 >= 67.0001) & (temp_r_t_1 < 104)] = 1
                sum_r_t_1[(temp_r_t_1 >= 104.0001) & (temp_r_t_1 < 141)] = 2
                sum_r_t_1[(temp_r_t_1 >= 141.0001) & (temp_r_t_1 < 178)] = 3
                sum_r_t_1[(temp_r_t_1 >= 178.0001) & (temp_r_t_1 < 215)] = 4
                sum_r_t_1[(temp_r_t_1 >= 215.0001)] = 5

        elif "soccer-v2" in self.env_name:  # per state: min_r=0, max_r=10 => seg_size=50, min_r=0, max_r=500
            if self.num_ratings == 2:
                sum_r_t_1[(temp_r_t_1 < 250)] = 0
                sum_r_t_1[(temp_r_t_1 >= 250.0001)] = 1
            elif self.num_ratings == 3:
                sum_r_t_1[(temp_r_t_1 < 166.67)] = 0
                sum_r_t_1[(temp_r_t_1 >= 166.6701) & (temp_r_t_1 < 333.33)] = 1
                sum_r_t_1[(temp_r_t_1 >= 333.3301)] = 2
            elif self.num_ratings == 4:
                sum_r_t_1[(temp_r_t_1 < 125.0)] = 0
                sum_r_t_1[(temp_r_t_1 >= 125.001) & (temp_r_t_1 < 250)] = 1
                sum_r_t_1[(temp_r_t_1 >= 250.0001) & (temp_r_t_1 < 375.0)] = 2
                sum_r_t_1[(temp_r_t_1 >= 375.0001)] = 3
            elif self.num_ratings == 5:
                sum_r_t_1[(temp_r_t_1 < 100)] = 0
                sum_r_t_1[(temp_r_t_1 >= 100.0001) & (temp_r_t_1 < 200)] = 1
                sum_r_t_1[(temp_r_t_1 >= 200.0001) & (temp_r_t_1 < 300)] = 2
                sum_r_t_1[(temp_r_t_1 >= 300.0001) & (temp_r_t_1 < 400)] = 3
                sum_r_t_1[(temp_r_t_1 >= 400.0001)] = 4
            else:
                sum_r_t_1[temp_r_t_1 < 83.3] = 0
                sum_r_t_1[(temp_r_t_1 >= 83.3001) & (temp_r_t_1 < 166.67)] = 1
                sum_r_t_1[(temp_r_t_1 >= 166.6701) & (temp_r_t_1 < 250)] = 2
                sum_r_t_1[(temp_r_t_1 >= 250.0001) & (temp_r_t_1 < 333.33)] = 3
                sum_r_t_1[(temp_r_t_1 >= 333.3301) & (temp_r_t_1 < 416.66)] = 4
                sum_r_t_1[(temp_r_t_1 >= 416.6601)] = 5

        else:
            if self.num_ratings == 2:
                sum_r_t_1[(temp_r_t_1 < 25)] = 0
                sum_r_t_1[(temp_r_t_1 >= 25.0001)] = 1
            elif self.num_ratings == 3:
                sum_r_t_1[(temp_r_t_1 < 16.67)] = 0
                sum_r_t_1[(temp_r_t_1 >= 16.6701) & (temp_r_t_1 < 33.33)] = 1
                sum_r_t_1[(temp_r_t_1 >= 33.3301)] = 2
            elif self.num_ratings == 4:
                sum_r_t_1[(temp_r_t_1 < 12.5)] = 0
                sum_r_t_1[(temp_r_t_1 >= 12.5001) & (temp_r_t_1 < 25)] = 1
                sum_r_t_1[(temp_r_t_1 >= 25.0001) & (temp_r_t_1 < 37.5)] = 2
                sum_r_t_1[(temp_r_t_1 >= 37.5001)] = 3
            elif self.num_ratings == 5:
                sum_r_t_1[(temp_r_t_1 < 10)] = 0
                sum_r_t_1[(temp_r_t_1 >= 10.0001) & (temp_r_t_1 < 20)] = 1
                sum_r_t_1[(temp_r_t_1 >= 20.0001) & (temp_r_t_1 < 30)] = 2
                sum_r_t_1[(temp_r_t_1 >= 30.0001) & (temp_r_t_1 < 40)] = 3
                sum_r_t_1[(temp_r_t_1 >= 40.0001)] = 4
            else:
                sum_r_t_1[temp_r_t_1 < 8.3] = 0
                sum_r_t_1[(temp_r_t_1 >= 8.3001) & (temp_r_t_1 < 16.67)] = 1
                sum_r_t_1[(temp_r_t_1 >= 16.6701) & (temp_r_t_1 < 25)] = 2
                sum_r_t_1[(temp_r_t_1 >= 25.0001) & (temp_r_t_1 < 33.33)] = 3
                sum_r_t_1[(temp_r_t_1 >= 33.3301) & (temp_r_t_1 < 41.66)] = 4
                sum_r_t_1[(temp_r_t_1 >= 41.6601)] = 5
        labels = sum_r_t_1

        return sa_t_1, r_t_1, labels

    def get_label_from_vlm(self, sa_t_1, r_t_1, img_t_1, extra_t_1, logger=None):
        n_sequences = img_t_1.shape[0]
        sum_r_t_1 = []
        for i in range(n_sequences):
            trajectory_rgb_frames = img_t_1[i].copy()[-1, :, :, :][None, :, :, :]
            feedback, query_info = vlm_reasoning_rating_metaworld(
                raw_observations=trajectory_rgb_frames,
                model_names=("gemini-1.5-pro-002", "gemini-1.5-flash-002"),
                env_name=self.env_name,
                max_rating=self.num_ratings - 1,
                rank=0
            )
            sum_r_t_1.append(feedback)

        labels = np.array(sum_r_t_1)[:, None]

        return sa_t_1, r_t_1, labels

    def get_label_from_vlm_parallel(self, sa_t_1, r_t_1, img_t_1, extra_t_1, logger=None):
        n_sequences = img_t_1.shape[0]
        n_processes = self.n_processes_query

        n_sequences_per_process = int(n_sequences / n_processes)
        sum_r_t_1 = []
        args_list = []
        rank = 0
        rgb_frames_per_process = []
        for i in range(n_sequences):
            trajectory_rgb_frames = img_t_1[i].copy()[-1, :, :, :][None, :, :, :]
            rgb_frames_per_process.append(trajectory_rgb_frames)
            if len(rgb_frames_per_process) == n_sequences_per_process:
                args_list.append((rank, rgb_frames_per_process, self.env_name, self.num_ratings, self.image_height))

                rgb_frames_per_process = []
                rank += 1

        with mp.Pool(n_processes) as pool:
            outputs = pool.map(query_vlm, args_list)

        # Construct outputs from multi-processes into same original order
        # Sort all processes
        chunks_order = []
        for i in range(n_processes):
            chunks_order.append(outputs[i][0][2])
        sorted_chunk_indices = np.argsort(chunks_order)

        for chunk_idx in sorted_chunk_indices:
            cur_chunk = outputs[chunk_idx]

            # Sort sequence in each chunk
            seq_order = []
            for i in range(len(cur_chunk)):
                seq_order.append(cur_chunk[i][1])
            sorted_seq_indices = np.argsort(seq_order)

            for seq_idx in sorted_seq_indices:
                sum_r_t_1.append(cur_chunk[seq_idx][0])

        labels = np.array(sum_r_t_1)[:, None]
        successes = []
        for i in range(n_sequences):
            successes.append(extra_t_1[i].squeeze().item()['success'])

        query_cached = os.path.join(logger._log_dir, "query_cached")
        if not os.path.exists(query_cached):
            os.makedirs(query_cached, exist_ok=True)
        cached_data = {
            'state_action': sa_t_1,
            'reward': r_t_1,
            'image': img_t_1,
            'extra': extra_t_1,
            'label': copy.deepcopy(labels),
        }
        self.total_feedback_until_now += labels.shape[0]
        with open(f"{query_cached}/cached_query_total{self.total_feedback_until_now}.pkl", "wb") as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        all_feedback = np.array(labels).flatten()
        prediction_highest = (all_feedback == (self.num_ratings - 1)).astype(float).tolist()
        try:
            TN, FP, FN, TP = confusion_matrix(successes, prediction_highest).ravel()
            accuracy = (TN + TP) / (TN + TP + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            fpr = FP / (FP + TN)

            self.vlm_label_acc = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fpr': fpr}
        except Exception as e:
            self.vlm_label_acc = {}
            print(e)

        return sa_t_1, r_t_1, labels

    def uniform_sampling(self, logger=None):
        # get queries
        sa_t_1, r_t_1, img_t_1, extra_t_1 =  self.get_queries(mb_size=self.mb_size)

        # get labels
        if self.vlm_feedback and self.image_reward:
            if self.use_cached:
                if self.all_cached_queries is None:
                    self.all_cached_queries = sorted(os.listdir(self.query_cached), key=extract_query_index)
                data = np.load(os.path.join(self.query_cached, self.all_cached_queries[self.cached_loaded_cnt]), allow_pickle=True)
                self.cached_loaded_cnt += 1
                img_t_1 = copy.deepcopy(data['image'])
                labels = copy.deepcopy(data['label'])
                self.total_feedback_until_now += labels.shape[0]
                print('\033[93m' + f"Loaded cached query, total feedback: {self.total_feedback_until_now}" + '\033[0m')
            else:
                # sa_t_1, r_t_1, labels = self.get_label_from_vlm(sa_t_1, r_t_1, img_t_1)
                sa_t_1, r_t_1, labels = self.get_label_from_vlm_parallel(sa_t_1, r_t_1, img_t_1, extra_t_1, logger=logger)
        else:
            sa_t_1, r_t_1, labels = self.get_label(sa_t_1, r_t_1, img_t_1)

        if len(labels) > 0:
            if self.image_reward:
                self.put_queries(img_t_1,labels)
            else:
                self.put_queries(sa_t_1, labels)

        return len(labels)

    def generate_training_indices(self, labels, batch_size, n_classes, n_training_iters, shuffle=False):
        # Get unique labels and their counts
        unique_labels = np.arange(n_classes)
        label_indices = {label: np.where(labels == label)[0].tolist() for label in unique_labels}

        # Shuffle indices for each label
        if shuffle:
            for label in label_indices:
                np.random.shuffle(label_indices[label])

        class_counts = np.zeros(n_classes, dtype=np.int32)
        for class_id in range(n_classes):
            class_counts[class_id] = len(label_indices[class_id])

        sorted_class_indices = np.argsort(class_counts)
        minimum_samples_per_classes = class_counts // n_training_iters
        remaining_samples_per_classes = class_counts - minimum_samples_per_classes * n_training_iters

        batches = []
        for i in range(n_training_iters - 1):
            batch = []
            # Fill minimum samples for each class
            for label in sorted_class_indices:
                end_idx = minimum_samples_per_classes[label]
                indices = label_indices[label][:end_idx]
                class_counts[label] -= len(indices)
                batch.extend(indices)
                del label_indices[label][:end_idx]

            # Fill the minority classes first
            if len(batch) < batch_size:
                for label in sorted_class_indices:
                    if remaining_samples_per_classes[label] > 0:
                        if len(label_indices[label]) > 0:
                            batch.append(label_indices[label][0])
                            del label_indices[label][0]
                            remaining_samples_per_classes[label] -= 1
                            class_counts[label] -= 1
                        else:
                            remaining_samples_per_classes[label] -= 1
                            class_counts[label] -= 1
                    if len(batch) == batch_size:
                        break

            # If the batch still not enough, fill the major class
            if len(batch) < batch_size:
                major_label = sorted_class_indices[-1]
                n_added_major = batch_size - len(batch)
                if n_added_major > class_counts[major_label]:
                    n_added_major = class_counts[major_label]
                batch.extend(label_indices[major_label][:n_added_major])
                del label_indices[major_label][:n_added_major]
                class_counts[major_label] -= n_added_major

            if shuffle:
                np.random.shuffle(batch)
            batches.extend(batch)
            if (class_counts <= 0).all():
                break

        batch = []
        if (class_counts > 0).any():
            for label in sorted_class_indices:
                if len(label_indices[label]) > 0:
                    batch.extend(label_indices[label])
                    del label_indices[label]
            batches.extend(batch)

        return batches

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index

        # generate indices of dataset for training
        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total_batch_index = []

        unique_labels = np.arange(self.num_ratings)
        label_indices = {label: np.where(self.buffer_label[:max_len][:, 0] == label)[0] for label in unique_labels}
        print(f"Rating stats: " + "".join(f"Rating {label}: {len(label_indices[label])}, " for label in label_indices))

        for _ in range(self.de):
            total_batch_index.append(self.generate_training_indices(
                labels=self.buffer_label[:max_len][:, 0],
                batch_size=self.train_batch_size,
                n_classes=self.num_ratings,
                n_training_iters=num_epochs,
                shuffle=True
            ))

        class_counts = np.zeros(self.num_ratings, dtype=np.int32)
        for class_id in range(self.num_ratings):
            class_counts[class_id] = len(label_indices[class_id])

        if self.weighting_loss:
            buffer_weights = np.ones((max_len, 1), dtype=np.float32)

            if (class_counts != 0).all():
                class_weights = [(class_counts.sum() - x) / class_counts.sum() for i, x in enumerate(class_counts)] # invert class count
                class_weights = np.asarray(class_weights) / np.min(class_weights)

                for class_id in range(self.num_ratings):
                    buffer_weights[self.buffer_label[:max_len, 0] == class_id] = class_weights[class_id]
            else:
                class_weights = np.ones(self.num_ratings)
        else:
            buffer_weights = None

        max_len = len(total_batch_index[0])     # Compute again for new generated indices
        total = 0
        for epoch in tqdm(range(num_epochs), desc=f"Training reward (inner loop, bs={self.train_batch_size})"):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                labels = self.buffer_label[idxs]


                if self.num_ratings == 2:
                    num_B = 0
                    num_G = 0

                    for label in labels:

                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=2)
                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        target_onehot = target_onehot * self.label_target + self.label_margin

                    if member == 0:
                        total += labels.size(0)

                    if self.image_reward:
                        sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3))
                        sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                        sa_t_1 = sa_t_1[:, -1, :, :, :]

                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat = r_hat1

                    pred = ((r_hat) - (torch.min(r_hat))) / ((torch.max(r_hat)) - (torch.min(r_hat)) + self.eps_norm)

                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()


                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_G = np_pred[num_B+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k = self.softmax_temp

                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-1)*(k)


                    our_Q = torch.cat([Q_0, Q_1], axis=-1)
                    batch_weights = torch.from_numpy(buffer_weights[idxs]).to(device) if (buffer_weights is not None) else None

                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        if self.reward_loss == "ce":
                            curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    else:
                        if self.reward_loss == "ce":
                            weight = torch.from_numpy(class_weights).float().to(device) if batch_weights is not None else None
                            if weight is not None:
                                criterion = nn.CrossEntropyLoss(weight=weight)
                                curr_loss = criterion(our_Q, labels)
                            else:
                                curr_loss = self.CEloss(our_Q, labels)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())

                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                elif self.num_ratings == 3:
                    num_B = 0
                    num_N = 0
                    num_G = 0

                    for label in labels:

                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_N += 1
                        elif label == 2:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=3)
                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        target_onehot = target_onehot * self.label_target + self.label_margin

                    if member == 0:
                        total += labels.size(0)

                    if self.image_reward:
                        sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3))
                        sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                        sa_t_1 = sa_t_1[:, -1, :, :, :]

                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat = r_hat1

                    pred = ((r_hat) - (torch.min(r_hat))) / ((torch.max(r_hat)) - (torch.min(r_hat)) + self.eps_norm)

                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()


                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_N = np_pred[num_B+num_N-1]
                    upper_bound_G = np_pred[num_B+num_N+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k = self.softmax_temp

                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_2 = -(pred-upper_bound_N)*(pred-1)*(k)


                    our_Q = torch.cat([Q_0, Q_1, Q_2], axis=-1)
                    batch_weights = torch.from_numpy(buffer_weights[idxs]).to(device) if (buffer_weights is not None) else None

                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        if self.reward_loss == "ce":
                            curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    else:
                        if self.reward_loss == "ce":
                            weight = torch.from_numpy(class_weights).float().to(device) if batch_weights is not None else None
                            if weight is not None:
                                criterion = nn.CrossEntropyLoss(weight=weight)
                                curr_loss = criterion(our_Q, labels)
                            else:
                                curr_loss = self.CEloss(our_Q, labels)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())

                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                elif self.num_ratings == 4:
                    num_VB = 0
                    num_B = 0
                    num_G = 0
                    num_VG = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_G += 1
                        elif label == 3:
                            num_VG += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=4)
                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        target_onehot = target_onehot * self.label_target + self.label_margin

                    if member == 0:
                        total += labels.size(0)

                    if self.image_reward:
                        sa_t_1 = np.transpose(sa_t_1, (0, 1, 4, 2, 3))
                        sa_t_1 = sa_t_1.astype(np.float32) / 255.0
                        sa_t_1 = sa_t_1[:, -1, :, :, :]

                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat = r_hat1

                    pred = ((r_hat) - (torch.min(r_hat))) / ((torch.max(r_hat)) - (torch.min(r_hat)) + self.eps_norm)

                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()

                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_G+num_VG-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)

                    k = self.softmax_temp
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_G)*(k)
                    Q_3 = -(pred-upper_bound_G)*(pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3], axis=-1)
                    batch_weights = torch.from_numpy(buffer_weights[idxs]).to(device) if (buffer_weights is not None) else None

                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        if self.reward_loss == "ce":
                            curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    else:
                        if self.reward_loss == "ce":
                            weight = torch.from_numpy(class_weights).float().to(device) if batch_weights is not None else None
                            if weight is not None:
                                criterion = nn.CrossEntropyLoss(weight=weight)
                                curr_loss = criterion(our_Q, labels)
                            else:
                                curr_loss = self.CEloss(our_Q, labels)
                        elif self.reward_loss == "mae":
                            curr_loss = self.mae_loss(our_Q, target_onehot, batch_weights)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())

                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                else:
                    raise NotImplementedError

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        return ensemble_acc, class_counts


def query_vlm(params):
    rank, rgb_frames, env_name, num_ratings, image_height = params
    subprocess_n_sequences = len(rgb_frames)

    return_list = []
    for i in range(subprocess_n_sequences):
        feedback, query_info = vlm_reasoning_rating_metaworld(
            raw_observations=rgb_frames[i],
            model_names=("gemini-1.5-pro-002", "gemini-1.5-flash-002"),
            # model_names=("gemini-1.5-flash-002", "gemini-1.5-flash-002"),
            env_name=env_name,
            max_rating=num_ratings - 1,
            rank=rank
        )
        return_list.append((feedback, i, rank))
    return return_list