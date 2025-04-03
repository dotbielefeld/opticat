import sys
import os
import uuid

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.nn.utils import clip_grad_norm_

from RLS.util.pool import ParamType, Configuration
from RLS.actors import Actor_GLU_BETA
from RLS.ac_util import initialize_weights, BatchMetrics, TrainingMetrics
from ppo_ac import PPO_AC



class PPO_AC_BETA(PPO_AC):

    def __init__(self, conf_space, state_dim, scenario):
        super().__init__(conf_space, state_dim, scenario)

        self.actor = Actor_GLU_BETA(state_dim, [len(p.bound) for p in self.cat_param], len(self.cont_param), scenario.num_blocks, scenario.num_neurons)

        if self.scenario.lr_schedule:
            self.optimizer = optim.Adam(self.actor.parameters(), lr=self.scenario.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.9)
        else:
            self.optimizer = optim.Adam(self.actor.parameters(), lr=self.scenario.lr)


    def init_weights(self, init):
        if init == "uniform":
            self.actor.apply(initialize_weights)
            for i, (alpha_head, beta_head) in enumerate(zip(self.actor.continuous_heads_alpha, self.actor.continuous_heads_beta)):
                nn.init.constant_(alpha_head.weight, 0.0)
                nn.init.constant_(beta_head.weight, 0.0)
                nn.init.constant_(alpha_head.bias, -5.0)
                nn.init.constant_(beta_head.bias, -5.0)
            for head in self.actor.categorical_heads:
                nn.init.constant_(head.weight, 0.0)
                nn.init.constant_(head.bias, 0.0)
        elif init == "xavier":
            self.actor.apply(initialize_weights)
        elif init == "none":
            pass

    def actions_to_conf(self, actions):
        # transfrom actions of actor to a configurations to be used on the target algorithm
        confs = []
        for action in actions:

            action_cat = action[ :len(self.cat_param)]
            action_cont = action[ len(self.cat_param):]

            conf = {}
            for cat_action, param in zip(action_cat, self.cat_param):
                conf[param.name] = param.bound[int(cat_action.item())]

            for cont_action, param in zip(action_cont, self.cont_param):
                parameter_range = param.bound
                scale = parameter_range[1] - parameter_range[0]
                shift = parameter_range[0]

                if param.scale == 'l':
                    log_range_min = torch.log10(torch.tensor(parameter_range[0])) if parameter_range[0] > 0 else torch.tensor(0.0)
                    log_range_max = torch.log10(torch.tensor(parameter_range[1]))
                    log_scaled_action = cont_action.item() * (log_range_max - log_range_min) + log_range_min
                    scaled_action = float(10 ** log_scaled_action)

                    if scaled_action > parameter_range[1]:
                        scaled_action = parameter_range[1]
                    elif scaled_action < parameter_range[0]:
                        scaled_action = parameter_range[0]

                else:
                    scaled_action = cont_action.item() * scale + shift

                conf[param.name] = scaled_action
                if param.type == ParamType.integer:
                    conf[param.name] = int(round(conf[param.name]))

            confs.append(Configuration(uuid.uuid4(), conf, "ppo" ))

        return confs

    def conf_to_actions(self, conf):
        # Given a conf the target algorithm we can also transform it into an action of the actor
        actions = []
        # Convert categorical parameters to actions
        for param in self.cat_param:
            value = conf[param.name]
            action = torch.tensor(param.bound.index(value))
            actions.append(action)

        # Convert continuous parameters to actions
        for param in self.cont_param:
            if param.name in conf.keys():
                value = conf[param.name]
            else:
                value = 0
            parameter_range = param.bound
            scale = parameter_range[1] - parameter_range[0]
            shift = parameter_range[0]

            if param.scale == 'l':
                log_range_min = torch.log10(torch.tensor(parameter_range[0])) if parameter_range[0] > 0 else torch.tensor(0.0)
                log_range_max = torch.log10(torch.tensor(parameter_range[1]))
                log_value = torch.log10(torch.tensor(value))
                action = (log_value - log_range_min) / (log_range_max - log_range_min)
            else:
                action = torch.tensor((value - shift) / scale)

            actions.append(action)
        return torch.stack(actions)


    def get_continuous_actions(self, continuous_alpha, continuous_beta):
        # sample contious actions from a beta dist
        actions = []
        log_probs = []
        for alpha, beta in zip(continuous_alpha, continuous_beta):

            dist = Beta(alpha.squeeze(1) + 1, beta.squeeze(1) + 1)
            action = dist.rsample()

            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return actions, log_probs

    def get_action_cat(self,cat_logits):
        # sample cat actions
        actions = []
        log_probs = []

        for logit in cat_logits:
            m = torch.distributions.Categorical(logits=logit)
            action = m.sample()

            log_prob = m.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return actions, log_probs

    def calculate_log_probs_for_actions(self, state, actions_taken):

        # split actions taken into cat and cont actions
        actions_taken_cat = actions_taken[:, :len(self.cat_param)]
        actions_taken_cont = actions_taken[:, len(self.cat_param):]

        # Handle cat actions
        state = state.squeeze(1)
        logits, continuous_alpha, continuous_beta = self.actor(state)

        if len(self.cat_param) > 0:
            log_probs_cat = []
            entropies_cat = []
            for logit, at, param in zip(logits, actions_taken_cat.transpose(0, 1), self.cat_param):
                m = torch.distributions.Categorical(logits=logit.squeeze(1))
                log_prob = m.log_prob(at)

                if len(param.bound) > 1:  # Check if there is more than one category
                    entropy = m.entropy()
                    max_entropy = torch.log(torch.tensor(float(len(param.bound))))
                    entropies_cat.append(entropy / max_entropy) # norm the enropy
                else:
                    # If only one category, entropy is zero
                    entropies_cat.append(torch.zeros_like(log_prob))

                log_probs_cat.append(log_prob)
            log_probs_cat = torch.stack(log_probs_cat, dim=1)
            entropies_cat = torch.stack(entropies_cat, dim=1)

        else:
            log_probs_cat = torch.tensor([])
            entropies_cat = torch.tensor([])

        if len(self.cont_param) > 0:
            log_probs_cont = []
            entropies_cont = []
            for alpha, beta, action_taken in zip(continuous_alpha, continuous_beta, actions_taken_cont.transpose(0, 1)):
                dist = Beta(alpha.squeeze(1) + 1, beta.squeeze(1) + 1)
                original_action = action_taken
                log_prob = dist.log_prob(original_action)
                entropy = torch.min(dist.entropy() / -5.2168, torch.tensor(1.0)) # normalise by entopy of  Beta(1, 500)
                log_prob_adjusted = log_prob
                log_probs_cont.append(log_prob_adjusted)
                entropies_cont.append(entropy)

            entropies_cont = torch.stack(entropies_cont, dim=1)
            log_probs_cont = torch.stack(log_probs_cont, dim=1)
        else:
            log_probs_cont = torch.tensor([])
            entropies_cont = torch.tensor([])

        log_probs = torch.cat((log_probs_cat, log_probs_cont), dim=1)
        return log_probs, entropies_cat, entropies_cont

    def get_conf(self, instance_features):

        cat_logits, continuous_alpha, continuous_beta = self.actor(instance_features)

        if len(self.cat_param) > 0:
            cat_actions, cat_log_probs = self.get_action_cat(cat_logits)
        else:
            cat_actions, cat_log_probs = torch.tensor([]), torch.tensor([])

        if len(self.cont_param) > 0:
            cont_actions, cont_log_probs = self.get_continuous_actions(continuous_alpha, continuous_beta)
        else:
            cont_actions, cont_log_probs = torch.tensor([]), torch.tensor([])

        actions = torch.cat((cat_actions, cont_actions), dim=1)
        log_probs = torch.cat((cat_log_probs, cont_log_probs), dim=1)
        return actions, log_probs

    def return_conf(self, states):

        states = torch.stack([torch.from_numpy(array) for array in states]).float()
        states = states.squeeze(1)

        with torch.no_grad():
            actions, old_log_probs = self.get_conf(states)

        confs = self.actions_to_conf(actions)

        self.actions_store[self.episode] = actions
        self.log_prob_store[self.episode] = old_log_probs
        self.states_store[self.episode] = states

        return confs

    def train(self, rewards, v):

        rewards = torch.stack([torch.tensor([reward], dtype=torch.float64) for reward in rewards])
        v = torch.stack([torch.tensor([vs], dtype=torch.float64) for vs in v])
        advantages = (rewards - v).squeeze()

        if self.scenario.relu_advantage and self.scenario.norm == "znorm":
          advantages = torch.clamp(advantages, min=-1e-6)

        # transform advantage to a matrix to perform normalization instancewise
        advantages = advantages.view(int(advantages.size(0)) // int(self.scenario.racesize), int(self.scenario.racesize))

        if self.scenario.norm == "znorm":
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-6)

        elif self.scenario.norm == "fnorm":
            advantages = F.normalize(advantages, dim=1, p=1)

        advantages = advantages.view(-1, 1).squeeze()

        if self.scenario.relu_advantage and self.scenario.norm == "fnorm":
          advantages = torch.clamp(advantages, min=-1e-6)

        self.advantage_store[self.episode] = advantages
        self.reward_store[self.episode] = rewards
        self.v_store[self.episode] = v

        training_metrics = TrainingMetrics()

        for e in range(self.ppo_epochs):
            states, actions, rewards, old_log_probs, v, advantage = self.shuffle_data(self.states_store[self.episode], self.actions_store[self.episode], self.reward_store[self.episode], self.log_prob_store[self.episode], self.v_store[self.episode], self.advantage_store[self.episode])

            batch_metrics = BatchMetrics()
            start_idx = 0

            for batch in range(int(np.ceil(states.shape[0]/self.batch_size))):
                states_batch, actions_batch, rewards_batch, old_log_probs_batch, v_batch , advantages_batch = self.get_batch(states, actions, rewards, old_log_probs, v,advantage, start_idx)
                start_idx += self.batch_size

                new_log_prob_s, entropies_cat ,entropies_cont = self.calculate_log_probs_for_actions(states_batch, actions_batch)
                entropies = torch.cat((entropies_cat, 1 - entropies_cont), dim=1)

                self.optimizer.zero_grad()
                ppo_loss, clip_fraction = self.ppo_loss_batch(
                    old_log_probs_batch,
                    new_log_prob_s,
                    advantages_batch,
                    self.epsilon,
                    entropies_cat,
                    entropies_cont
                )

                ppo_loss.backward()

                total_norm = 0
                for p in self.actor.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f'Epoch: {e}, Gradient Norm: {total_norm}')

                kl_divergence = self.compute_kl_divergence(old_log_probs_batch, new_log_prob_s)
                clip_grad_norm_(self.actor.parameters(), self.clip_value)

                self.optimizer.step()

                if self.scenario.lr_schedule:
                        self.scheduler.step()

                batch_metrics.update(
                    ppo_loss_val=ppo_loss.detach().numpy(),
                    clip_fraction_val=clip_fraction.detach().numpy(),
                    kl_divergence_val=kl_divergence.detach().numpy(),
                    entropy_val=entropies.mean().detach().numpy(),
                    gradnorm_val=total_norm
                )

            training_metrics.update_epoch(batch_metrics)

        self.episode = self.episode + 1

        final_metrics = training_metrics.compute_final_metrics()
        avg_loss, avg_cf, avg_kl, grad_norms, avg_en = final_metrics

        return avg_loss, avg_cf, avg_kl, grad_norms, avg_en

    def get_conf_prod(self, states):
        states = torch.stack([torch.from_numpy(array) for array in states]).float()

        states = states.squeeze(1)

        self.actor.eval()
        logits, continuous_alpha, continuous_beta = self.actor(states)

        # Handle categorical actions
        if len(self.cat_param) > 0:
            actions_cat = [torch.argmax(logit, dim=-1) for logit in logits]
            actions_cat = torch.stack(actions_cat, dim=1)
        else:
            actions_cat = torch.tensor([])

        # Handle continuous actions
        if len(self.cont_param) > 0:
            actions_cont = []
            for alpha, beta in zip(continuous_alpha, continuous_beta):
                # calc the mean of the dist to take as action
                action = (alpha.squeeze(1) + 1) / (alpha.squeeze(1) + beta.squeeze(1) + 2)
                actions_cont.append(action)
            actions_cont = torch.stack(actions_cont, dim=1)
        else:
            actions_cont = torch.tensor([])

        # Combine actions
        actions = torch.cat((actions_cat, actions_cont), dim=1)
        confs = self.actions_to_conf(actions)
        return confs

    def save_model(self, path):
        torch.save(self.actor, f'{path}')

