from RLS.util.pool import ParamType
import torch
import random

class PPO_AC:

    def __init__(self, conf_space, state_dim, scenario):
        self.conf_space = conf_space
        self.state_dim = state_dim

        self.process_confspace()
        self.epsilon = 0.2
        self.ppo_epochs = scenario.ppo_epochs
        self.episode = 0
        self.batch_size = scenario.batchsize
        self.clip_value = scenario.clip_value

        self.ec = scenario.ec

        self.actions_store = {}
        self.states_store = {}
        self.reward_store = {}
        self.v_store = {}
        self.log_prob_store = {}
        self.advantage_store = {}

        self.epoch_rewards = []
        self.scenario = scenario


    def process_confspace(self):
        self.cat_param = []
        self.cont_param = []
        for param in self.conf_space:
            if param.type == ParamType.categorical:
                self.cat_param.append(param)
            elif param.type == ParamType.continuous:
                self.cont_param.append(param)
            elif param.type == ParamType.integer:
                self.cont_param.append(param)
            else:
                raise ValueError

    def compute_kl_divergence(self,old_log_probs, new_log_probs):
        # Compute the probabilities from log probabilities
        old_probs = torch.exp(old_log_probs)
        # Compute the KL divergence
        kl_div = old_probs * (old_log_probs - new_log_probs)
        return kl_div.mean()

    def shuffle_data(self,states, actions, rewards, old_log_probs, v, advantage):
        combined = list(zip(states, actions, rewards, old_log_probs, v, advantage))
        random.shuffle(combined)

        states, actions, rewards, old_log_probs, v, advantage = zip(*combined)

        states = torch.stack(list(states))
        actions = torch.stack(list(actions))
        rewards = torch.stack(list(rewards))
        old_log_probs = torch.stack(list(old_log_probs))
        v = torch.stack(list(v))
        advantage = torch.stack(list(advantage))

        return states, actions, rewards, old_log_probs, v, advantage

    def ppo_loss_batch(self, old_prob_log, new_prob_log, advantage, epsilon, entropies_cat, entropies_cont):

        #old_prob_log = old_prob_log.sum(dim=1)
        #new_prob_log = new_prob_log.sum(dim=1)
        # I am computing the loss for every head and then take the mean over that
        # Computing loss per head and averaging: If your action heads represent independent or separable actions
        # (e.g., different dimensions of an action space or categorical/continuous action spaces),
        # you might want to compute the loss for each head separately and then average them. T
        # his would make sense if each head represents a
        # distinct decision-making process that is somewhat independent of the others.

       # print(new_prob_log.shape, old_prob_log.shape)
        ratio = torch.exp(new_prob_log - old_prob_log)
        advantage = advantage.unsqueeze(1)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        if self.scenario.entropy_loss:
            cat_entropy_loss = -self.ec * entropies_cat.mean()
            cont_entropy_loss = self.ec *  entropies_cont.mean()

            entropy_loss = cat_entropy_loss + cont_entropy_loss
            loss = policy_loss + entropy_loss
        else:
            loss = policy_loss

        clip_fraction = ((ratio < (1 - epsilon)) | (ratio > (1 + epsilon))).float().mean()

        return loss, clip_fraction

    def get_batch(self, states, actions, rewards, old_log_probs, v,advantage, start_idx):

        states = states[start_idx : start_idx + self.batch_size]
        actions = actions[start_idx: start_idx + self.batch_size]
        rewards = rewards[start_idx: start_idx + self.batch_size]
        old_log_probs = old_log_probs[start_idx: start_idx + self.batch_size]
        v = v[start_idx: start_idx + self.batch_size]
        advantage = advantage[start_idx: start_idx + self.batch_size]

        return states, actions, rewards, old_log_probs, v, advantage

    def save_model(self, path):
        torch.save(self.actor, f'{path}')
