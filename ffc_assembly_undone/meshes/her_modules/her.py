import numpy as np
import math

class her_sampler:
    def __init__(self, replay_strategy, replay_k,gamma, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.gamma = gamma
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        #print(T,rollout_batch_size)
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        transitions["epr"]=np.zeros((batch_size,1))
        # to get the params to re-compute reward

        for ind,(i,j) in enumerate(zip(episode_idxs,t_samples)):
            for k in range(j,T):
                r=self.reward_func(episode_batch["ag_next"][i][k],transitions["g"][ind],None)+1
                if r>0:
                    transitions["epr"][ind,0]=pow(self.gamma,k-j)
                    break

        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
