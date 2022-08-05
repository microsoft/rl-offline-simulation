import numpy as np
import copy, itertools

from offsim4rl.core import RevealedRandomnessEnv

class PerStateRejectionSampling(RevealedRandomnessEnv):
    # Rejection sampler that acts as an environment
    def __init__(self, buffer, nS=25, nA=5, latent_state_func=lambda s: s):
        self.raw_buffer = copy.deepcopy(buffer) # (s, a, r, s', done, p), where p is the logging policy's probability
        self.nS = nS
        self.nA = nA
        self.latent_state_func = latent_state_func
        self._calculate_latent_state() # (z, s, a, r, s', done, p)
        self.reset_sampler()
        self.reset()
    
    def _calculate_latent_state(self):
        try:
            self.buffer = [(self.latent_state_func(s), s, *rest) for s, *rest in self.raw_buffer]
        except:
            self.buffer = [(self.latent_state_func(info), s, a, r, s_, done, p, info) for s, a, r, s_, done, p, info in self.raw_buffer]
    
    def reset_sampler(self, seed=None):
        # Construct queues based on the latent state of from-state
        self.queues = {k:list(v) for k,v in itertools.groupby(sorted(self.buffer, key=lambda elt:elt[0]), key=lambda elt: elt[0])}
        
        # Shuffle queues
        for z in self.queues.keys():
            np.random.default_rng(seed=seed).shuffle(self.queues[z])
    
    def reset(self, seed=None):
        self.rejection_sampling_rng = np.random.default_rng(seed=seed)
        self.s = 0 # assume known starting state
        return self.s
    
    def step(self, action):
        raise NotImplementedError(
            'PerStateRejectionSampling requires the agent to reveal its randomness via the step_dist method. '
            + 'Modify your agent or use the QueueBased simulator instead.')

    def step_dist(self, p_new):
        s = self.s
        reject = True
        while reject:
            z = self.latent_state_func(s)
            if len(self.queues[z]) == 0:
                return None, None, None, None
            _z, _s, a, r, s_next, done, p_log, info = self.queues[z].pop(0)
            assert z == _z
            M = (p_new / p_log).max()
            u = self.rejection_sampling_rng.random()
            if u <= p_new[a] / p_log[a] / M:
                reject = False
        self.s = s_next
        return s_next, r, done, {'z': z, 'a': a, 'p': p_log}

def qlearn_psrs(
    env, n_episodes, behavior_policy, gamma, alpha=0.1, epsilon=1.0, 
    Q_init=None, save_Q=0,
):
    if not callable(epsilon):
        epsilon_func = lambda episode: epsilon
    else:
        epsilon_func = epsilon
    
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((env.nS, env.nA))
    else:
        Q = Q_init.copy().astype(float)
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    memory_buffer = []
    episode = 0
    terminate = False
    while episode < n_episodes and not terminate:
        G = 0
        t = 0
        S = env.reset(seed=episode)
        done = False
        while not done: # S is not a terminal state
            p = behavior_policy(Q[[S],:], dict(epsilon=epsilon_func(episode)))[0]
            S_, R, done, info = env.step(p)
            if S_ is None:
                terminate = True
                break
            A = info['a']
            memory_buffer.append((S, A, R, S_, done, p, info))
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        
        # once episode is done
        Gs.append(G)
        episode += 1
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), 
        'TD_errors': np.array(TD_errors),
        'memory': memory_buffer,
    }

def expSARSA_psrs(
    env, n_episodes, pi, gamma, alpha=0.1, 
    Q_init=None, save_Q=0,
):
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((env.nS, env.nA))
    else:
        Q = Q_init.copy().astype(float)
    
    Gs = []
    Qs = [Q.copy()]
    memory_buffer = []
    episode = 0
    terminate = False
    while episode < n_episodes and not terminate:
        G = 0
        t = 0
        S = env.reset(seed=episode)
        done = False
        while not done: # S is not a terminal state
            p = pi[S]
            S_, R, done, info = env.step(p)
            if S_ is None:
                terminate = True
                break
            A = info['a']
            memory_buffer.append((S, A, R, S_, done, p, info))
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (Q[S_] @ pi[S_]) - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        
        # once episode is done
        Gs.append(G)
        episode += 1
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), 
        'memory': memory_buffer,
    }

def evalMC_psrs(env, n_episodes, pi, gamma):
    Gs = []
    lengths = []
    episode = 0
    terminate = False
    while episode < n_episodes and not terminate:
        G = 0
        t = 0
        S = env.reset(seed=episode)
        done = False
        while not done: # S is not a terminal state
            p = pi[S]
            S_, R, done, info = env.step(p)
            if S_ is None:
                terminate = True
                break
            A = info['a']
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
        
        lengths.append(t)
        if done: 
            assert not terminate
            Gs.append(G)
            episode += 1
    
    return np.array(Gs), np.array(lengths) # cumulative reward for each episode
