import numpy as np
import copy, itertools

class PSRS(object):
    # Rejection sampler that acts as an environment
    def __init__(self, buffer, nS=25, nA=5):
        self.raw_buffer = buffer # (s, a, r, s', done, p, info), where p is the logging policy's probability
        self.nS = nS
        self.nA = nA
        self._calculate_latent_state() # self.buffer contains (z, s, a, r, z', s', done, p, info)
        self.reset_sampler()
        self.reset()
    
    def _calculate_latent_state(self):
        self.buffer = [(info['z'], s, a, r, info['next_z'], s_, done, p, info) for s, a, r, s_, done, p, info in self.raw_buffer]
    
    def reset_sampler(self, seed=None):
        self.init_queue = [(elt[0], elt[1]) for elt in self.buffer if elt[-1]['t'] == 0]
        np.random.default_rng(seed=seed).shuffle(self.init_queue)
        
        # Construct queues based on the latent state of from-state
        self.queues = {k:list(v) for k,v in itertools.groupby(sorted(self.buffer, key=lambda elt:elt[0]), key=lambda elt:elt[0])}
        
        # Shuffle queues
        for z in self.queues.keys():
            np.random.default_rng(seed=seed).shuffle(self.queues[z])
    
    def reset(self, seed=None):
        self.rejection_sampling_rng = np.random.default_rng(seed=seed)
        if len(self.init_queue) == 0:
            self.s = None
            return None
        self.z, self.s = self.init_queue.pop(0)
        return self.s
    
    def step(self, p_new):
        s = self.s
        z = self.z
        reject = True
        while reject:
            if len(self.queues[z]) == 0:
                return None, None, None, None
            _z, _s, a, r, z_next, s_next, done, p_log, info = self.queues[z].pop(0)
            assert z == _z
            M = (p_new / p_log).max()
            u = self.rejection_sampling_rng.random()
            if u <= p_new[a] / p_log[a] / M:
                reject = False
        self.s = s_next
        self.z = z_next
        return s_next, r, done, {'z': z, 'a': a, 'p': p_log}

class PSRS_Exo(object):
    # Rejection sampler that acts as an environment
    # o: observation
    # s: latent state
    # x: exogenous state
    def __init__(self, buffer, nO=25, nA=5, o_split_func=lambda o: (o,0), o_combine_func=lambda s,x: s):
        self.raw_buffer = copy.deepcopy(buffer) # (o, a, r, o', done, p), where p is the logging policy's probability
        self.nS = self.nO = nO
        self.nA = nA
        self.o_split_func = o_split_func
        self.o_combine_func = o_combine_func
        self._calculate_latent_state() # self.buffer contains ((s, x), a, r, (s', x'), done, p)
        self.reset_sampler()
        self.reset()
    
    def _calculate_latent_state(self):
        self.buffer = [(self.o_split_func(o), a, r, self.o_split_func(o_), *rest) for o, a, r, o_, *rest in self.raw_buffer]
    
    def reset_sampler(self, seed=None):
        self.init_queue = [elt[0] for elt in self.raw_buffer if elt[-1]['t'] == 0]
        np.random.default_rng(seed=seed).shuffle(self.init_queue)
        
        # Construct queues based on the latent state of from-state
        self.s_queues = {k:[(s, a, r, s_, *rest) for ((s, _), a, r, (s_, _), *rest) in v] 
                         for k,v in itertools.groupby(sorted(self.buffer, key=lambda elt:elt[0][0]), key=lambda elt:elt[0][0])}
        self.x_queues = {k:[(x, x_) for ((_, x), _, _, (_, x_), *_) in v] for k,v in itertools.groupby(sorted(self.buffer, key=lambda elt:elt[0][1]), key=lambda elt:elt[0][1])}
        
        # Shuffle queues
        for s in self.s_queues.keys():
            np.random.default_rng(seed=seed).shuffle(self.s_queues[s])
        for x in self.x_queues.keys():
            np.random.default_rng(seed=seed).shuffle(self.x_queues[x])
    
    def reset(self, seed=None):
        self.rejection_sampling_rng = np.random.default_rng(seed=seed)
        if len(self.init_queue) == 0:
            self.s = None
            return None
        self.o = self.init_queue.pop(0)
        return self.o
    
    def step(self, p_new):
        o = self.o
        reject = True
        while reject:
            s, x = self.o_split_func(o)
            if len(self.s_queues[s]) == 0:
                return None, None, None, None
            if len(self.x_queues[x]) == 0:
                return None, None, None, None
            _s, a, r, s_next, done, p_log, info = self.s_queues[s].pop(0)
            _x, x_next = self.x_queues[x].pop(0)
            assert s == _s
            assert x == _x
            M = (p_new / p_log).max()
            u = self.rejection_sampling_rng.random()
            if u <= p_new[a] / p_log[a] / M:
                reject = False
        self.o = self.o_combine_func(s_next, x_next)
        return self.o, r, done, {'s': s, 'a': a, 'p': p_log}

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
        if S is None:
            terminate = True
            break
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
        if S is None:
            terminate = True
            break
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
        if S is None:
            terminate = True
            break
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
