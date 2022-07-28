import numpy as np
from tqdm import tqdm

def _random_argmax(x):
    return np.random.choice(np.where(x == np.max(x))[0])

def uniformly_random_policy(Q, args):
    pi = np.ones_like(Q) / Q.shape[1]
    return pi

def greedy_policy(Q, args):
    pi = np.zeros_like(Q)
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1
    return pi

def soft_greedy_policy(Q, args):
    pi = np.zeros_like(Q)
    for s in range(len(Q)):
        pi[s, np.where(np.isclose(Q[s], np.max(Q[s])))[0]] = 1
    return pi / pi.sum(axis=1, keepdims=True)

def epsilon_greedy_policy(Q, args):
    #  ε : explore
    # 1-ε: exploit
    epsilon = args['epsilon']
    pi = np.ones_like(Q) * epsilon / (Q.shape[1])
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1 - epsilon + epsilon / (Q.shape[1])
    return pi

def qlearn(
    env, n_episodes, behavior_policy, gamma, alpha=0.1, epsilon=1.0, 
    Q_init=None, memory=None, save_Q=0, show_tqdm=True, seed=None,
):
    rng = np.random.default_rng(seed=seed)
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
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            if save_Q: Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {
            'Qs': np.array(Qs), 
            'TD_errors': np.array(TD_errors),
            'memory': memory,
        }
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    memory_buffer = []
    
    for episode in tqdm(range(n_episodes), disable=(not show_tqdm)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        S = env.reset()
        done = False
        while not done: # S is not a terminal state
            p = behavior_policy(Q[[S],:], dict(epsilon=epsilon_func(episode)))[0]
            A = rng.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            memory_buffer.append((S, A, R, S_, done, p, info))
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), # history of all Q-values per update
        'TD_errors': np.array(TD_errors), # temporal difference error for each update
        'memory': memory_buffer, # all trajectories/experience, tuples of (s,a,r,s', done)
    }

def expSARSA(
    env, n_episodes, pi, gamma, alpha=0.1, 
    Q_init=None, memory=None, save_Q=0, show_tqdm=True, seed=None,
):
    rng = np.random.default_rng(seed=seed)
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((env.nS, env.nA))
    else:
        Q = Q_init.copy().astype(float)
    
    if memory is not None:
        raise NotImplementedError
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    memory_buffer = []
    
    for episode in tqdm(range(n_episodes), disable=(not show_tqdm)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        S = env.reset()
        done = False
        while not done: # S is not a terminal state
            p = pi[S]
            A = rng.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            memory_buffer.append((S, A, R, S_, done, p, info))
            TD_errors.append(R + gamma * (Q[S_] @ pi[S_]) - Q[S,A])
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (Q[S_] @ pi[S_]) - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), # history of all Q-values per update
        'TD_errors': np.array(TD_errors), # temporal difference error for each update
        'memory': memory_buffer, # all trajectories/experience, tuples of (s,a,r,s', done)
    }

def z_expSARSA(
    env, n_episodes, pi, gamma, alpha=0.1, 
    Q_init=None, memory=None, save_Q=0, show_tqdm=True, seed=None, latent_state_func=None,
):
    rng = np.random.default_rng(seed=seed)
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((env.nS, env.nA))
    else:
        Q = Q_init.copy().astype(float)
    
    if memory is not None:
        raise NotImplementedError
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    memory_buffer = []
    
    for episode in tqdm(range(n_episodes), disable=(not show_tqdm)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        S = env.reset()
        done = False
        while not done: # S is not a terminal state
            z = latent_state_func(S)
            p = pi[z]
            A = rng.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            z_ = latent_state_func(S_)
            memory_buffer.append((S, A, R, S_, done, p, info))
            TD_errors.append(R + gamma * Q[z_].max() - Q[z,A])
            
            # Perform update
            Q[z,A] = Q[z,A] + alpha(episode) * (R + gamma * (Q[z_] @ pi[z_]) - Q[z,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), # history of all Q-values per update
        'TD_errors': np.array(TD_errors), # temporal difference error for each update
        'memory': memory_buffer, # all trajectories/experience, tuples of (s,a,r,s', done)
    }

def evalMC(env, n_episodes, pi, gamma, seed=None):
    rng = np.random.default_rng(seed=seed)
    Gs = []
    lengths = []
    episode = 0
    while episode < n_episodes:
        G = 0
        t = 0
        S = env.reset(seed=episode)
        done = False
        while not done: # S is not a terminal state
            p = pi[S]
            A = rng.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
        
        lengths.append(t)
        if done:
            Gs.append(G)
            episode += 1
    
    return np.array(Gs), np.array(lengths)
