import gym
import numpy as np
from spinup.algos.pytorch.ppo import core
from spinup.algos.pytorch.ppo.ppo import PPOBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import time
import torch
from torch.optim import Adam

from offsim4rl.agents.agent import AdaptiveAgent
from offsim4rl.data import ProbDistribution


class PPOAgent(AdaptiveAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        logger=EpochLogger(),
        seed=0,
        steps_per_epoch=1000,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        save_freq=10,
        validate=False,
        val_kwargs=dict()
    ):
        super().__init__(observation_space, action_space, actor_critic, ac_kwargs, seed)

        setup_pytorch_for_mpi()
        self.logger = logger
        self.logger.save_config(locals())

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq

        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape

        # Sync params across processes
        sync_params(self.model)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.model.pi, self.model.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.model.v.parameters(), lr=vf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.model)

        self.epochs = 0
        self.steps = 0
        self.start_time = time.time()

    @property
    def action_dist_type(self):
        return ProbDistribution.Discrete

    def begin_episode(self, observation):
        self.obs = observation
        self.ep_ret = 0
        self.ep_len = 0
        a, _, logp = self.model.step(torch.as_tensor(self.obs, dtype=torch.float32))
        return a, logp

    def commit_action(self, action):
        pass

    def step(self, reward, next_observation, terminated, truncated):
        a, v, logp = self.model.step(torch.as_tensor(self.obs, dtype=torch.float32))
        self.ep_ret += reward
        self.ep_len += 1
        self.buf.store(self.obs, a, reward, v, logp)
        self.logger.store(VVals=v)

        self.obs = next_observation
        timeout = truncated or self.ep_len == self.max_ep_len
        terminal = terminated or timeout

        epoch_ended = self.steps == self.local_steps_per_epoch - 1
        self.steps += 1
        if not terminal and not epoch_ended:
            return a, logp

        # terminal or epoch_ended
        if epoch_ended and not terminal:
            print('Warning: trajectory cut off by epoch at %d steps.'% self.ep_len, flush=True)

        # if trajectory didn't reach terminal state, bootstrap value target
        if timeout or epoch_ended:
            _, v, _ = self.model.step(torch.as_tensor(self.obs, dtype=torch.float32))
        else:
            v = 0
        self.buf.finish_path(v)

        if terminal:
            # only save EpRet / EpLen if trajectory finished
            self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)

        if epoch_ended:
            self.adapt()

            # Log info about epoch
            self.logger.log_tabular('Epoch', self.epochs)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (self.epochs + 1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - self.start_time)
            self.logger.dump_tabular()
            self.epochs += 1

            if (self.epochs % self.save_freq == 0):
                logger.save_state({}, self.epochs)

            self.steps = 0

        return a, logp

    def adapt(self):
        transitions = self.buf.get()

        pi_l_old, pi_info_old = self._compute_loss_pi(transitions)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(transitions).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(transitions)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.model.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(transitions)
            loss_v.backward()
            mpi_avg_grads(self.model.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old)
        )

    def _compute_loss_pi(self, transitions):
        obs, act, adv, logp_old = transitions["obs"], transitions["act"], transitions["adv"], transitions["logp"]

        # Policy loss
        pi, logp = self.model.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def _compute_loss_v(self, transitions):
        obs, ret = transitions['obs'], transitions['ret']
        return ((self.model.v(obs) - ret) ** 2).mean()


if __name__ == '__main__':
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('CartPole-v1_ppo_new', 0, data_dir='./outputs')
    logger = EpochLogger(**logger_kwargs)
    env = gym.make('CartPole-v1')
    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        logger=logger,
        seed=0,
    )

    mpi_fork(1)  # run parallel code with mpi
    num_interactions = 50000
    obs, _ = env.reset()
    a, logp = agent.begin_episode(obs)
    for t in range(num_interactions):
        obs, r, terminated, truncated, _ = env.step(a)
        a, logp = agent.step(r, obs, terminated, truncated)

        if terminated or truncated:
            obs, _ = env.reset()
            a, logp = agent.begin_episode(obs)
