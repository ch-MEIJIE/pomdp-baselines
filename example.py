import gymnasium as gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu

# import environments
# from envs.pomdp.wrappers import POMDPWrapper

import PyFlyt.gym_envs  # noqa
from env_wrapper import PyFlytEnvWrapper
# import recurrent model-free RL (separate architecture)
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN

# import the replay buffer
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from utils import helpers as utl
from torch.nn import functional as F


@torch.no_grad()
def collect_rollouts(
    num_rollouts, random_actions=False, deterministic=False, train_mode=True, policy_storage=None
):
    """collect num_rollouts of trajectories in task and save into policy buffer
    :param
        random_actions: whether to use policy to sample actions, or randomly sample action space
        deterministic: deterministic action selection?
        train_mode: whether to train (stored to buffer) or test
    """
    if not train_mode:
        assert random_actions is False and deterministic is True

    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        # get hidden state at timestep=0, None for mlp
        action, reward, internal_state = agent.get_initial_info()

        if train_mode:
            # temporary storage
            obs_list, act_list, rew_list, next_obs_list, term_list = (
                [],
                [],
                [],
                [],
                [],
            )

        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor([env.action_space.sample()])
                action = F.one_hot(
                    action.squeeze(-1).long(), num_classes=9
                    ).float()  # (1, A)
                # extend a dim for action
                action = action.unsqueeze(0)
            else:
                # policy takes hidden state as input for rnn, while takes obs for mlp
                (action, _, _, _), internal_state = agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                )
            # observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(
                env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            # update statistics
            steps += 1
            rewards += reward.item()

            # early stopping env: such as rmdp, pomdp, generalize tasks. term ignores timeout
            term = (
                False
                if "TimeLimit.truncated" in info or steps >= max_trajectory_len
                else done_rollout
            )

            if train_mode:
                # append tensors to temporary storage
                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

            # set: obs <- next_obs
            obs = next_obs.clone()

        if train_mode:
            # add collected sequence to buffer
            policy_storage.add_episode(
                observations=ptu.get_numpy(
                    torch.cat(obs_list, dim=0)),  # (L, dim)
                actions=ptu.get_numpy(torch.cat(act_list, dim=0)),  # (L, dim)
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                next_observations=ptu.get_numpy(
                    torch.cat(next_obs_list, dim=0)
                ),  # (L, dim)
            )
        print(
            "Mode:",
            "Train" if train_mode else "Test",
            "env_steps",
            steps,
            "total rewards",
            rewards,
        )
        total_steps += steps
        total_rewards += rewards

    if train_mode:
        return total_steps
    else:
        return total_rewards / num_rollouts


def update(num_updates, policy_storage):
    rl_losses_agg = {}
    # print(num_updates)
    for update in range(num_updates):
        # sample random RL batch: in transitions
        batch = ptu.np_to_pytorch_batch(
            policy_storage.random_episodes(batch_size))
        # RL update
        rl_losses = agent.update(batch)

        for k, v in rl_losses.items():
            if update == 0:  # first iterate - create list
                rl_losses_agg[k] = [v]
            else:  # append values
                rl_losses_agg[k].append(v)
    # statistics
    for k in rl_losses_agg:
        rl_losses_agg[k] = np.mean(rl_losses_agg[k])
    return rl_losses_agg


if __name__ == "__main__":
    # set gpu
    num_env = 1
    cuda_id = 0  # -1 if using cpu
    ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

    # env_name = "Pendulum-v1"
    # env = gym.make(env_name)
    # env = POMDPWrapper(env, partially_obs_dims=[2])
    env = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-Velocity-Gates-Asyn_v1",
        seed=1
    )

    max_trajectory_len = 200
    # act_dim = env.action_space.shape[0]
    # obs_dim = env.observation_space.shape[0]
    act_dim = 9
    act_classes = 9
    obs_dim = env.obs_atti_size + env.obs_target_size + env.obs_bound_size + 1
    print(env, obs_dim, act_dim, max_trajectory_len)

    agent = Policy_RNN(
        obs_dim=obs_dim,
        action_dim=act_dim,
        action_classes=act_classes,
        encoder="lstm",
        algo_name="sacd",
        action_embedding_size=8,
        observ_embedding_size=32,
        reward_embedding_size=8,
        rnn_hidden_size=128,
        dqn_layers=[128, 128],
        policy_layers=[128, 128],
        lr=0.0003,
        gamma=0.9,
        tau=0.005,
    ).to(ptu.device)

    num_updates_per_iter = 1.0  # training frequency
    sampled_seq_len = 64  # context length
    buffer_size = 1e6
    batch_size = 32

    num_iters = 150
    num_init_rollouts_pool = 5
    num_rollouts_per_iter = 1
    total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
    n_env_steps_total = max_trajectory_len * total_rollouts
    _n_env_steps_total = 0
    print("total env episodes", total_rollouts,
          "total env steps", n_env_steps_total)

    policy_storage = SeqReplayBuffer(
        max_replay_buffer_size=int(buffer_size),
        observation_dim=obs_dim,
        action_dim=act_dim,
        sampled_seq_len=sampled_seq_len,
        sample_weight_baseline=0.0,
    )

    env_steps = collect_rollouts(
        num_rollouts=num_init_rollouts_pool,
        random_actions=True,
        train_mode=True,
        policy_storage=policy_storage
    )
    _n_env_steps_total += env_steps

    # evaluation parameters
    last_eval_num_iters = 0
    log_interval = 5
    eval_num_rollouts = 10
    learning_curve = {
        "x": [],
        "y": [],
    }

    while _n_env_steps_total < n_env_steps_total:

        env_steps = collect_rollouts(
            num_rollouts=num_rollouts_per_iter, train_mode=True, policy_storage=policy_storage)
        _n_env_steps_total += env_steps

        train_stats = update(
            int(num_updates_per_iter * env_steps),
            policy_storage=policy_storage
        )

        current_num_iters = _n_env_steps_total // (
            num_rollouts_per_iter * max_trajectory_len
        )
        if (
            current_num_iters != last_eval_num_iters
            and current_num_iters % log_interval == 0
        ):
            last_eval_num_iters = current_num_iters
            average_returns = collect_rollouts(
                num_rollouts=eval_num_rollouts,
                train_mode=False,
                random_actions=False,
                deterministic=True,
                policy_storage=policy_storage,
            )
            learning_curve["x"].append(_n_env_steps_total)
            learning_curve["y"].append(average_returns)
            print(_n_env_steps_total, average_returns)
