import PyFlyt.gym_envs  # noqa
import gymnasium as gym
import numpy as np


class PyFlytEnvWrapper:
    def __init__(
        self,
        render_mode: str = "human",
        env_id: str = "PyFlyt/QuadX-UVRZ-Gates-v2",
        seed: int = 0
    ) -> None:
        self.env = gym.make(
            env_id,
            render_mode=render_mode,
            seed=seed,
            agent_hz=2
        )
        self.targets_num = self.env.unwrapped.targets_num
        self.act_size = self.env.action_space.n
        self.obs_atti_size = self.env.observation_space['attitude'].shape[0]
        self.obs_target_size = \
            self.env.observation_space['target_deltas'].feature_space.shape[0]

        # TODO: Flatten the target delta bound space in ENV
        self.obs_bound_size = \
            self.env.observation_space["target_delta_bound"].shape[0]
        
        self.action_space = self.env.action_space
        self.state_updated = np.zeros((1,))

    def reset(self):
        obs, _ = self.env.reset()
        self.state_atti = obs['attitude']
        self.state_targ = obs['target_deltas'][0]
        self.state_bound = obs['target_delta_bound']
        self.state_updated[0] = obs['updated']

        obs = self.concat_state()

        return obs

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        self.state_atti = obs['attitude']
        # For getting a unifed observation space, we pad the target deltas
        self.state_targ = obs['target_deltas'][0]
        self.state_bound = obs['target_delta_bound']
        self.state_updated[0] = obs['updated']

        obs = self.concat_state()
        done = term or trunc

        return obs, reward, done, info

    def concat_state(self):
        return np.concatenate(
            [self.state_atti, self.state_targ, self.state_bound, self.state_updated]
        )
