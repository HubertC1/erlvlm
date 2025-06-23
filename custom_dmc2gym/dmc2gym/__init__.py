import gym
from gym.envs.registration import register
from typing import Optional


class TimeLimit(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        else:
            truncated = False
        done = terminated or truncated

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        obs = self.env.reset(**kwargs)
        return obs


def make(
        domain_name,
        task_name,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=True
):
    env_id = 'dmc_%s_%s_%s-v1' % (domain_name, task_name, seed)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='dmc2gym.wrappers:Custom_DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=None,
        )
    env = gym.make(env_id, disable_env_checker=True)
    env = TimeLimit(env, max_episode_steps)
    return env
