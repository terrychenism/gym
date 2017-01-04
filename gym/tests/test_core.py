import gym
from gym import core
from gym.envs import register


class ArgumentEnv(core.Env):
    calls = 0

    def __init__(self, arg):
        self.calls += 1
        self.arg = arg

def test_env_instantiation():
    # This looks like a pretty trivial test, but given our usage of
    # __new__, it's worth having.
    env = ArgumentEnv('arg')
    assert env.arg == 'arg'
    assert env.calls == 1


register(
    id='test.StepsLimitCartpole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=2,
    )

def test_steps_limit_restart():
    env = gym.make('test.StepsLimitCartpole-v0')
    env.reset()

    # Episode has started
    _, _, done, info = env.step(env.action_space.sample())
    assert done == False

    # Limit reached, now we get a done signal and the env resets itself
    _, _, done, info = env.step(env.action_space.sample())
    assert done == True
    assert env._monitor.episode_id == 1

    env.close()
