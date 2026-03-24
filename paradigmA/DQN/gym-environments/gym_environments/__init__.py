from gymnasium.envs.registration import register

register(
    id="GraphEnv-v1",
    entry_point="gym_environments.envs.environment1:Env1",
)
