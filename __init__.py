# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

register(
    id="rl_mm/SO101-v0",
    entry_point="rl_mm.envs:SO101Arm",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
register(
    id="rl_mm/SO101-v1",
    entry_point="rl_mm.envs:SO101Arm2",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
