from gym.envs.registration import register

register(
    id='VisualSim-v0',
    entry_point='visual_sim.envs:VisualSim',
)