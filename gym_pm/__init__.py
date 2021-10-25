import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Rail-v0',
    entry_point='gym_pm.envs:Rail_Env'
)

register(
    id='Assembly-v0',
    entry_point='gym_pm.envs:Assembly_Env'
)