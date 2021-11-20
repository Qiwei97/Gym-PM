import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Rail-v1',
    entry_point='gym_pm.envs:Rail_Env'
)

register(
    id='Rail-v2',
    entry_point='gym_pm.envs:Railv2_Env'
)

register(
    id='Assembly-v1',
    entry_point='gym_pm.envs:Assembly_Env'
)