import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PM-v0',
    entry_point='gym_pm.envs:PM_Env'
)