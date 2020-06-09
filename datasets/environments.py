from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS
import math
from copy import deepcopy

params = deepcopy(DEFAULT_PARAMS)
params.set('tex_rand', 1, 1, 4)


class SimpleRectangularEnvironment(MiniWorldEnv):

    def __init__(self,
                 width, height,
                 obs_width=64, obs_height=64,
                 window_width=600, window_height=600,
                 domain_rand=True):
        self.width = width
        self.height = height

        super().__init__(obs_width=obs_width, obs_height=obs_height,
                         window_width=window_width, window_height=window_height,
                         domain_rand=domain_rand, params=params)
        
    def place_agent_at(self, pos, dir):
        pos = (pos[0], 0, pos[1])
        self.agent.pos = pos
        self.agent.dir = dir

    def _gen_world(self):
        room = self.add_rect_room(0, self.width, 0, self.height,
                                  floor_tex='asphalt', wall_tex='concrete', ceil_tex='concrete', no_ceiling=True)
        self.place_agent(room)
        
    def move(self, movement, rotation):
        self.step_count += 1
        self.turn_agent(rotation * 180 / math.pi)
        self.move_agent(movement, 0)
        obs = self.render_obs()
        return obs
