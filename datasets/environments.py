from gym_miniworld.miniworld import MiniWorldEnv
import math


class SimpleRectangularEnvironment(MiniWorldEnv):

    def __init__(self,
                 width, height,
                 obs_width=64, obs_height=64,
                 window_width=600, window_height=600):
        self.width = width
        self.height = height

        super().__init__(obs_width=obs_width, obs_height=obs_height,
                         window_width=window_width, window_height=window_height)
        
    def place_agent_at(self, pos, dir):
        pos = (pos[0], 0, pos[1])
        self.agent.pos = pos
        self.agent.dir = dir

    def _gen_world(self):
        room = self.add_rect_room(0, self.width, 0, self.height,
                                  floor_tex='asphalt', wall_tex='brick_wall', ceil_tex='concrete', no_ceiling=True)
        self.place_agent(room)
        
    def move(self, movement, rotation):
        self.step_count += 1
        self.turn_agent(rotation * 180 / math.pi)
        self.move_agent(movement, 0)
        obs = self.render_obs()
        return obs
