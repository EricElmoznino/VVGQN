from gym_miniworld.miniworld import MiniWorldEnv
import os
import contextlib


class SimpleRectangularEnvironment(MiniWorldEnv):

    def __init__(self,
                 width, height,
                 obs_width=60, obs_height=60,
                 window_width=600, window_height=600):
        self.width = width
        self.height = height
        self.start_pos = None
        self.start_dir = None

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            super().__init__(obs_width=obs_width, obs_height=obs_height,
                             window_width=window_width, window_height=window_height)

    def reset_state(self, start_pos, start_dir):
        self.start_pos = (start_pos[0], 0, start_pos[1])
        self.start_dir = start_dir
        return super().reset()
        
    def place_agent(self, room, pos, dir):
        return self.place_entity(self.agent, room=room, pos=pos, dir=dir)

    def _gen_world(self):
        room = self.add_rect_room(0, self.width, 0, self.height,
                                  floor_tex='asphalt', wall_tex='brick_wall', ceil_tex='concrete', no_ceiling=True)
        self.place_agent(room, pos=self.start_pos, dir=self.start_dir)
        
    def step(self, movement, rotation):
        assert self.start_pos is not None and self.start_dir is not None
        self.step_count += 1
        self.turn_agent(rotation)
        self.move_agent(movement, 0)
        obs = self.render_obs()
        return obs
