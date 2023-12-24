from typing import Any
import random
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from twenty48.Board import Board


class twenty48Env(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self._reset_state = Board.TextRepresentation("00000000000000000000000000000000", 0).get_board_from_text_rep()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3)
        self._observation_spec = array_spec.BoundedArraySpec(shape=(16,), dtype=np.int32, minimum=0)
        self._state = self._reset_state.copy().set_random_square().set_random_square()
        self._prev_state = None
        self._episode_ended = False

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_state(self) -> Board:
        return self._state

    def set_state(self, state: Board) -> None:
        self._state = state

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        # TODO: only allow policy to select legal moves
        # self._state.display()
        if self._episode_ended:
            return self.reset()

        if not self._state.check_legal_move(Board.Move(action)):
            action = random.choices(self._state.get_legal_moves(), k=1)[0]

        self._prev_state = self._state.copy()
        self._state.move(Board.Move(action)).set_random_square()

        reward = self._state.get_score() - self._prev_state.get_score()
        if not self._state.get_legal_moves():
            self._episode_ended = True
            # print('gameover')

            return ts.termination(np.array(self._state.get_tiles(), dtype=np.int32), reward=reward)

        return ts.transition(np.array(self._state.get_tiles(), dtype=np.int32), reward=reward, discount=0.75)

    def _reset(self) -> ts.TimeStep:
        self._state = self._reset_state.copy().set_random_square().set_random_square()
        self._prev_state = None
        self._episode_ended = False
        return ts.restart(np.array(self._state.get_tiles(), dtype=np.int32))


    def render(self):
        self._state.display()


