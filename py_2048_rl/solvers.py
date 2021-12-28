import os
import numpy as np
import tensorflow as tf
from py_2048_game import solvers
from py_2048_rl import agent


class TensorflowSolver(solvers.BaseSolver):
    def __init__(self, agent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent or self.get_default_agent()

    def get_default_agent(self):
        return agent.get_default_agent(os.environ)

    def solve(self, game):
        state = game.state
        state = np.matrix.reshape(state, (1, 16))

        self.agent.model.predict(state)

        pred_actions = self.agent.model.predict(state)[0]
        avai_actions = game.available_actions()
        action = avai_actions[tf.argmin(pred_actions[avai_actions])]
        reward = game.do_action(action)
        return (
            game.state,
            action,
            reward
        )
