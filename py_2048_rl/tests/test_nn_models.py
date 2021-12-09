from unittest import TestCase
import numpy as np

from py_2048_rl.game.game import Game
from py_2048_rl.flex_rl_model import nn_model


class NN_ModelInitTest(TestCase):
    def test_init(self):
        model = nn_model.NN_Model(
            lr=0.0001,
        )

class NN_ModelTest(TestCase):
    def setUp(self):
        self.model = nn_model.NN_Model(
            lr=0.0001,
        )

    def test_create_tf_model(self):
        tf_model = self.model.create_tf_model()
