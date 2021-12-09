from unittest import TestCase
import numpy as np

from py_2048_rl.game.game import Game
from py_2048_rl.flex_rl_model import episodes


class EpisodeTest(TestCase):
    def setUp(self):
        self.game = Game()

    def test_init(self):
        action = 0
        state = np.matrix.flatten(self.game.state())
        reward = self.game.do_action(action)
        state_ = np.matrix.flatten(self.game.state())
        episode = episodes.Episode(
            state=state,
            next_state=state_,
            action=action,
            reward=reward,
            score=self.game.score(),
            done=self.game.game_over(),
        )


class EpisodeDbInitTest(TestCase):
    def test_init(self):
        db = episodes.EdpisodeDB(
            mem_size=50000,
            input_dims=[16],
        )

class EpisodeDbTest(TestCase):
    def setUp(self):
        self.game = Game()
        action = 0
        state = np.matrix.flatten(self.game.state())
        reward = self.game.do_action(action)
        state_ = np.matrix.flatten(self.game.state())
        self.episode = episodes.Episode(
            state=state,
            next_state=state_,
            action=action,
            reward=reward,
            score=self.game.score(),
            done=self.game.game_over(),
        )

    def test_store_episode(self):
        db = episodes.EdpisodeDB(
            mem_size=50000,
            input_dims=[16],
        )
        db.store_episode(self.episode)

    def test_get_random_data_batch(self):
        # Setup
        db = episodes.EdpisodeDB(
            mem_size=50000,
            input_dims=[16],
        )
        db.store_episode(self.episode)
        # Run
        db.get_random_data_batch(1)
