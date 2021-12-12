from unittest import TestCase
from py_2048_rl.flex_rl_model import agent
from py_2048_rl.game.game import Game


class AgentInitTest(TestCase):
    def test_init(self):
        agent_ = agent.Agent()


class AgentTest(TestCase):
    def setUp(self):
        self.agent = agent.Agent()

    def test_learn(self):
        self.agent.learn(0)

    def test_learn_on_repeat(self):
        self.agent.learn_on_repeat()

    def test_accumulate_episode_data(self):
        self.agent.accumulate_episode_data()

    def test_play_game(self):
        action_choice = lambda g: 0
        self.agent.play_game(action_choice)

    def test_action_random(self):
        game = Game()
        action = self.agent.action_random(game)
        self.assertIn(action, (0, 1, 2, 3))

    def test_action_greedy_epsilon(self):
        game = Game()
        action = self.agent.action_greedy_epsilon(game)
        self.assertIn(action, (0, 1, 2, 3))
