from unittest import TestCase, mock
from py_2048_game import Game
from py_2048_rl import agent


class AgentInitTest(TestCase):
    def test_init(self):
        agent_ = agent.Agent()


class AgentTest(TestCase):
    def setUp(self):
        self.agent = agent.Agent(
            batch_size=1,
        )

    def test_learn(self):
        self.agent.learn(0)

    def test_learn_on_repeat(self):
        self.agent.learn_on_repeat()

    def test_accumulate_episode_data(self):
        self.agent.accumulate_episode_data()

    @mock.patch('py_2048_game.Game.game_over', side_effect=(False, False, True))
    def test_play_game(self, mock):
        action_choice = lambda g: 0
        self.agent.play_game(action_choice)

    def test_action_greedy_epsilon(self):
        game = Game()
        action = self.agent.action_greedy_epsilon(game)
        self.assertIn(action, (0, 1, 2, 3))
