from unittest import TestCase
from py_2048_rl.flex_rl_model import agent


class AgentTest(TestCase):
    def test_init(self):
        agent_ = agent.Agent()
