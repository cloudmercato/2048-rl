import tensorflow as tf2
import numpy as np
import os
import py_2048_rl.game.game as game
import py_2048_rl.game.play as play
import episodes


class Agent():
  def __init__(self, **kwargs):
    self.__hash = {}

    self.__hash["lr"] = 0.001
    self.__hash["gamma"] = 0.99
    self.__hash["n_actions"] = 4
    self.__hash["epsilon"] = 1
    self.__hash["batch_size"] = 10000
    self.__hash["input_dims"] = 16
    self.__hash["epsilon_dec"] = 1e-3
    self.__hash["epsilon_min"] = 0.01
    self.__hash["mem_size"] = 1000000
    self.__hash["fname"] = 'model.h5'

    for k in kwargs.keys():
      self.__hash[k] = kwargs[k]

    """
    Special parameters.
    """
    if "model" not in self.__hash.keys():
      self.__create_default_model()

    if "episode_db" not in self.__hash.keys():
      self.__hash["episode_db"] = episodes.EdpisodeDB()

  def __create_default_model(self):
    model: tf2.keras.models.Sequential = tf2.keras.Sequential([
      tf2.keras.layers.Dense(16, activation='relu'),
      tf2.keras.layers.Dense(64, activation='relu'),
      tf2.keras.layers.Dense(self.__hash["n_actions"], activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    self.__hash["model"] = model

  def learn(self):
    self.accumulate_episode_data()
    ep_db = self.__hash["episode_db"]
    m1 = self.__hash["model"]
    states, states_, actions, rewards, dones = \
      ep_db.get_random_data_batch(self.__hash['batch_size'])

    q_eval = tf2.Variable(tf2.constant(m1.predict(states)))
    q_next = tf2.Variable(tf2.constant(m1.predict(states_)))
    q_target = t2.Variable(q_eval)
    batch_index = tf2.range(self.__hash['batch_size'])
    q_target[batch_index, actions] = rewards + self.__hash["gamma"] * q_next.numpy().max(axis=1)*dones
    m1.train_on_batch(states, q_target)

    # Adjust the epsilon
    self.__hash["epsilon"] = self.__hash["epsilon"]  - self.__hash["epsilon_dec"] \
      if self.__hash["epsilon"] >self.__hash["epsilon_min"] else self.__hash["epsilon_min"]

  def accumulate_episode_data(self):
    ep_db = self.__hash["episode_db"]
    while ep_db.mem_cntr < self.__hash["batch_size"]:
      self.play_game()

  def play_game(self, action_choice):
    game1 = game.Game()
    state = None
    action = None
    state = None
    state_ = None
    reward = None
    score = None
    e_db = self.__hash["episode_db"]

    while not game1.game_over():
      action = action_choice(game1)
      state = game1.state()
      reward = game1.do_action()
      state_ = game1.state()
      e_db.store_episode(
        episodes.Episode(state=state,
                         new_state=state_,
                         action=action,
                         reward=reward,
                         score=game.score(),
                         done=game1.game_over()))

  def action_random(self, curr_game):
    return np.random.choice(curr_game.available_actions())

  def action_greedy_epsilon(self, curr_game):
    if np.random.random() < self.__hash["epsilon"]:
      return self.action_random(curr_game)

    state = curr_game.state()
    m1 = self.__hash["model"]
    actions = m1.predict(state)
    return tf2.argmax(actions)

  def save_model(self):
    m1 = self.__hash["model"]
    m1.save(self.__hash["fname"])

  def load_model(self):
    self.__hash["model"] = tf2.keras.models.load_model(self.__hash["fname"])
