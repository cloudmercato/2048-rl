import tensorflow as tf2
import tensorboard_plugin_profile
import numpy as np
import os
import py_2048_rl.game.game as game
import py_2048_rl.logging.logger as logger
import episodes


class Agent():
  def __init__(self, **kwargs):
    self.__hash = {}
    self.__hash["batch_size"] = 10000
    self.__hash["mem_size"] = 50000
    self.__hash["input_dims"] = [16]
    self.__hash["lr"] = 0.001
    self.__hash["gamma"] = 0.99
    self.__hash["n_actions"] = 4
    self.__hash["epsilon"] = 1
    self.__hash["epsilon_dec"] = 1e-3
    self.__hash["epsilon_min"] = 0.01
    self.__hash["model_load_file"] = None
    self.__hash["model_save_file"] = 'model.h5'
    self.__hash["model_auto_save"] = True
    self.__hash["log_dir"] = "/app/logs"
    self.__hash["tf_proc_debug"] = False
    self.__hash["training_epochs"] = 1

    for k in kwargs.keys():
      self.__hash[k] = kwargs[k]

    """
    Special parameters.
    """
    if "model" not in self.__hash.keys():
      self.__create_default_model()

    if "episode_db" not in self.__hash.keys():
      self.__hash["episode_db"] = episodes.EdpisodeDB(self.__hash["mem_size"],\
                                                      self.__hash["input_dims"],
                                                      tf_proc_debug=self.__hash["tf_proc_debug"])

    self.__hash["last_game_score"] = 0
    self.__hash["logger"] = logger.Logger()

    self.__hash["tensorboard_callback"] =\
      tf2.keras.callbacks.TensorBoard(log_dir=self.__hash["log_dir"],
                                      histogram_freq=1,
                                      profile_batch = '500,520')

    tf2.profiler.experimental.server.start( 6009 )

    self.__hash["training_histories"] = []

    tf2.debugging.set_log_device_placement(self.__hash["tf_proc_debug"])
    if self.__hash["tf_proc_debug"]:
      tf2.debugging.experimental.enable_dump_debug_info(self.__hash["log_dir"],
                                                        tensor_debug_mode="FULL_HEALTH",
                                                        circular_buffer_size=-1)

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
    tbcb = self.__hash["tensorboard_callback"]
    tf2.debugging.set_log_device_placement(self.__hash["tf_proc_debug"])

    states, states_, actions, rewards, dones = \
      ep_db.get_random_data_batch(self.__hash['batch_size'])

    q_eval = tf2.Variable(tf2.constant(m1.predict(states.numpy())))
    q_next = tf2.Variable(tf2.constant(m1.predict(states_.numpy())))
    q_target = q_eval.numpy()
    batch_index = np.arange(self.__hash['batch_size'])
    q_target[batch_index, actions] = rewards + self.__hash["gamma"] * \
                                     np.max(q_next.numpy(), axis=(1))
    history = m1.fit(tf2.constant(states),
                     q_target,
                     callbacks=[tbcb],
                     epochs=self.__hash["training_epochs"])

    self.__hash["training_histories"].append(history)

    # Adjust the epsilon
    self.__hash["epsilon"] = self.__hash["epsilon"]  - self.__hash["epsilon_dec"] \
      if self.__hash["epsilon"] >self.__hash["epsilon_min"] else self.__hash["epsilon_min"]

  def learn_on_repeat(self, n_games=1):
    log = self.__hash["logger"]
    min_score = 0
    max_score = 0
    avg_score = 0.0
    sum_scores = 0
    run_num = 0

    for i in range(n_games):
      self.learn()
      self.play_game(self.action_greedy_epsilon)
      if self.__hash["model_auto_save"]: self.save_model()

      min_score = self.__hash["last_game_score"] if i == 0 \
        else min(min_score, self.__hash["last_game_score"] )

      max_score = max(max_score, self.__hash["last_game_score"] )
      sum_scores += self.__hash["last_game_score"]
      run_num = i+1
      avg_score = sum_scores / run_num

      log.generic_output(field_names = ["Learning run",
                                        "Min score",
                                        "Average score",
                                        "Last score",
                                        "Max score"], \
                            field_content = [run_num.__str__(),
                                             min_score.__str__(),
                                             avg_score.__str__(),
                                             self.__hash["last_game_score"].__str__(),
                                             max_score.__str__()])


  def accumulate_episode_data(self):
    ep_db = self.__hash["episode_db"]
    # Bail if there's nothing to do.
    if ep_db.mem_cntr >= self.__hash["batch_size"]: return

    log = self.__hash["logger"]
    log.freeform_output("Initial data accumulation. Collection size = " +
                        self.__hash["mem_size"].__str__() +
                         " episodes.")

    while ep_db.mem_cntr < self.__hash["batch_size"]:
      self.play_game(self.action_random)

    log.freeform_output("Initial data accumulation completed.")

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
      state = np.matrix.flatten( game1.state() )
      reward = game1.do_action(action)
      state_ = np.matrix.flatten( game1.state() )
      e_db.store_episode(
        episodes.Episode(state=state,
                         next_state=state_,
                         action=action,
                         reward=reward,
                         score=game1.score(),
                         done=game1.game_over()))

    self.__hash["last_game_score"] = game1.score()

  def action_random(self, curr_game):
    return np.random.choice(curr_game.available_actions())

  def action_greedy_epsilon(self, curr_game):
    if np.random.random() < self.__hash["epsilon"]:
      return self.action_random(curr_game)

    state = curr_game.state()
    state = np.matrix.reshape(state, (1, 16))
    m1 = self.__hash["model"]
    actions = m1.predict(state)
    actions = actions[0][curr_game.available_actions()]
    return np.argmax(actions)

  def save_model(self):
    m1 = self.__hash["model"]
    m1.save(self.__hash["model_save_file"])

  def load_model(self):
    self.__hash["model"] = tf2.keras.models.load_model(self.__hash["model_load_file"])
