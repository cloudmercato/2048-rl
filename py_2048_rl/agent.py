import logging

import tensorflow as tf
import numpy as np

from py_2048_game import Game
from py_2048_rl import episodes
from py_2048_rl import models

logger = logging.getLogger('py2048')


class Agent:
    def __init__(
            self,
            batch_size=10000,
            mem_size=50000,
            input_dims=[16],
            lr=0.001,
            gamma=0.99,
            gamma1=0.99,
            epsilon=1,
            epsilon_dec=1e-3,
            epsilon_min=0.01,
            model_load_file=None,
            model_save_file='model.h5',
            model_auto_save=True,
            log_dir="/tmp/logs",
            training_epochs=1,
            **kwargs
        ):
        self.__hash = {}
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims[::]
        self.lr = lr
        self.gamma = gamma
        self.gamma1 = gamma1
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_load_file = model_load_file
        self.model_save_file = model_save_file
        self.model_auto_save = model_auto_save
        self.log_dir = log_dir
        self.training_epochs = training_epochs

        for k, v in kwargs.items():
            self.__hash[k] = kwargs[k]


        if "episode_db" not in self.__hash.keys():
            self.episode_db = episodes.EdpisodeDB(
                self.mem_size,
                self.input_dims,
            )

        if self.model_load_file:
            self.model = self.load_model()
        else:
            self.model = models.DEFAULT_MODEL
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss='mean_squared_error'
            )
        self.last_game_score = 0
        self.last_move_count = 0

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            profile_batch='500,520'
        )

        self.training_histories = []

    def learn(self, run):
        self.accumulate_episode_data()
        ep_db = self.episode_db
        m1 = self.model

        states, states_, actions, rewards, scores, dones = \
            ep_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(tf.constant(m1.predict(states.numpy())))
        q_next = tf.Variable(tf.constant(m1.predict(states_.numpy())))
        q_target = q_eval.numpy()

        batch_index = np.arange(self.batch_size)
        q_target[batch_index, actions] = rewards + self.gamma * \
             np.max(q_next.numpy(), axis=(1)) + \
             self.gamma1 * scores.numpy()
        history = m1.fit(
            tf.constant(states),
            q_target,
            callbacks=[self.tensorboard_callback],
            epochs=self.training_epochs
        )
        # Log
        tf.summary.scalar('Game score', data=self.last_game_score, step=run)
        tf.summary.scalar('Game move', data=self.last_move_count, step=run)

        for name in history.history:
            tf.summary.scalar(name, data=history.history[name][-1], step=run)

        self.training_histories.append(history)

        # Adjust the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec

    def learn_on_repeat(self, n_games=1):
        min_score = 0
        max_score = 0
        avg_score = 0.0
        sum_scores = 0
        run_num = 0

        for i in range(n_games):
            self.learn(i)
            self.play_game(self.action_greedy_epsilon)

            if self.model_auto_save:
                self.save_model()

            if i == 0:
                min_score = self.last_game_score
            else:
                min_score = min(min_score, self.last_game_score)
            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            run_num += 1
            avg_score = sum_scores / run_num

            logger.info('Step %d: Run=%s min=%s avg=%s last=%s max=%s',
                        i, run_num, max_score, avg_score, self.last_game_score, max_score)

    def accumulate_episode_data(self):
        ep_db = self.episode_db
        # Bail if there's nothing to do.
        if ep_db.mem_cntr >= self.batch_size:
            return

        logger.debug("Initial data accumulation. Collection size = %s episodes.",
                     self.mem_size)
        while ep_db.mem_cntr < self.batch_size:
            self.play_game(self.action_random)
        logger.debug("Initial data accumulation completed.")

    def play_game(self, action_choice):
        game = Game()
        e_db = self.episode_db

        while not game.game_over():
            action = action_choice(game)
            state = np.matrix.flatten(game.state())
            reward = game.do_action(action)
            state_ = np.matrix.flatten(game.state())
            episode = episodes.Episode(
                state=state,
                next_state=state_,
                action=action,
                reward=reward,
                score=game.score,
                done=game.game_over()
            )
            e_db.store_episode(episode)

        self.last_game_score = game.score
        self.last_move_count = game.move_count

    def action_random(self, curr_game):
        return np.random.choice(curr_game.available_actions())

    def action_greedy_epsilon(self, curr_game):
        if np.random.random() < self.epsilon:
            return self.action_random(curr_game)

        state = curr_game.state()
        state = np.matrix.reshape(state, (1, 16))
        m1 = self.model
        actions = m1.predict(state)
        actions = actions[0][curr_game.available_actions()]
        return np.argmax(actions)

    def save_model(self):
        m1 = self.model
        m1.save(self.model_save_file)

    def load_model(self):
        return tf.keras.models.load_model(self.model_load_file)
