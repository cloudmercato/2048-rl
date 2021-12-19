import logging

import tensorflow as tf
import numpy as np

from py_2048_game import Game
from py_2048_rl import episodes
from py_2048_rl import models

logger = logging.getLogger('py2048')


def random_action_callback(game):
    return np.random.choice(game.available_actions())


class Agent:
    def __init__(
            self,
            batch_size=10000,
            mem_size=50000,
            input_dims=[16],
            lr=0.001,
            gamma=0.99,
            gamma1=0.99,
            gamma2=0.99,
            gamma3=0.99,
            epsilon=1,
            epsilon_dec=1e-3,
            epsilon_min=0.01,
            model_load_file=None,
            model_save_file='model.h5',
            model_auto_save=True,
            model_collect_random_data=True,
            log_dir="/tmp/",
            training_epochs=1,
            **kwargs
        ):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims[::]
        self.lr = lr
        self.gamma = gamma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_load_file = model_load_file
        self.model_save_file = model_save_file
        self.model_auto_save = model_auto_save
        self.model_collect_random_data = model_collect_random_data
        self.log_dir = log_dir
        self.training_epochs = training_epochs

        self.episode_db = episodes.EdpisodeDB(
            self.mem_size,
            self.input_dims,
        )

        if self.log_dir:
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                profile_batch='500,520'
            )

        if self.model_load_file:
            try:
                self.model = self.load_model()
                # Disabling pre-loading random data for model pre-loaded from file.
                self.model_collect_random_data = False
            except OSError as err:
                logger.warning("Cannot load model %s: %s", self.model_load_file, err)
                logger.warning("Building new model")
                self.model = self._make_model()
        else:
            self.model = self._make_model()
        self.last_game_score = 0
        self.last_move_count = 0

    def _make_model(self):
        model = models.DEFAULT_MODEL
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mean_squared_error'
        )
        return model

    def learn(self, run):
        if self.model_collect_random_data:
            self.accumulate_episode_data()

        # Exit if no data to learn from
        if self.episode_db.mem_cntr == 0:
            return

        sel_size, states, states_, actions, rewards, scores, n_moves, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(sel_size)
        q_target[batch_index, actions] = tf.math.l2_normalize(
            1/tf.math.exp(
                tf.math.l2_normalize(
                    rewards +
                    self.gamma * np.max(q_next, axis=1) +
                    self.gamma1 * scores.numpy() +
                    self.gamma2 * scores.numpy() *
                    dones.numpy() +
                    self.gamma3 * n_moves.numpy()
                )
            )
        )

        callbacks = []
        if self.log_dir:
            callbacks.append(self.tb_callback)

        history = self.model.fit(
            states.numpy(),
            q_target,
            callbacks=callbacks,
            epochs=self.training_epochs
        )
        # Adjust the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
        # Log
        tf.summary.scalar('Game score', data=self.last_game_score, step=run)
        tf.summary.scalar('Game move', data=self.last_move_count, step=run)
        tf.summary.scalar('Epsilon', data=self.epsilon, step=run)

        for name in history.history:
            tf.summary.scalar(name, data=history.history[name][-1], step=run)

    def learn_on_repeat(self, n_games=1):
        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_games):
            self.learn(i)
            self.play_game(self.action_greedy_epsilon)

            if self.model_auto_save:
                self.save_model()

            if i != 0:
                min_score = min(min_score, self.last_game_score)
            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            avg_score = sum_scores / (i+1)

            logger.info('Step %d: min=%s avg=%s last=%s max=%s',
                        i, max_score, avg_score, self.last_game_score, max_score)
            if self.log_dir:
                file_writer.flush()

        if self.log_dir:
            file_writer.close()

    def accumulate_episode_data(self):
        # Bail if there's nothing to do.
        if self.episode_db.mem_cntr >= self.batch_size:
            return

        if not self.model_collect_random_data:
            return

        logger.debug("Initial data accumulation. Collection size = %s episodes.",
                     self.mem_size)
        while self.episode_db.mem_cntr < self.batch_size:
            self.play_game(random_action_callback)
        logger.debug("Initial data accumulation completed.")

    def play_game(self, action_callback):
        game = Game()

        while not game.game_over():
            action = action_callback(game)
            state = np.matrix.flatten(game.state())
            reward = game.do_action(action)
            state_ = np.matrix.flatten(game.state())
            episode = episodes.Episode(
                state=state,
                next_state=state_,
                action=action,
                reward=reward,
                score=game.score,
                n_moves=game.move_count,
                done=game.game_over()
            )
            self.episode_db.store_episode(episode)

        self.last_game_score = game.score
        self.last_move_count = game.move_count

    def action_greedy_epsilon(self, game):
        if np.random.random() < self.epsilon:
            return random_action_callback(game)

        return self.action_greedy(game)

    def action_greedy(self, game):
        state = game.state()
        state = np.matrix.reshape(state, (1, 16))

        pred_actions = self.model.predict(state)[0]
        avai_actions = game.available_actions()
        return avai_actions[np.argmin(pred_actions[avai_actions])]

    def play_on_repeat(self, n_games=1):
        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_games):
            self.play_game(self.action_greedy_epsilon)

            if i != 0:
                min_score = min(min_score, self.last_game_score)
            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            avg_score = sum_scores / (i+1)

            logger.info('Step %d: min=%s avg=%s last=%s max=%s',
                        i, max_score, avg_score, self.last_game_score, max_score)
            if self.log_dir:
                file_writer.flush()

        if self.log_dir:
            file_writer.close()

    def save_model(self):
        self.model.save(self.model_save_file)

    def load_model(self):
        return tf.keras.models.load_model(self.model_load_file)
