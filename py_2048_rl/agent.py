import importlib
import logging

import tensorflow as tf
import numpy as np

from py_2048_game import Game
from py_2048_rl import episodes

logger = logging.getLogger('py2048')


def random_action_callback(game):
    """Returns a randomly selected valid action"""
    return np.random.choice(game.available_actions())


class Agent:
    """Agent class for Reinforcement Learning

    Instance contains parameters and functionality to facilitate RL interaction with the environment which is the
    2048 game. 2048 game is implemented by way of a py_2048_game.Game instance.
    """

    def __init__(
            self,
            batch_size=10000,
            mem_size=50000,
            input_dims=[16],
            lr=0.0001,
            lr_min=0.00001,
            lr_redux=0.9,
            lr_patience=2,
            lr_verbose=2,
            gamma=0.99,
            gamma1=0.99,
            gamma2=0.99,
            gamma3=0.99,
            min_base=1e-06,
            q_base=1.0,
            epsilon=1,
            epsilon_dec=1e-3,
            epsilon_min=0.01,
            model_path='py_2048_rl.models.DEFAULT_MODEL',
            model_load_file=None,
            model_save_file='model.h5',
            model_auto_save=True,
            model_collect_random_data=True,
            log_dir="/tmp/",
            training_epochs=1,
            game_qc_threshold=0.5,
            game_max_replay_on_fail=50,
            **kwargs
        ):
        """Agent instance initialization

        Provides a functional instance with default settings.
        """

        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims[::]
        self.lr = lr
        self.lr_min = lr_min
        self.lr_redux = lr_redux
        self.lr_patience = lr_patience
        self.lr_verbose = lr_verbose
        self.gamma = gamma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.min_base = min_base
        self.q_base = q_base
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_path = model_path
        self.model_load_file = model_load_file
        self.model_save_file = model_save_file
        self.model_auto_save = model_auto_save
        self.model_collect_random_data = model_collect_random_data
        self.log_dir = log_dir
        self.training_epochs = training_epochs
        self.game_qc_threshold = game_qc_threshold
        self.game_max_replay_on_fail = game_max_replay_on_fail

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
            except OSError as err:
                logger.warning("Cannot load model %s: %s", self.model_load_file, err)
                logger.warning("Building new model")
                self.model = self._make_model()
        else:
            self.model = self._make_model()

        self.game_count = 0
        self.last_game_score = 0
        self.last_move_count = 0

        # Determines whether Q-values are to be maximum-bound or minimum-bound
        # only maximum-bound supported at this point.
        self.q_val_opt_max=True

        self.max_game_score = 0
        self.min_game_score = 0

    def adapt_epsilon(self):
        """Epsilon management function

        Epsilon determines explore vs predict ratio.
        """
        val = self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec

        return val

    def arg_sel_func(self):
        """Returns appropriate function for label selection based on Q-value

        argmax is maximum bound, argmin if minimum bound
        """
        if self.q_val_opt_max:
            return np.argmax
        return np.argmin

    def get_q_mod_func(self):
        """Returns appropriate function for Q-value data prep function

        Maximum-based for maximum bound calculation, minimum-based for minimum bound.
        """
        if self.q_val_opt_max:
            return self.get_modeling_data_q_max
        return self.get_modeling_data_q_min

    def _make_model(self):
        """MNN Model creation logic

        Creates a model as specified by the path, or DEFAULT.
        The definitions are in models.py
        """
        class_name = self.model_path.split('.')[-1]
        module_path = '.'.join([i for i in self.model_path.split('.')][:-1])
        models = importlib.import_module(module_path)
        model = getattr(models, class_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mean_squared_error',
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        return model

    def get_modeling_data_q_min(self):
        """Returns data for model training (states and labels) for minimum-bound calculation

        Factors involved: reward, next state prediction, ongoing score, game score, number of actions in game.
        """
        # Return None if there is no data to learn from
        if self.episode_db.mem_cntr == 0:
            return None, None

        sel_size, states, states_, actions, rewards, scores, n_moves, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(sel_size)
        q_target[batch_index, actions] = 1/(
            self.q_base +
            rewards +
            self.gamma *
            1/(self.min_base + np.min(q_next, axis=1)) +
            self.gamma1 * scores.numpy() +
            self.gamma2 * scores.numpy() *
            dones.numpy() +
            self.gamma3 * n_moves.numpy()
        )

        return states, q_target

    def get_modeling_data_q_max(self):
        """Returns data for model training (states and labels) for maximum-bound calculation

        Factors in the formula: reward, next state prediction, max tile base 2 exp. value, end game score,
        number of empty tiles.
        """
        # Return None if there is no data to learn from
        if self.episode_db.mem_cntr == 0:
            return None, None

        sel_size, states, states_, actions, rewards, scores, n_moves, dones = \
            self.episode_db.get_random_data_batch(self.batch_size)

        q_eval = tf.Variable(self.model.predict(states.numpy()))
        q_next = tf.Variable(self.model.predict(states_.numpy()))
        q_target = q_eval.numpy()

        batch_index = np.arange(sel_size)
        q_target[batch_index, actions] = tf.math.l2_normalize(
            self.q_base +
            rewards +
            self.gamma * np.max(q_next, axis=1) +
            self.gamma1 * tf.math.reduce_sum(tf.cast(tf.equal(states, 0), tf.int32), axis=1
                                             ).numpy() +
            self.gamma2 * scores.numpy() * dones.numpy() +
            self.gamma3 * np.max(states, axis=1)
        )

        return states, q_target

    def learn(self, run):
        """Facilitates single learn (model training) run
        """
        if self.model_collect_random_data:
            self.accumulate_episode_data()

        q_mod_func = self.get_q_mod_func()
        states, q_target = q_mod_func()

        if q_target is None:
            return

        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='categorical_accuracy',
                                                          factor=self.lr_redux,
                                                          patience=self.lr_patience,
                                                          min_lr=self.lr_min,
                                                          verbose=self.lr_verbose
                                                          )
                    ]

        if self.log_dir:
            callbacks.append(self.tb_callback)

        history = self.model.fit(
            states.numpy(),
            q_target,
            callbacks=callbacks,
            epochs=self.training_epochs
        )

        # Log
        tf.summary.scalar('Game score', data=self.last_game_score, step=run)
        tf.summary.scalar('Game move', data=self.last_move_count, step=run)
        tf.summary.scalar('Epsilon', data=self.epsilon, step=run)

        # Tensorboard tracking the metrics
        for name in history.history:
            tf.summary.scalar(name, data=history.history[name][-1], step=run)


    def learn_on_repeat(self, n_cycles=1, games_per_cycle=1, refill_episode_db=False):
        """Facilitates multiple model training runs as specified.

        parameters:
        n_cycles: number of training cycles

        games_per_cycle: the number of games to play between two successive cycles

        refill_episode_db: play enough games between two successive cycles to fully replace the whole EpisodeDB
        if set to True
        """

        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_cycles):
            self.learn(i)

            if self.model_auto_save:
                self.save_model()

            episode_count = 0
            cycle_game_count = 0

            while True:
                self.play_game(self.action_greedy_epsilon)
                cycle_game_count += 1
                episode_count += self.last_move_count
                self.game_count += 1

                if self.game_count == 1:
                    min_score = self.last_game_score
                else:
                    min_score = min(min_score, self.last_game_score)

                max_score = max(max_score, self.last_game_score)
                sum_scores += self.last_game_score
                avg_score = sum_scores / self.game_count

                logger.info('Game %d: min=%s avg=%s last=%s max=%s',
                            self.game_count, max_score, avg_score, self.last_game_score, max_score)

                tf.summary.scalar('Game score', data=self.last_game_score, step=self.game_count)
                tf.summary.scalar('Game move', data=self.last_move_count, step=self.game_count)
                tf.summary.scalar('Epsilon', data=self.epsilon, step=self.game_count)

                if ((not refill_episode_db) and
                    (cycle_game_count >= games_per_cycle)) or \
                    ((refill_episode_db) and
                     (episode_count >= self.episode_db.mem_size)):
                    break

            if self.log_dir:
                file_writer.flush()

        if self.log_dir:
            file_writer.close()


    def accumulate_episode_data(self):
        """"Fills instance's EpisodeDB instance with data.

         Collects random environment/game data.
         """

        # Bail if there's nothing to do.
        if self.episode_db.mem_cntr >= self.batch_size:
            return

        logger.debug("Initial data accumulation. Collection size = %s episodes.",
                     self.mem_size)
        while self.episode_db.mem_cntr < self.batch_size:
            self.play_game(random_action_callback, replay_on_fail=False)
        logger.debug("Initial data accumulation completed.")

    def game_qc(self, game):
        """Game quality control

         Takes an instance of Game as argument.
         Returns False (fail) if the score in that game is below the game_qc_threshold multiplied
         by the the maximum game score obtained thus far.
         Otherwise, returns True.

         Example: the highest score attained thus far is 10,000. A new game has just been concluded. The score
         in that game is 6,000. This game is accepted.

         Example: the highest score attained thus far is 10,000. A new game has just been concluded. The score
         in that game is 4,000. This game is rejected.
         """
        if game.score < self.game_qc_threshold * self.max_game_score:
            return False

        return True


    def play_game(self,
                  action_callback,
                  replay_on_fail=True,
                  max_replays=0,
                  record_in_episode_db=True
                  ):
        """Execute a full game fitting for data collection

         Arguments:

         action_callback: the function that takes a Game() instance as an argument and returns a next move (action)
         recommendation

         replay_on_fail: if True replay until the Game instance resulting passes quality control, for no more than the
         prescribed number of tries.
         """

        replay_cnt = 0
        top_cnt = 0
        prev_top_cnt = 0
        top_move_cnt = 0

        replay_lim = max_replays

        #
        # Defaulting to self.game_max_replay_on_fail
        # ig max_replays not specified.
        #
        if max_replays == 0:
            replay_lim = self.game_max_replay_on_fail

        if replay_on_fail and replay_lim == 0:
            replay_lim = self.game_max_replay_on_fail

        candidate_arr = []

        while True:
            episode_arr = []
            game = Game()

            while not game.game_over():
                action = action_callback(game)
                state = np.matrix.flatten(game.state)
                reward = game.do_action(action)
                state_ = np.matrix.flatten(game.state)
                episode_arr.append(
                    episodes.Episode(
                        state=state,
                        next_state=state_,
                        action=action,
                        reward=reward,
                        score=game.score,
                        n_moves=game.move_count,
                        done=game.game_over()
                    )
                )

            top_cnt = max(game.score, top_cnt)

            if top_cnt > prev_top_cnt:
                candidate_arr = episode_arr.copy()
                prev_top_cnt = top_cnt
                top_move_cnt = game.move_count

            replay_cnt += 1

            # Exiting the cycle once the game passes game_qc
            # or if the quality control is not required
            # or database recording is not requested
            # or the replay limit is reached
            if not record_in_episode_db or \
                    not replay_on_fail or \
                    self.game_qc(game) or \
                    replay_cnt >= replay_lim:
                break

        if record_in_episode_db:
            for e in candidate_arr:
                self.episode_db.store_episode(e)

        self.max_game_score = max(self.max_game_score, top_cnt)

        if self.min_game_score == 0:
            self.min_game_score = top_cnt
        else:
            self.min_game_score = min(self.min_game_score, top_cnt)

        self.last_game_score = top_cnt
        self.last_move_count = top_move_cnt

    def action_greedy_epsilon(self, game):
        """Random selection with the probability of epsilon

        Greedy (most profitable) action otherwise.
        """

        if np.random.random() < self.adapt_epsilon():
            return random_action_callback(game)

        return self.action_greedy(game)

    def action_greedy(self, game):
        """Greedy (most profitable) game action selection.

         NN model prediction is used ass a presumed best selection.
         """
        state = game.state
        state = np.matrix.reshape(state, (1, 16))

        pred_actions = self.model.predict(state)[0]
        avai_actions = game.available_actions()
        sel_f = self.arg_sel_func()
        return avai_actions[sel_f(pred_actions[avai_actions])]

    def play_on_repeat(self, n_games=1):
        """Execute the number of games as specified by n_games
        """

        min_score = 0
        max_score = 0
        sum_scores = 0

        if self.log_dir:
            file_writer = tf.summary.create_file_writer(self.log_dir)
            file_writer.set_as_default()

        for i in range(n_games):
            self.play_game(self.action_greedy,
                           replay_on_fail=False,
                           record_in_episode_db=False
                           )

            if self.model_auto_save:
                self.save_model()

            if i == 0:
                min_score = self.last_game_score
            else:
                min_score = min(min_score, self.last_game_score)

            max_score = max(max_score, self.last_game_score)
            sum_scores += self.last_game_score
            avg_score = sum_scores / (i+1)

            logger.info('Step %d: min=%s avg=%s last=%s max=%s',
                        i, min_score, avg_score, self.last_game_score, max_score)

            if self.log_dir:
                tf.summary.scalar('Game move (inference)', data=self.last_move_count, step=i)
                tf.summary.scalar('Game score (inference)', data=self.last_game_score, step=i)
                file_writer.flush()

        if self.log_dir:
            file_writer.close()

    def save_model(self):
        """Save current model to a file
         """
        self.model.save(self.model_save_file)


    def load_model(self):
        """Load a model in from file
        """
        return tf.keras.models.load_model(self.model_load_file)


def get_default_agent(env):
    """Obtain an Agent instance defined by env
    """
    return Agent(**env)
