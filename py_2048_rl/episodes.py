"""EpisodeDB package

Maintains the game data for modeling.
"""
import logging
import tensorflow as tf

logger = logging.getLogger('py2048')

# Data record structure
# We use a single tensor to store all the data.
# Positions:
# state -> 0-15
# next_state -> 16-31
# action -> 32
# reward -> 33
# score -> 34
# n_moves -> 35
# done -> 36
# All fields recorded as int32
DATA_REC_SIZE = 37
STATE_REC_SIZE = 16
STATE_POS = 0
NEW_STATE_POS = 16
ACTION_POS = 32
REWARD_POS = 33
SCORE_POS = 34
N_MOVES_POS = 35
DONE_POS = 36

class Episode:
    """Storage class designed to store a state of the environment

    state: current state
    next_state: the following state
    action: the action/move
    reward: reward resulting from move
    score: total score in game thus far
    n_moves: number of moves so far
    done: True is game over, False otherwise
    """
    def __init__(self, state, next_state, action, reward, score, n_moves, done, **kwargs):
        self.state = tf.reshape(tf.Variable(state, dtype=tf.int32), shape=(STATE_REC_SIZE,))
        self.next_state = tf.reshape(tf.Variable(next_state, dtype=tf.int32), shape=(STATE_REC_SIZE,))
        self.action = tf.Variable(action, dtype=tf.int32)
        self.reward = tf.Variable(reward, dtype=tf.int32)
        self.score = tf.Variable(score, dtype=tf.int32)
        self.n_moves = tf.Variable(n_moves, dtype=tf.int32)
        self.done = tf.cast(done, dtype=tf.int32)

    def get_content_tensor(self):
        return tf.concat([self.state,
                          self.next_state,
                          [self.action],
                          [self.reward],
                          [self.score],
                          [self.n_moves],
                          [self.done]
                          ],
                        0
                       )
class EdpisodeDB:
    """Collection of accrued Episode instances

    Includes methods to access a random selection of entries for NN modeling.
    """

    def __init__(self, mem_size, input_dims, **kwargs):
        """Class initialization

        Initializes the class instance of dimensions mem_size (stack ize)
        x input_dims (the dimension/size of input, 16 for a game state)
        """

        self.mem_size = tf.Variable(mem_size, dtype=tf.int32)
        self.storage = tf.zeros(shape=(0,DATA_REC_SIZE), dtype=tf.int32)

    def store_episode(self, e, **kwargs):
        """Add an Episode instance to the database
        """

        # Shedding the last slice if storage is full.
        if self.storage.shape[0] == self.mem_size:
            self.storage = tf.slice(self.storage,
                                    [0, 0],
                                    [self.mem_size - 1, DATA_REC_SIZE]
                                    )

        self.storage = tf.concat([[e.get_content_tensor()], self.storage], 0)

    def get_random_data_batch(self, batch_size):
        """Return a random selection out of the data tensor.
        """
        if batch_size >= self.storage.shape[0]:
            return tf.Variable(self.storage)

        sel_range_matrix = tf.reshape(tf.cast(tf.range(self.storage.shape[0]),
                                              dtype=tf.int64
                                              ),
                                      [1, self.storage.shape[0]]
                                      )

        sel_index = tf.random.uniform_candidate_sampler(sel_range_matrix, self.storage.shape[0],
                                                        batch_size,
                                                        True,
                                                        self.storage.shape[0]
                                                        )[0]

        return tf.gather(self.storage, sel_index)
