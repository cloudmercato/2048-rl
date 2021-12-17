import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('py2048')


class Episode:
    def __init__(self, state, next_state, action, reward, score, n_moves, done, **kwargs):
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.score = score
        self.n_moves = n_moves
        self.done = done


class EdpisodeDB:
    def __init__(self, mem_size, input_dims, **kwargs):
        self.mem_size = mem_size
        self.mem_cntr = 0
        states_dims = [mem_size]
        states_dims.extend(input_dims)
        self.states_mem = tf.Variable(np.zeros(states_dims, np.float32))
        self.new_states_mem = tf.Variable(np.zeros(states_dims, np.float32))

        self.action_mem = tf.Variable(np.zeros(mem_size, dtype=np.int32))
        self.reward_mem = tf.Variable(np.zeros(mem_size, dtype=np.float32))
        self.score_mem = tf.Variable(np.zeros(mem_size, dtype=np.float32))
        self.n_moves_mem = tf.Variable(np.zeros(mem_size, dtype=np.float32))
        self.done_mem = tf.Variable(np.zeros(mem_size, dtype=np.bool))

    def store_episode(self, e, **kwargs):
        ind = self.mem_cntr % self.mem_size
        ind_arr = tf.Variable(tf.constant([
            [ind, 0], [ind, 1], [ind, 2], [ind, 3],
            [ind, 4], [ind, 5], [ind, 6], [ind, 7],
            [ind, 8], [ind, 9], [ind, 10], [ind, 11],
            [ind, 12], [ind, 13], [ind, 14], [ind, 15]
        ]))

        self.states_mem.scatter_nd_update(ind_arr, tf.constant(e.state, dtype=tf.float32))
        self.new_states_mem.scatter_nd_update(ind_arr, tf.constant(e.next_state, dtype=tf.float32))
        self.action_mem.scatter_nd_update([ind], [e.action])
        self.reward_mem.scatter_nd_update([ind], [e.reward])
        self.score_mem.scatter_nd_update([ind], [e.score])

        # Special processing for boolean in done_mem
        done_mem_np = self.done_mem.numpy()
        done_mem_np[ind] = e.done
        self.done_mem = tf.Variable(done_mem_np)

        self.mem_cntr += 1

    def get_random_data_batch(self, batch_size):
        total_db_size = min(self.mem_cntr, self.mem_size)
        batch_arr = np.random.choice(total_db_size, batch_size, replace=False)
        states_batch = tf.Variable(self.states_mem.numpy()[batch_arr])
        new_states_batch = tf.Variable(self.new_states_mem.numpy()[batch_arr])
        action_batch = tf.Variable(self.action_mem.numpy()[batch_arr])
        reward_batch = tf.Variable(self.reward_mem.numpy()[batch_arr])
        score_batch = tf.Variable(self.score_mem.numpy()[batch_arr])
        n_moves_batch = tf.Variable(self.n_moves_mem.numpy()[batch_arr])
        done_batch = tf.Variable(self.done_mem.numpy()[batch_arr])
        return states_batch, new_states_batch, action_batch, reward_batch, score_batch, n_moves_batch, done_batch
