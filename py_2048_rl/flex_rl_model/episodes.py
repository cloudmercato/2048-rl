import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('py2048')


class Episode:
    def __init__(self, state, next_state, action, reward, score, done, **kwargs):
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.score = score
        self.done = done

        self.__hash = {}
        self.__hash["tf_proc_debug"] = False
        for k in kwargs.keys():
            self.__hash[k] = kwargs[k]

        tf.debugging.set_log_device_placement(self.__hash["tf_proc_debug"])


class EdpisodeDB:
    def __init__(self, mem_size, input_dims, **kwargs):
        self.__hash = {}
        self.__hash["tf_proc_debug"] = False

        for k in kwargs.keys():
            self.__hash[k] = kwargs[k]

        self.mem_size = mem_size
        self.mem_cntr = 0
        states_dims = [mem_size]
        states_dims.extend(input_dims)
        self.states_mem = tf.Variable(tf.constant(0., shape=states_dims, dtype=tf.float32))
        self.new_states_mem = tf.Variable(tf.constant(0., shape=states_dims, dtype=tf.float32))
        self.action_mem = tf.Variable(tf.constant(0, shape=(mem_size), dtype=tf.int32))
        self.reward_mem = tf.Variable(tf.constant(0, shape=(mem_size), dtype=tf.int32))
        self.score_mem = tf.Variable(tf.constant(0, shape=(mem_size), dtype=tf.int32))
        self.done_mem = tf.Variable(tf.constant(False, shape=(mem_size), dtype=tf.bool))

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
        self.done_mem = tf.Variable(tf.constant(done_mem_np))

        self.mem_cntr += 1

    def get_random_data_batch(self, batch_size):
        total_db_size = min(self.mem_cntr, self.mem_size)
        batch_arr = np.random.choice(total_db_size, batch_size, replace=False)
        states_batch = tf.Variable(tf.constant(self.states_mem.numpy()[batch_arr]))
        new_states_batch = tf.Variable(tf.constant(self.new_states_mem.numpy()[batch_arr]))
        action_batch = tf.Variable(tf.constant(self.action_mem.numpy()[batch_arr]))
        reward_batch = tf.Variable(tf.constant(self.reward_mem.numpy()[batch_arr]))
        score_batch = tf.Variable(tf.constant(self.score_mem.numpy()[batch_arr]))
        done_batch = tf.Variable(tf.constant(self.done_mem.numpy()[batch_arr]))
        return states_batch, new_states_batch, action_batch, reward_batch, score_batch, done_batch
