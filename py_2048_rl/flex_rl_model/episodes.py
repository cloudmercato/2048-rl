import numpy as np
import tensorflow as tf2
import os


class Episode():
  def __init__(self, state, next_state, action, reward, score, done, **kwags):
    self.state = state
    self.next_state = next_state
    self.action = action
    self.reward = reward
    self.score = score
    self.done = done


class EdpisodeDB():
  def __init__(self, mem_size, input_dims):
    self.mem_size = mem_size
    self.mem_cntr = 0
    self.states_mem = tf2.Variable(tf.constant(shape=input_dims, dtype=tf.float32))
    self.new_states_mem = tf2.Variable(tf.constant(shape=input_dims, dtype=tf.float32))
    self.action_mem = tf2.Variable(tf.constant(shape=input_dims, dtype=tf2.int32))
    self.reward_mem = tf2.Variable(tf.constant(shape=input_dims, dtype=tf2.int32))

  def store_episode(self, e, kwargs):
    ind = tf2.constant([self.mem_cntr % self.mem_size])
    self.states_mem.scatter_nd_update(ind, tf.comstant(e.state, dtype=tf2.float32))
    self.new_states_mem.scatter_nd_update(ind, tf.comstant(e.next_state, dtype=tf2.float32))
    self.action_mem.scatter_nd_update(ind, tf.comstant(e.action, dtype=tf2.int32))
    self.reward_mem.scatter_nd_update(ind, tf.comstant(e.reward, dtype=tf2.int32))
    self.mem_cntr += 1

  def get_random_data_batch(self, batch_size):
    total_db_size = min(self.mem_cntr, self.mem_size)
    batch_arr = np.random.choice(total_db_size, batch_size, replace=False)
    states_batch = tf2.Variable(tf.constant(states_mem.numpy[batch_arr]))
    new_states_batch = tf2.Variable(tf.constant(new_states_mem.numpy[batch_arr]))
    action_batch = tf2.Variable( tf.constant(action_mem.numpy[batch_arr]) )
