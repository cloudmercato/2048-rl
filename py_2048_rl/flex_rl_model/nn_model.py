"""
Neural Network model for RL analysis
"""
import tensorflow as tf2


#--- BEGIN NN_Model ---
class NN_Model(object):
  """
  Init params supported

  """
  def __init__(self, **kwargs):
    self.__hash = {}
    self.__hash["layers_shape"] = [[16], [64], [64], [4]]
    self.__hash["layers_activation_func"] = [None, tf2.keras.activations.mse,
                                             tf2.keras.activations.mse, None]
    self.__hash["default_activation_function"] = tf2.nn.relu
    self.__hash["nn_type"] = "dqn"
    self.__hash["log_dir"] = "/app/logs"

    # Copy content in from the argument hash.
    for k in kwargs.keys():
      self.__hash[k] = kwargs[k]

    if self.get_param("tf_model") is None:
      self.__hash["tf_model"] = self.create_tf_model()

  def create_tf_model(self):
    layers = tf2.keras.layers
    l_arr = []
    l =None
    m_shape = self.get_param("layers_shape")
    act_func = self.get_param("default_activation_function")

    if len(m_shape) < 2: return None

    for i in range( len(self.get_param("layers_shape")) ):
      if i == 0:
        l = layers.Input(shape=m_shape[i])
      elif i == len(m_shape) - 1:
        l = layers.Output(shape=m_shape[i])
      else:
        l = layers.Dense(shape=m_shape[i], activation=act_func)

      l_arr.append(l)

    return tf2.keras.model.Sequential(l_arr)

  def get_param(self, name):
    if name in self.__hash.keys(): return self.__hash[name]
    return None

  def set_param(self, name, value):
    self.__hash[name] = value


#--- END NN_Model
