"""
Neural Network model for RL analysis
"""
import tensorflow as tf2
import agent


#--- BEGIN NN_Model ---
class NN_Model(object):
  """
  Init params supported

  """
  def __init__(self, **kwargs):
    self.__hash = {}
    self.__hash["layers_shape"] = [[16], [64], [64], [4]]
    self.__hash["layers_activation_func"] = [None, tf2.keras.activations.relu,
                                             tf2.keras.activations.relu, None]
    self.__hash["layer_units"] = [16, 256, 256, 4]
    self.__hash["layers_type"] = [tf2.keras.layers.Dense, tf2.keras.layers.Dense,
                                  tf2.keras.layers.Dense,tf2.keras.layers.Dense]
    self.__hash["optimizer"] = tf2.keras.optimizers.Adam
    self.__hash["loss_function"] =  tf2.keras.losses.mse
    self.__hash["learning_rate"] = 0.0001
    self.__hash["gamma"] = 0.99
    self.__hash["epsilon"] = 1
    self.__hash["epsilon_dec"] = 1e-3
    self.__hash["epsilon_min"] = 0.01
    self.__hash["nn_type"] = "dqn"
    self.__hash["log_dir"] = "/app/logs"
    self.__hash["tf_model"] = None
    self.__hash["agent"] = None
    self.__hash["batch_size"] = 10000
    self.__hash["mem_size"] = 50000
    self.__hash["training_epochs"] = 1
    self.__hash["log_dir"] = "/app/logs"
    self.__hash["tf_proc_debug"] = False

    # Copy content in from the argument hash.
    for k in kwargs.keys():
      self.__hash[k] = kwargs[k]

    if self.get_param("tf_model") is None:
      self.__hash["tf_model"] = self.create_tf_model()

    if self.get_param("agent") is None:
      self.__hash["agent"] = agent.Agent(model=self.__hash["tf_model"],
                                         batch_size=self.__hash["batch_size"],
                                         lr=self.__hash["lr"],
                                         log_dir=self.__hash["log_dir"],
                                         gamma=self.__hash["gamma"],
                                         epsilon=self.__hash["epsilon"],
                                         epsilon_dec=self.__hash["epsilon_dec"],
                                         epsilon_min = self.__hash["epsilon_min"],
                                         tf_proc_debug=self.__hash["tf_proc_debug"])

    tf2.debugging.set_log_device_placement(self.__hash["tf_proc_debug"])


  def create_tf_model(self):
    layers = tf2.keras.layers
    l_arr = []
    l =None
    l_type = None
    unit_dims = None
    m_shape = self.get_param("layers_shape")
    act_func = None

    if len(m_shape) < 2: return None

    for i in range( len(self.get_param("layers_shape")) ):
      l_type = self.__hash["layers_type"][i]
      act_func = self.__hash["layers_activation_func"][i]
      unit_dims = self.__hash["layer_units"][i]
      l = l_type(unit_dims, activation=act_func)

      l_arr.append(l)

    mod = tf2.keras.models.Sequential(l_arr)
    opt = self.__hash["optimizer"]
    loss = self.__hash["loss_function"]
    mod.compile(optimizer=opt(learning_rate = self.__hash["learning_rate"]),\
                loss=loss)
    return mod

  def get_param(self, name):
    if name in self.__hash.keys(): return self.__hash[name]
    return None

  def set_param(self, name, value):
    self.__hash[name] = value


#--- END NN_Model
