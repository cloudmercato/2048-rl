import tensorflow as tf2

class ExecutionDataTBLogger(tf2.keras.callbacks.Callback):
  def __init__(self, tb_callback, model, **kwargs):
    self.tb_callback = tb_callback
    self.model = model

    self.__hash = {}
    self.__hash["log_dir"] = "/app/logs"
    for k in kwargs.keys(): self.__hash[k] = kwargs[k]

    file_writer = tf2.summary.create_file_writer( self.__hash["log_dir"] )
    file_writer.set_as_default()

  def on_epoch_begin(self, epoch, logs=None):
    keys = list(logs.keys())

  def on_epoch_end(self, epoch, logs=None):
    tf2.summary.scalar('mean', data=(self.model.metrics[0])(self.model), step=epoch)
