import tensorflow as tf2

class Logger():
  def __init__(self, **kwargs):
    pass

  def generic_output(self, **kwargs):
    if "field_names" not in kwargs.keys(): return

    field_names = kwargs["field_names"]

    if "field_content" not in kwargs.keys(): return

    field_content = kwargs["field_content"]

    print("Timestamp: " + tf2.timestamp().numpy().__str__())

    for i in range( len(field_names) ):
      print(field_names[i] + " : " + field_content[i])

