import argparse
import nn_model
import agent

# Making sure we are running the main routine.
if __name__ != "__main__":
  exit()

learn_runs = 100

parser = argparse.ArgumentParser()


nn_mod = nn_model.NN_Model()
agent = agent.Agent(model=nn_mod.get_param("tf_model"))
agent.learn_on_repeat(learn_runs)



