import argparse
import nn_model
import agent

# Making sure we are running the main routine.
if __name__ != "__main__":
  exit()

parser = argparse.ArgumentParser()
parser.add_argument('--learn_runs', type=int, default=100, help='Number of model learn runs')
args = parser.parse_args()

nn_mod = nn_model.NN_Model()
agent = agent.Agent(model=nn_mod.get_param("tf_model"))
agent.learn_on_repeat(args.learn_runs)



