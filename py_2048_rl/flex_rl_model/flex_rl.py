import argparse
import nn_model
import agent

# Making sure we are running the main routine.
if __name__ != "__main__":
  exit()

parser = argparse.ArgumentParser()
parser.add_argument('--learn_runs', type=int, default=100, help='Number of model learn runs')
parser.add_argument('--mem_size', type=int, default=10000,
                    help='Learning episode DB size (in number of episodes')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--log_dir', default="/app/logs",
                    help='Tensorboard log directory')

args = parser.parse_args()

nn_mod = nn_model.NN_Model(lr=args.lr, log_dir=args.log_dir)

agent = agent.Agent(model=nn_mod.get_param("tf_model"),
                    batch_size=args.mem_size,
                    lr=args.lr,
                    log_dir=args.log_dir)

agent.learn_on_repeat(args.learn_runs)



