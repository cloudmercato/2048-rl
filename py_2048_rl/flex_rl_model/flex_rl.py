import argparse
import nn_model
import agent

# Making sure we are running the main routine.
if __name__ != "__main__":
  exit()

parser = argparse.ArgumentParser()
parser.add_argument('--learn_runs', type=int, default=100, help='Number of model learn runs')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Learning episode DB size (in number of episodes')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--log_dir', default="/app/logs",
                    help='Tensorboard log directory')
parser.add_argument('--training_epochs', type=int, default=1,
                    help='Number of epoch rns for every model training run')
parser.add_argument('--tf_proc_debug', default=False, type=bool,
                    help='Determines whether TF compute device info is displayed. Default = False')

args = parser.parse_args()

nn_mod = nn_model.NN_Model(batch_size = args.batch_size,
                           lr=args.lr,
                           taining_epochs=args.training_epochs,
                           log_dir=args.log_dir,
                           tf_proc_debug=args.tf_proc_debug)

agent = nn_mod.get_param("agent")

agent.learn_on_repeat(args.learn_runs)



