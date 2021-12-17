import os
import argparse
import logging

import tensorflow as tf

from py_2048_rl.agent import Agent


logger = logging.getLogger('py2048')
tf_logger = logging.getLogger('tensorflow')

parser = argparse.ArgumentParser()
parser.add_argument('action', default='train', choices=('train', 'infer'), nargs='?')
parser.add_argument('--learn-runs', type=int, default=100, help='Number of model learn runs')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Training batch selection size (in number of episodes')
parser.add_argument('--mem_size', type=int, default=50000,
                    help='Learning episode DB size (in number of episodes')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Gamma')
parser.add_argument('--gamma1', type=float, default=0.99,
                    help='Gamma1')
parser.add_argument('--gamma2', type=float, default=0.99,
                    help='Gamma2')
parser.add_argument('--gamma3', type=float, default=0.99,
                    help='Gamma3')
parser.add_argument('--epsilon', type=float, default=1.0,
                    help='Epsilon')
parser.add_argument('--epsilon-min', type=float, default=0.01,
                    help='Epsilon - start value')
parser.add_argument('--epsilon-dec', type=float, default=0.001,
                    help='Epsilon - step value')

parser.add_argument('--model-load-file', default=None,
                    help='Model load file path (h5)')
parser.add_argument('--model-save-file', default='model.h5',
                    help='Model save file path (h5)')
parser.add_argument('--training-epochs', type=int, default=1,
                    help='Number of epoch rns for every model training run')
parser.add_argument('--model-disable-autosave', default=True, action="store_false",
                    dest="model_auto_save")

parser.add_argument('--log-dir', default=None,
                    help='Tensorboard log directory')
parser.add_argument('--tf-log-device', default=False, action="store_true",
                    help='Determines whether TF compute device info is displayed.')
parser.add_argument('--tf-dump-debug-info', default=False, action="store_true")
parser.add_argument('--tf-profiler-port', default=0, type=int)
parser.add_argument('--verbose', '-v', default=3, type=int)
parser.add_argument('--tf-verbose', '-tfv', default=2, type=int)


def main():
    args = parser.parse_args()

    log_verbose = 60 - (args.verbose*10)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_verbose)
    logger.addHandler(log_handler)
    logger.setLevel(log_verbose)

    tf_log_verbose = 60 - (args.tf_verbose*10)
    tf_logger.setLevel(tf_log_verbose)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', str(args.tf_verbose))

    logger.debug('Config: %s', vars(args))

    tf.debugging.set_log_device_placement(args.tf_log_device)
    if args.log_dir and args.tf_dump_debug_info:
        tf.debugging.experimental.enable_dump_debug_info(
            args.log_dir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
    if args.tf_profiler_port:
        tf.profiler.experimental.server.start(args.tf_profiler_port)

    agent = Agent(
        batch_size=args.batch_size,
        mem_size=args.mem_size,
        # input_dims=args.input_dims,
        input_dims=[16],
        lr=args.lr,
        gamma=args.gamma,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        gamma3=args.gamma3,
        epsilon=args.epsilon,
        epsilon_dec=args.epsilon_dec,
        epsilon_min=args.epsilon_min,
        model_load_file=args.model_load_file,
        model_save_file=args.model_save_file,
        model_auto_save=args.model_auto_save,
        log_dir=args.log_dir,
        training_epochs=args.training_epochs,
    )

    if args.action == 'train':
        agent.learn_on_repeat(args.learn_runs)
    elif args.action == 'infer':
        agent.play_on_repeat(args.learn_runs)

if __name__ == "__main__":
    main()
