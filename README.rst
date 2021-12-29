2048 Reinforcement learning
===========================

Deep Q-Learning Project to play 2048, inspired from `georgwiese/2048-rl`_. See `this presentation`_ for an introduction.


The game class is stored in `cloudmercato/2048-game`_.

Install
-------

::

    pip install -r requirements.txt
    pip install https://github.com/cloudmercato/2048-game/archive/refs/heads/master.zip
    python setup.py install

Console
-------

The main way to use this repository is its CLI ``2048-rl``: ::

    $ 2048-rl --help
    usage: 2048-rl [-h] [--runs RUNS] [--games-per-cycle GAMES_PER_CYCLE]
                   [--batch-size BATCH_SIZE] [--mem-size MEM_SIZE] [--lr LR]
                   [--gamma GAMMA] [--gamma1 GAMMA1] [--gamma2 GAMMA2]
                   [--gamma3 GAMMA3] [--epsilon EPSILON]
                   [--epsilon-min EPSILON_MIN] [--epsilon-dec EPSILON_DEC]
                   [--model-path MODEL_PATH] [--model-load-file MODEL_LOAD_FILE]
                   [--model-save-file MODEL_SAVE_FILE]
                   [--training-epochs TRAINING_EPOCHS] [--model-disable-autosave]
                   [--disable-collect-random-data] [--refill-episode-db]
                   [--log-dir LOG_DIR] [--tf-log-device] [--tf-dump-debug-info]
                   [--tf-profiler-port TF_PROFILER_PORT] [--verbose VERBOSE]
                   [--tf-verbose TF_VERBOSE]
                   [{train,infer}]

    positional arguments:
      {train,infer}

    optional arguments:
      -h, --help            show this help message and exit
      --runs RUNS           Number of runs
      --games-per-cycle GAMES_PER_CYCLE
                            Number of games per model learn run/cycle
      --batch-size BATCH_SIZE
                            Training batch selection size (in number of episodes
      --mem-size MEM_SIZE   Learning episode DB size (in number of episodes
      --lr LR               Learning rate
      --gamma GAMMA         Gamma
      --gamma1 GAMMA1       Gamma1
      --gamma2 GAMMA2       Gamma2
      --gamma3 GAMMA3       Gamma3
      --epsilon EPSILON     Epsilon
      --epsilon-min EPSILON_MIN
                            Epsilon - start value
      --epsilon-dec EPSILON_DEC
                            Epsilon - step value
      --model-path MODEL_PATH
                            Python path to the model to compile
      --model-load-file MODEL_LOAD_FILE
                            Model load file path (h5)
      --model-save-file MODEL_SAVE_FILE
                            Model save file path (h5)
      --training-epochs TRAINING_EPOCHS
                            Number of epoch rns for every model training run
      --model-disable-autosave
      --disable-collect-random-data
      --refill-episode-db
      --log-dir LOG_DIR     Tensorboard log directory
      --tf-log-device       Determines whether TF compute device info is
                            displayed.
      --tf-dump-debug-info
      --tf-profiler-port TF_PROFILER_PORT
      --verbose VERBOSE, -v VERBOSE
      --tf-verbose TF_VERBOSE, -tfv TF_VERBOSE


Basically you can:

- Train (train) a model, record it, reload an existing one
- Solve (infer) game with an existing model

.. _georgwiese/2048-rl: https://github.com/georgwiese/2048-rl
.. _this presentation: https://docs.google.com/presentation/d/1I9RS3SMdMp8Uk9C6eyS6jK_w_34BKCrvkN-kWau1MU4/edit?usp=sharing
.. _cloudmercato/2048-game: https://github.com/cloudmercato/2048-game
