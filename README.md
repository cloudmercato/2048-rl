# 2048-rl

Deep Q-Learning Project to play 2048.
See [this presentation](https://docs.google.com/presentation/d/1I9RS3SMdMp8Uk9C6eyS6jK_w_34BKCrvkN-kWau1MU4/edit?usp=sharing) for an introduction.

## Getting Started

Install [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/index.html), python & pip.
Then, run:

```bash
pip install -r requirements.txt
pip install https://github.com/cloudmercato/2048-game/archive/refs/heads/master.zip
python setup.py
```

Now, you should be able to run the tests:

```bash
python -m unittest py_2048_rl.tests
```

## Run training

```bash
python -m py_2048_rl.console --help
```

## Source Code Structure

All python source code lives in `py_2048_rl`.
