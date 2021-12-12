FROM tensorflow/tensorflow 

WORKDIR /2048-rl
ADD . /2048-rl

RUN pip install https://github.com/cloudmercato/2048-game/archive/refs/heads/master.zip
RUN pip install .
