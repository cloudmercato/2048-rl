FROM tensorflow/tensorflow 

WORKDIR /2048-rl
ADD . /2048-rl

RUN mkdir -p /log_dir/
RUN mkdir -p /models/

RUN pip install https://github.com/cloudmercato/2048-game/archive/refs/heads/master.zip
RUN python setup.py develop

VOLUME /log_dir/
VOLUME /models/

CMD ["2048-rl", "--log-dir=/log_dir/", "--model-save-file=/models/current.h5", "--tf-profiler-port=6007"]

EXPOSE 6006/TCP
EXPOSE 6007/TCP
