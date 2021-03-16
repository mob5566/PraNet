FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install thop scipy==1.2.0 tensorboard future

WORKDIR /workdir

COPY res2net50_v1b_26w_4s-3cf99910.pth .
COPY PraNet-19.pth ./snapshots/PraNet_Res2Net/
COPY lib lib/
COPY utils utils/
COPY My* ./

ENTRYPOINT ["python", "./MyTest.py"]
