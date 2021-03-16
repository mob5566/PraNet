#!/bin/sh
docker run --gpus all -it --rm --ipc=host -v $(pwd):/workdir \
	-v $(pwd)/../SINet/Dataset:/workdir/data \
	-w /workdir \
	pranet python MyTrain.py \
		--train_path data/TrainDataset \
		--train_save PraNet_Res2Net_v3 \
		--epoch 200
