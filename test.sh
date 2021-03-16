#!/bin/sh
if [ "$#" -ge 1 ] && [ -f "$1" ] ; then
	docker run --gpus all -it --rm --ipc=host -v $(pwd):/workdir \
		-w /workdir \
		pranet python MyTest.py \
		--img_path "$1"
else
	docker run --gpus all -it --rm --ipc=host -v $(pwd):/workdir \
		-v $(pwd)/../SINet/Dataset:/workdir/data \
		-w /workdir \
		pranet python MyTest.py --pth_path snapshots/PraNet_Res2Net_v3/PraNet-59.pth
fi
