# draw-pytorch

Pytorch implementation of [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623.pdf) on the MNIST generation task.

| With Attention   |
| -------------  |
| <img src="http://i.imgur.com/XfAkXPw.gif" width="100%">|


## Usage

`python train.py`  downloads the MNIST dataset to ./data/mnist and train the DRAW model with attention for both reading and writing. After training, the weights files are written to ./save/weights_final.tar and the generated images are written to ./image/.png

`python generate.py`	 loads wieghts from save/weights_final.tar  and generates images

The weights_final.tar file is trained for 50 epoch with minibatch size 64 on GTX 1080 GPU.

## Reference
https://github.com/ericjang/draw