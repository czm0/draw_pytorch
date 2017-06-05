# draw

Pytorch implementation of [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623.pdf) on the MNIST generation task.

| With Attention   |
| -------------  |
| <img src="http://i.imgur.com/XfAkXPw.gif" width="100%">|


## Usage

`python train.py`  downloads the MNIST dataset to ./data/ and train the DRAW model with attention for both reading and writing. After training, the weights files is written to ./save/weights_final.tar and the generated images is written to ./image/count_test.png
