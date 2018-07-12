# faceswap-pytorch
Deepfakes_Faceswap using pytorch   *JUST FOR STUDY AND RESEARCH*

This is pytorch version compared with https://github.com/joshua-wu/deepfakes_faceswap which using Keras.

![Processing image](https://github.com/Oldpan/faceswap-pytorch/blob/master/Screenshot%20from%202018-04-16%2015-36-47.png)


#### Source code you can download directly from the github page.
### Source code,training images and trained model(~300MB):
https://space.oldpan.me/f/cccac1136338407797cb/?dl=1


## Requirement:
```
Python == 3.6
pytorch <= 0.4.0
```
pytorch-v0.4.0 is supported.
 You need a modern GPU and CUDA support for better performance.

## How to run:
`python train.py` for simple run

if you don't use trained model,only after about 1000 epoch can you see the result and after 10000 epoch the result is the same with the above picture.

ps : The Pytorch trains a little faster than Keras using tf backend.
