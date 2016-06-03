# SqueezeNet Keras Implementation
This is the Keras implementation of SqueezeNet using functional API (arXiv [1602.07360](https://arxiv.org/pdf/1602.07360.pdf)).
SqueezeNet is a small model of AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
The original model was implemented in [caffe](https://github.com/DeepScale/SqueezeNet).

## Reference
[pysqueezenet by yhenon](https://github.com/yhenon/pysqueezenet)

Differences:
* Switch from Graph model to Keras 1.0 functional API
* Fix the bug of pooling layer 

## Model Visualization
![](model.png)
