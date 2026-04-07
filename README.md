# 🍀 BQAE-TV
BQAE-TV is a Bayesian quaternion deep image inpainting model for recovering structurally coherent and visually natural color images from a single corrupted input.

# 🔧 Installation
* Install [Python 3.7](https://www.python.org/downloads/) on Linux or Windows.
* This project is implemented based on TensorFlow 2.x, so please make sure to install a compatible TensorFlow 2.x version in your environment.
* If you want to run on a GPU, you will need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Please refer to their official websites for the corresponding compatible versions.

# 🚂 Running BQAE-TV
Simply open BQAE-TV.py and run it to achieve the desired color image recovery performance.

# 📖 Article
For more details about the methodology and experimental results, please refer to the paper:

**A Bayesian deep prior-based quaternion matrix completion for color image inpainting**  

# 🔔 Note
The input to a quaternion deep model is usually constructed by appending an all-zero real part. In our recent investigation, however, we considered an alternative strategy that uses a conventional convolution operation in the first layer.
