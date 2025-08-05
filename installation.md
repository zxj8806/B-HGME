# BHGME Installation

First, create a basic conda environment with python 3.8

```
conda create --name BHGME python=3.8
```

Now, activate your environment and utilize the requirements.txt file to install non pytorch dependencies

```
conda activate BHGME
pip install -r requirements.txt
```

After that installation make sure you have numpy==1.24.1 installed, and reinstall with pip if necessary. Now, to install pytorch dependencies, we will use the following pip wheels (CUDA 12.1).

```
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Finally, we will install the pytorch-geometric dependencies.

```
pip install torch-geometric==2.6.1
pip install torch-cluster==1.6.3+pt24cu121 torch-scatter==2.1.2+pt24cu121 torch-sparse==0.6.18+pt24cu121 torch-spline-conv==1.2.2+pt24cu121
```
