## 任务描述

设计一个神经网络用于 CIFAR10数据集分类任务，并尽可能提高准确率。

## 环境配置

建议使用虚拟环境

```
conda create -n cv_project python=3.8 -y
conda activate cv_project
```

安装pytorch

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

项目依赖的工具包如下

```
conda install -c conda-forge matplotlib pandas scikit-learn tqdm pyyaml jupyter pillow
```

```
conda install -c conda-forge tensorboard
```

## 模型训练和测试

#### 参数设置

您可以在`models/model.py`中调整模型构架和激活函数等。

您可以在`train.py`中修改模型训练使用的损失函数，训练器和学习率衰减方法等。

通过在`train.py`中指定模型编号来指定模型训练结果的存储位置。

#### 模型训练

运行`train.py`即可进行训练，训练结果将保存在`model_data`文件夹下指定编号文件夹内，包含模型训练过程中的平均梯度，损失，最佳模型权重以及tensorboard文件。

#### 使用tensorboard查看模型训练结果

进入项目根目录后在终端中运行以下命令：

```python
# 启动TensorBoard
tensorboard --logdir=model_data
```

打开默认的链接 ` http://localhost:6000/` 即可查看训练结果。

