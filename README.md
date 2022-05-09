## 基于MMdetection的Maskrcnn的实例语义分割与onnx模型的树莓派部署

![image](./pics/title.png)

#### 通用的环境配置

**1.Build custom operators for ONNX Runtime**
在项目文件夹下通过git下载mmcv，例如：

```python
# 切换到项目文件夹的路径
cd /media/team515/xia/astudy/Deep_learning_code/Mask-RCNN-mmdetection
# 先安装下载工具git
apt-get install git
# 通过git下载mmcv的包
git clone https://github.com/open-mmlab/mmcv.git
```

下载onnxruntime-linux ， 把 ONNXRUNTIME_DIR 添加到系统环境变量里：

```python
# 先安装下载工具wget
sudo apt install -y wget unzip
# 下载onnxruntime-linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
# 解压缩onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
# 打开系统的环境变量文件.bashrc
vim ~/.bashrc
# 按方向键↓，一直到文件最低端，按i对.bashrc进行编辑，将下面的两句话粘贴上去，""中的路径请替换成自己的
export ONNXRUNTIME_DIR="/media/team515/xia/astudy/Deep_learning_code/Mask-RCNN-mmdetection/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
# 按Esc退出编辑，按":"，再输入"wq"回车，保存修改
# 使环境变量生效
source ~/.bashrc
```

从源码安装mmcv：

```python
# 切换到mmcv的安装目录
cd mmcv
pip install onnxruntime==1.8.1
pip install onnx
# 从源码安装mmcv
MMCV_WITH_OPS=1 MMCV_WITH_ORT=1 python setup.py develop
pip install scikit-learn
```

**2.安装mmdetction**

```python
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
pip install mmdet
```

**3.mmdetection pytorch权重转换成onnx**

```python
python deployment/pytorch2onnx.py \
    configs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py \
    checkpoints/latest.pth \
    --output-file result.onnx \
    --input-img deployment/color.jpg \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
```

#### (python) onnx_inference.py的使用      

将onnx_inference.py中的onnx_file替换成.onnx文件输出的目录，例如“./result.onnx”，将测试图片的输入路径修改成自己的。然后执行：

~~~python
# 执行测试
python onnx_inference.py
~~~

#### (c++)runDet.cpp的使用方法

首先，安装cmake。在项目文件夹下：

```python
sudo apt install -y g++
sudo apt install -y cmake
```

安装opencv：

```python
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mv opencv-4.x opencv
mkdir -p build && cd build
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
cmake ../opencv
make -j4
sudo make install
```

运行：

```python
# 回到项目文件夹下
cd ..
# 进入inference_C++文件夹下
cd inference_C++
# 创建build
mkdir build
cd build
cmake ..
make
# 运行生成的可执行文件runDet
./runDet
```

