1. mmdetection pytorch权重转换成onnx

参考网址：[https://mmdetection.readthedocs.io/en/v2.21.0/tutorials/pytorch2onnx.html](https://mmdetection.readthedocs.io/en/v2.21.0/tutorials/pytorch2onnx.html)

命令：

~~~python
python deployment/pytorch2onnx.py \
    configs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py \
    checkpoints/latest.pth \
    --output-file result.onnx \
    --input-img deployment/color.jpg \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
~~~

2. (python) onnx_inference.py的使用方法（强依赖，依赖mmcv的ops编译的动态链接库文件，依赖onnx的连接库）

- 环境准备（需要编译mmcv并配置onnxruntime）

（linux）https://github.com/open-mmlab/mmcv/blob/master/docs/en/deployment/onnxruntime_op.md/#how-to-build-custom-operators-for-onnx-runtime

- 推理脚本使用方式：

使用脚本之前修改权重文件以及测试图片路径

~~~python
python onnx_inference.py
~~~

__该方法只适用于Linux系统__

3. (c++)runDet.cpp的使用方法（弱依赖，onnx的连接库）

### 准备工作

- cmake（require）

window：官网下载，安装，安装时添加路径

https://cmake.org/download/#latest

- opencv（require）

windows：官网下载，安装

https://sourceforge.net/projects/opencvlibrary/files/opencv-win/

打开命令提示符:

~~~
setx -m OPENCV_DIR D:\OpenCV\Build\
~~~

其中后面的路径为实际安装的路径到build目录.

将 <Opencv安装位置>\build\x64\vc15\bin 和 <Opencv安装位置>\build\x64\vc15\lib添加到系统路径Path中。

- onnxruntime（required）

windows：https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1

下载解压，将解压完的路径添加新建变量名为ONNXRUNTIME_DIR。

### 运行部分

进入项目inference_C++文件夹：

新建build文件夹。

打开命令提示符。

~~~cmd
cd build
cmake ..
~~~

生成inference_C++.sln项目文件。

使用MSVC2019打开sln文件，右键项目生成，即可在Debug/release文件夹下生成可执行程序。

__注意__!运行之前需将

~~~
<onnxruntime解压的文件夹>\lib\onnxruntime.dll 
~~~

与执行文件放在同一目录下。

双击运行runDet.exe.

linux：

- opencv（require）

https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

- onnxruntime

~~~shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
~~~

### 运行部分

进入项目inference_C++文件夹.

~~~cmd
mkdir build
cd build
cmake ..
make
~~~

即可生成可执行程序。

~~~
./runet
~~~

查看结果。

