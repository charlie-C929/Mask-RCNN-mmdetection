## 基于MMdetection的Maskrcnn的实例语义分割与onnx模型的树莓派部署

![title](https://gitee.com/CN_13/images/raw/master/img/title.png)

********

[英文文档](README.md)|[配置文档](requirements.md)

- [首先需要对我们需要的环境进行配置](requirements.md).

### 如何使用?

#### (Python)将mmdetection的pth转化成onnx     

```shell
python deployment/pytorch2onnx.py \
    configs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py \
    checkpoints/latest.pth \
    --output-file result.onnx \
    --input-img deployment/color.jpg \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
```

#### (Python) 使用onnx_inference.py 进行onnx的推理

执行：

~~~shell
# 执行测试
python onnx_inference.py {your pic} --model {your onnx model}
~~~

#### (C++)使用onnx_inference.py 进行onnx的推理

运行：

```shell
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
./runDet {your pic} --model {your onnx model}
```

#### (C++)如何在Raspberry pi 3B/4B部署onnx进行推理?

- 您应该注意的一件事是，onnxruntime的发行版不能用于具有arvm7l架构的Raspberry Pi，您应该做的是从源代码重建onnxruntime。

- [从源代码进行编译onnxruntime](./interferencec++/config.md)进行准备。

- 下载[onnxruntime release](https://github.com/microsoft/onnxruntime/releases)并，使用你从源码编译出来的.so替换在/lib文件夹中的.so文件。

- 运行

```shell
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
./runDet {your pic} --model {your onnx model}
```



