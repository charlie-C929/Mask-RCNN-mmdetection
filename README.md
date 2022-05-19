## MMdetection-Maskrcnn-Onnx-Raspberry

![title](https://gitee.com/CN_13/images/raw/master/img/title.png)

********

[中文文档](README_zhCN.md)|[配置文档](requirements.md)

- [First you need to follow the requiremets.md to build the enviroment we need](requirements.md).

### How to use?

#### (Python)mmdetection pth convert to onnx     

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

#### (Python) Using onnx_inference.py  to do inference with onnx

Run：

~~~shell
# 执行测试
python onnx_inference.py {your pic} --model {your onnx model}
~~~

#### (C++)Using onnx_inference.py  to do inference with onnx

Run：

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

#### (C++)How to inference on the Raspberry pi 3B/4B?

- One thing you should notice is that the release of onnxruntime can't be use on the Raspberry Pi with the architecture of  arvm7l, what you should do is rebuild the onnxruntime from the sourse code.

- [Build the onnxruntime from sourse](./inference_C++/config.md) for preparation.
- Download the [release of onnxruntime](https://github.com/microsoft/onnxruntime/releases) and replace the .so files in /lib folder with the .so files you just built from the preparation.

- Run

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

