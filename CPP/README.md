# 步骤

## 第一步: 编译

```sh
mkdir build && cd build
cmake ..
make
cd ..
```

## 第二步: 推理测试

```sh
./build/trt_sample ../models/resnet50.onnx ../testimages/cat.jpg
```

预测结果如下:

```
engine file size: 127507893 bytes
0 io
binding_size is: 602112
1 io
binding_size is: 4000
input_dims[0] is: 1, 3, 224, 224, 
(3, 224, 224)
class: Egyptian cat | confidence: 87.4756% | index: 285
```
