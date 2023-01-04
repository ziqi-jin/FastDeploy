# 模型精度批量验证脚本

本目录下的Python脚本可以在CPU/GPU/昆仑芯/昇腾,以及后续的新增硬件上, 完成对高优模型的精度批量验证.
各模型的精度测试代码是基于Python部署demo修改而成, 当后续有新增硬件或者新增模型时，用户可以通过同样的方式(新增option和模型),添加新的Python代码来完成精度验证.


## 用法

### 1.准备数据集
- 分类模型需要ImageNet验证集以及标签
- 检测模型需要COCO2017验证集以及标签
- 分割模型需要Cityscape验证集以及标签
- PP-OCRv2/v3的数据集在准备脚本中会自行下载.

请将准备好的数据集解压至dataset目录中使用

### 2.精度验证
分类/检测/分割/OCR四个场景的精度验证启用方式是一样的.
其中分类, 检测和分割模型会返回预测精度, OCR模型会返回与GPU预测结果的差异.

```bash
# 进入分类模型目录下
cd classification
# 执行prepare.sh脚本,自动下载并解压模型至models文件夹下
bash prepare.sh
# 首先修改run.sh中的TARGET_DEVICE为想测试的硬件,之后执行run.sh脚本
bash run.sh
# 验证完毕的输出以及精度数据,会保存至log文件夹下,用户自行查看
```