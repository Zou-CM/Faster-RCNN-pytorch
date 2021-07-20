### 准备数据集

本工程用的是VOC2012，我把train和test里的标签和图片拿了出来，放在了最外层(imgs对应原数据中的JPEGImages，我也不记得为啥要改个名字。。)

```
工程根目录
......
--train
----Annotations
----imgs
--test
----Annotations
----imgs
......
```

如果没有找到VOC2012数据集可以点击链接: https://pan.baidu.com/s/1JKbj_IZ0_G2Qi48Jsv0Y6A  密码: os5n

准备好数据集后，运行utils/DataProcess.py，主要是读取一下xml文件，把相应的坐标标签信息保存成npy，方便训练时读取。

### 修改参数

在Config.py中可以修改相关参数，比如anchor个数、尺寸、长宽比、batchsize、backbone等。

### 训练模型

运行train.py，如果想实时查看损失变化，把注释部分打开，调用tensorboard进行监视，应该是需要安装tensorflow。

### 模型校验

训练好的模型会存放在checkpoints/FasterRCNN下，校验模型直接运行eval.py，测试结果会保存在dev_img下（原图和处理后的都在）

### 大概效果

由于显卡比较垃圾，没训练多少轮，batch_size也只能到2，大概练练，效果如下：

![2008_000014_detect](/home/zcm/Desktop/2008_000014_detect.jpg)