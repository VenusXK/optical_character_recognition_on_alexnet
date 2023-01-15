<!-- <div align=center>
<img decoding="async" src="./Readme_File/Poster_matplotlib.png" width="40%" >
</div> -->

# 基于卷积神经网络的印刷体识别 `Typographical-OCR`

### 简介
&emsp;&emsp;这是我在*中国石油大学(华东)机器学习* 课程的项目作业

&emsp;&emsp;This is my assignment of undergraduate education in China University of Petroleum (East China)
### 目录
- [基于卷积神经网络的印刷体识别 `Typographical-OCR`](#基于卷积神经网络的印刷体识别-typographical-ocr)
    - [简介](#简介)
    - [目录](#目录)
  - [一、研究意义 `Research significance`](#一研究意义-research-significance)
  - [二、数据描述 `Dataset description`](#二数据描述-dataset-description)
    - [1. 数据获取途径 `Access to dataset`](#1-数据获取途径-access-to-dataset)
    - [2. 使用数据内容组成 `Composition of dataset`](#2-使用数据内容组成-composition-of-dataset)
    - [3. 数据生成过程 `Dataset generation process`](#3-数据生成过程-dataset-generation-process)
    - [4. 数据处理后的结果 `Results of data processing`](#4-数据处理后的结果-results-of-data-processing)
  - [三、模型描述 `Model description`](#三模型描述-model-description)
    - [1. 数据处理流程 `Data processing processes`](#1-数据处理流程-data-processing-processes)
    - [2. 用到的算法 `The algorithm used`](#2-用到的算法-the-algorithm-used)

## 一、研究意义 `Research significance`
&emsp;&emsp;汉字作为中华民族文明发展的信息载体，已有数千年的历史，也是世界上使用人数最多的文字，它记录了璀璨的民族文化，展示了东方民族独特的思维和认知方法。随着计算机技术的推广应用，尤其是互联网的日益普及，人类越来越多地以计算机获得各种信息，大量的信息处理工作也都转移到计算机上进行。在日常生活和工作中，存在着大量的文字信息处理问题，因而将文字信息快速输入计算机的要求就变得非常迫切。现代社会的信息量空前丰富，其中绝大部分信息又是以印刷体的形式进行保存和传播的，这使得以键盘输入为主要手段的计算机输入设备变得相形见绌，输入速度低已经成为信息进入计算机系统的主要瓶颈，影响着整个系统的效率。

&emsp;&emsp;As the information carrier for the development of Chinese civilization, Chinese characters have a history of thousands of years and are also the most used script in the world, which records the brilliant national culture and shows the unique thinking and cognitive methods of Eastern peoples. With the promotion and application of computer technology, especially the increasing popularity of the Internet, human beings are increasingly using computers to obtain various information, and a large number of information processing work has also been transferred to computers. In daily life and work, there are a large number of text information processing problems, so the requirement to quickly input text information into the computer has become very urgent. The amount of information in modern society is unprecedentedly rich, and most of the information is preserved and disseminated in the form of printed form, which makes the computer input device with keyboard input as the main means dwarfed, and the low input speed has become the main bottleneck of information entering the computer system, affecting the efficiency of the entire system.

&emsp;&emsp;一方面，使用该技术可以提高计算机的使用效率，克服人与机器的矛盾。另一方面，该技术可以应用于快速识别身份证、银行卡、驾驶证等卡证类信息，将证件文字信息直接转换为可编辑文本，可以大大提高山东省相关部门工作效率，减少人力劳动成本，可以实时进行相关人员的身份核验，以便山东省各部门安全管理。

&emsp;&emsp;On the one hand, the use of this technology can improve the efficiency of computer use and overcome the contradiction between man and machine. On the other hand, this technology can be applied to quickly identify ID cards, bank cards, driver`s licenses and other card information, and directly convert the text information of the certificate into editable text, which can greatly improve the work efficiency of relevant departments in Shandong Province, reduce labor costs, and verify the identity of relevant personnel in real time for the safety management of various departments in Shandong Province.

## 二、数据描述 `Dataset description`

### 1. 数据获取途径 `Access to dataset`
&emsp;&emsp;利用 `Windows` 自带的字体文件库，用 `Python` 的 `PIL` 库绘图，每张图片上绘制一个文字，总共绘制 `3755` 个汉字，

&emsp;&emsp;Using the font file library that comes with WINDOWS, drawing with Python`s PIL library, drawing a text on each picture, drawing a total of 3755 Chinese characters.

### 2. 使用数据内容组成 `Composition of dataset`

&emsp;&emsp;根据汉字国标库绘图，同一字体生成 `976` 张图片作为训练集，生成 `244` 张图片作为训练中准确率的测试集，总共 `3755` 个汉字，训练集（不包含验证集）共 `3664880` 个文件，每个汉字有对应 `876` 张训练集图片和 `244` 张验证集图片，根据 `AlexNet` 要求每张图片大小应为 `227*227` 。

&emsp;&emsp;According to the Chinese character national standard library drawing, the same font generates 976 pictures as a training set, generates `244` pictures as a test set of accuracy in training, a total of 3755 Chinese characters, a training set (excluding verification set) a total of 3664880 files, each Chinese character has a corresponding 876 training set pictures and 244 verification set pictures, according to AlexNet requirements each picture size should be 227*227.

### 3. 数据生成过程 `Dataset generation process`
 1. 首先定义输入参数，其中包括输出目录、字体目录、测试集大小、图像尺寸、图像旋转幅度等等。

    Start by defining the input parameters, which include the output directory, font directory, test set size, image size, image rotation, and so on.

1. 接下来将得到的汉字与序号对应表读入内存，表示汉字序号到汉字的映射，用于后面的字体生成。

    Next, the table corresponding to the obtained Chinese characters and ordinal numbers is read into memory, indicating the mapping of IDs to Chinese characters, which is used for later font generation.

3. 我们对图像进行一定角度的旋转，将旋转角度存储到列表中，旋转角度的范围是 `[-rotate, rotate]` 。

    We rotate the image at an angle and store the rotation angle in a list with the range of rotation angles [-rotate, rotate].

4. 字体图像的生成使用的工具是 `Python` 自带的 `PIL` 库。该库里有图片生成函数，用该函数结合字体文件，可以生成我们想要的图片化的汉字。设定好生成的字体颜色为黑底白色，字体尺寸由输入参数来动态设定。

    The tool used to generate font images is Python`s built-in PIL library. There is an image generation function in this library, and with this function combined with font files, we can generate the Chinese characters we want to be picturesque. Set the generated font color to white on black, and the font size is dynamically set by input parameters.

5. 同时，我们对生成的字体进行适当的膨胀和腐蚀，以扩大训练集数量。

    At the same time, we do appropriate bloat and corrosion of the generated fonts to expand the number of training sets.

6. 执行如下指令，开始生成印刷体文字汉字集。

    Execute the following command to start generating a set of Chinese characters in printed characters.

    ```shell
    python gen_printed_char.py --out_dir [out_put_dir] --font_dir [windows_font_dir] --width [img_width] --height [img_height] --margin 4 --rotate 30 --rotate_step 1
    ```

7. 若生成 `227*227` 大小的图片，在 `2060` 显卡下总共生成时间近 `16` 小时，训练集共 `3664880` 个文件。

    If an image of 227*227 size is generated, the total generation time under 2060 is nearly 16 hours, and the training set has a total of 3664880 files.

### 4. 数据处理后的结果 `Results of data processing`

<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果1.png" width="80%" >

**图1** 训练集（不包含验证集）共3755个文字，总共3664880个文件

**Figure 1** The training set (excluding the validation set) contains 3755 characters and 3664880 files

</div>

<br>

<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果1.png" width="80%">

**图2** 训练集每个汉字共876张图片

**Figure 2** A total of 876 pictures of each Chinese character in the training set</div>

<br>

<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果2.png" width="80%" >

**图3** 验证集每个汉字共244张图片

**Figure 3** A total of 244 images for each Chinese character in the verification set</div>

## 三、模型描述 `Model description`
### 1. 数据处理流程 `Data processing processes`
1.	AlexNet对图像大小要求为 `227*227` ，生成图像时已设置图像大小为 `227*227` ；

    AlexNet has an image size requirement of 227\*227, and the image size has been set to 227\*227 when generating images;

2.	使用 `pytorch` 的 `ImageFolder` 库进行图像文件的选择， `ImageFolder` 假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名；

    Use pytorch`s ImageFolder library for image file selection, ImageFolder assumes that all files are saved in folders, and the same category of pictures is stored under each folder, and the folder name is class name;


3.	由于使用 `pytorch` 框架，需要将图像转换为 `Tensor`（张量）数据结构，利用 `torchvsion` 下面的 `transforms` 库将输入图片转换为 `Tensor` 格式，语句如下：

    Since the `pytorch` framework is used, the image needs to be converted to a `tensor` data structure, and the `transforms` library under `torchvsion` is used to convert the input image to `tensor` format with the following statement:

    ```py
    transforms = transforms.Compose([transforms.ToTensor()])
    data_test = ImageFolder(path_test, transform=transforms)
    ```
4.	使用 `pytorch` 的 `DataLoader` 库进行训练集的导入，此库导入训练集为成批导入，每一批既导入图像又导入标签，训练时根据每一批图片进行训练，通过设置 `batch_size` 可以控制每一批导入数据的量， `batch_size` 越小训练的轮次越多，本次实验设置 `batch_size` 为 `32` ，通过 `DataLoader` 导入训练集大致情况如下图所示：

    Use pytorch`s DataLoader library for the import of the training set, this library imports the training set as batch import, each batch both imports images and labels, training according to each batch of pictures for training, through the setting batch_size you can control the amount of data imported into each batch, the smaller the batch_size, the more rounds of training, the batch_size of this experiment is set to 32, The following figure shows the general situation of importing the training set through DataLoader:

<div align=center>
<img decoding="async" src="./Readme_File/DataLoader.png" width="80%">

**图4** 一个batchsize输入的图像样例

**Figure 4** An example of an image of a batchsize input</div>

<br>

5. 本次训练由于数据集较大，使用 `gpu` 加速，通过安装 `nvidia` 的 `cuda` 模块并下载 `pytorch` 的 `cuda 11.6` 版本配置 `pytorch` 环境，通过 `torch.cuda.is_available` 判断 `cuda` 是否可用，在可用的基础上通过 `device = torch.device("cuda")` 语句设置 `gpu` 硬件，并对通过 `DataLoader` 导入的每一批训练集数据通过 `images = images.to(device)` 和 `labels = labels.to(device)` 语句将训练集数据导入 `gpu` 。

    This training uses GPU acceleration due to the large data set, configures the PyTorch environment by installing NVIDIA`s CUDA module and downloading the CUDA 11.6 version of PyTorch, determines whether CUDA is available through the torch.cuda.is_available, and sets the GPU hardware through the device = Torch.Device ("CUDA") statement on the basis of availability. For each batch of training set data imported through DataLoader, import the training set data into the GPU through images = images.to (device) and labels = labels.to (device) statements.

### 2. 用到的算法 `The algorithm used`
1. `AlexNet`卷积神经网络结构
    <div align=center>
    <img decoding="async" src="./Readme_File/AlexNet.png" width="80%">

    **图5** AlexNet卷积神经网络结构

    **Figure 5** AlexNet卷积神经网络结构</div>

    <br>

    1. **`AlexNet` 网络第一个卷积层 `The first convolutional layer of the AlexNet`**
    
        输入的图片大小为: `224*224*3` ，为后续处理方便，普遍改为 `227*227*3` ，第一个卷积层为: `11*11*3` ，卷积核尺寸为 `11*11` ,有 `96` 个卷积核，卷积步长为 `4` ，卷积层后使用 `ReLU` 激活函数,输出的尺寸为 `（227-11）/4+1=55` ，其输出的每个特征图大小为 `55*55*96`；

        The input image size is: 224\*224\*3, for the convenience of subsequent processing, it is generally changed to 227\*227\*3, the first convolutional layer is: 11\*11\*3, the convolution kernel size is 11\*11, there are 96 convolution kernels, the convolution step size is 4, the ReLU activation function is used after the convolutional layer, the output size is (227-11)/4+1=55, and the size of each feature map output is 55\*55\*96;

        最大池化层的池化核大小为 `3*3` ,步长为 `2` ,输出的尺寸为  `（55-3）/2+1=27` ，因此特征图的大小为: `27*27*96` 。

        The size of the pooled kernel of the maximum pooling layer is 3\*3, the step size is 2, and the output size is (55-3)/2+1=27, so the size of the feature map is: 27\*27\*96.

    2. **`AlexNet` 网络第二个卷积层 `The second convolutional layer of the AlexNet`**

        输入的数据为 `27*27*96` ，数据被 `256` 个大小为 `5*5*96` 的卷积核进行卷积运算,步长为 `1` ,填充值为 `2` ,卷积后使用 `ReLU` 层进行处理；
        
        The input data is 27\*27\*96, and the data is convolved by 256 convolution kernels of size 5\*5\*96, with a step of 1 and a fill value of 2, which is processed using the ReLU layer after convolution;
    
        最大池化层,核大小为 `3*3` ,步长为 `2` ；

        The maximum pooling layer has a kernel size of 3\*3 and a step size of 2;
    
    3. **`AlexNet` 网络第三层至第五层卷积层 `The third to the fifth convolutional layer of the AlexNet`**
    
        第三层每组数据被尺寸为 `3*3*384` 的卷积核进行卷积运算,步长为 `1` ,填充值为 `1` ，卷积后使用 `ReLU` 层进行处理；

        Each set of data in the third layer is convolutioned by a convolution kernel with a size of 3\*3\*384, with a step size of 1 and a fill value of 1, and the ReLU layer is used for processing after convolution.
    
        第四层每组数据被尺寸大小为 `3*3*384` 的卷积核卷积运算,步长为 `1` 填充值为 `1` ，卷积后使用 `ReLU` 层进行处理；

        Each set of data in the fourth layer is processed by a convolution kernel convolution with a size of 3\*3\*384, with a step of 1 and a fill value of 1, and the ReLU layer is used for processing after convolution;
    
        第五层每组数据都被尺寸大小为 `3*3*256` 的卷积核进行卷积运算,步长为 `1` ,填充值为 `1` ，卷积后使用 `ReLU` 层进行处理；

        Each set of data in the fifth layer is convolutioned by a convolution kernel with a size of 3\*3\*256, with a step size of 1 and a fill value of 1, and then processed using the ReLU layer after convolution;
    
        经过 `3*3` 池化窗口，步长为 `2` ，池化后输出像素层；

        After the 3\*3 pooling window, the step size is 2, and the pixel layer is output after pooling;


    4. **`AlexNet` 网络第六层至第八层全连接层 `layer 6 to layer 8 full connectivity of the AlexNet`**
    
        第六层首先以 `0.5` 的概率舍弃数据，经过共 `4096` 个神经元处理，之后经过 `ReLU` 层处理；

        The sixth layer first discards the data with a probability of 0.5, after a total of 4096 neuron processing, and then through the ReLU layer;
    
        第七六层首先以 `0.5` 的概率舍弃数据，输入 `4096` 个特征值，输出 `4096` 个特征值，之后经过 `ReLU` 层处理；

        The seventh and sixth layers first discard the data with a probability of 0.5, input 4096 eigenvalues, output 4096 eigenvalues, and then process it through the ReLU layer;
    
        第八层输入 `4096` 个特征值，输出 `3755` 个特征值。

        The eighth layer inputs 4096 eigenvalues and outputs 3755 eigenvalues.

2. 梯度下降法 `SGD`
3. `Adam` 优化算法