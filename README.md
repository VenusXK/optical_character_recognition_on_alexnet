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
    - [3. 伪代码 `Pseudocode`](#3-伪代码-pseudocode)
    - [4. 流程图 `flow chart`](#4-流程图-flow-chart)
  - [四、算法实现 `Algorithm implementation`](#四算法实现-algorithm-implementation)
    - [1. `AlexNet` 卷积神经网络结构实现 `AlexNet convolutional neural network structure implementation`](#1-alexnet-卷积神经网络结构实现-alexnet-convolutional-neural-network-structure-implementation)
    - [2. 运行模型时图像分割算法实现 `Image segmentation algorithm implementation`](#2-运行模型时图像分割算法实现-image-segmentation-algorithm-implementation)
  - [五、运行结果及分析 `Operation results and analysis`](#五运行结果及分析-operation-results-and-analysis)
    - [1. 性能评价指标 `Performance evaluation indicators`](#1-性能评价指标-performance-evaluation-indicators)
    - [2. 模型的训练结果和分析 `Training results and analysis of the model`](#2-模型的训练结果和分析-training-results-and-analysis-of-the-model)
    - [3. 对未知图像进行分割预测的运行结果和分析 `Run results and analysis of segmentation prediction on unknown images`](#3-对未知图像进行分割预测的运行结果和分析-run-results-and-analysis-of-segmentation-prediction-on-unknown-images)
    - [总结](#总结)

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

    **Figure 5** AlexNet convolutional neural network structure</div>

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

2. `SGD` 梯度下降法 `Gradient descent SGD`
   
   算法原理：对损失函数进行一阶泰勒展开的近似，对近似函数求最小值，把最小值当作下一步用来迭代的值。

    Algorithm principle: Approximate the first-order Taylor expansion of the loss function, find the minimum value of the approximate function, and take the minimum value as the value used for iteration in the next step.
    
3. `Adam` 优化算法  `optimization algorithm Adam`
   
   一种相对于 `SGD` 梯度下降更快的优化算法

   An optimization algorithm that is faster than SGD gradient descent.
 
### 3. 伪代码 `Pseudocode`
1. **`AlexNet` 网络结构伪代码 `AlexNet structure pseudocode`**
   ```py
    class Alexnet(nn.Module):
        def __init__(self, class_num = 3755):
            super(Alexnet, self).__init__()
            self.featureExtraction = nn.Sequential(
                nn.Conv2d(),nn.ReLU(),nn.MaxPool2d(),
                nn.Conv2d(),nn.ReLU(),nn.MaxPool2d(),
                nn.Conv2d(),nn.ReLU(),
                nn.Conv2d(),nn.ReLU(),
            nn.Conv2d(),nn.ReLU(),nn.MaxPool2d()
            )
            self.fc = nn.Sequential(
                nn.Dropout(),nn.Linear(),nn.ReLU(),
                nn.Dropout(),nn.Linear(),nn.ReLU(),
                nn.Linear())

        def forward(self, x):
            x = self.featureExtraction(x)
            x = x.view(x.size(0), 256*6*6)
            x = self.fc(x)
            return x

   ```
2. **模型训练伪代码（以 `Adam` 为优化器）`Model training pseudocode (with 'Adam' as optimizer)`**
    ```py
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            images, labels = data
            outputs = model(images)
            result_loss = loss(outputs,labels)
            opt_Adam.zero_grad()
            result_loss.backward()
            opt_Adam.step()
            train_step = train_step+1
            if(train_step%300==0):
                with torch.no_grad():
                    for test_data  in test_loader:
                        img_test,label_test = test_data
                        outputs_test = model(img_test)
            if(train_step%1000 == 0):
                torch.save()
    ```
### 4. 流程图 `flow chart`
<div align=center>
<img decoding="async" src="./Readme_File/AlexNet.drawio.png" width="80%">

**图6** AlexNet实现流程图

**Figure 6** AlexNet implementation flowchart</div>

<br>

## 四、算法实现 `Algorithm implementation`

### 1. `AlexNet` 卷积神经网络结构实现 `AlexNet convolutional neural network structure implementation`

1. 用 `torch.nn` 库建立 `AlexNet` 的 `python` 类，通过 `class Alexnet (nn.Module)` 语句将 `torch.nn` 下的 `Module` 传入类中。
   
   Use the torch.nn library to build AlexNet's python classes, and use class Alexnet (nn. Module) statement passes the Module under torch.nn into the class.

2. 通过 `nn.Sequential` 方法构建一个序列容器，用于搭建神经网络的模块，卷积神经网络的各网络层按照传入构造器的顺序添加到 `nn.Sequential()` 容器中。

    By 'nn.Sequential' method builds a sequence container for building the modules of the neural network, and the network layers of the convolutional neural network are added to the 'nn.Sequential' container.

3. 同理，用 `nn.Sequential` 构造全连接层 `
   
   Similarly, use 'nn. Sequential' constructs fully connected layers
   
4. 定义 `forward` 函数，在 `torch.nn` 模块中， `forward()` 函数执行实际的消息传递和计算，模型会自动执行网络的各层，并输出最后的特征
   
   Define the forward function, in the torch.nn module, the forward function performs the actual messaging and calculations, and the model automatically executes the layers of the network and outputs the final features

    ```py
    class Alexnet(nn.Module):
        def __init__(self, class_num = 3755):
            super(Alexnet, self).__init__()
            self.featureExtraction = nn.Sequential(
                nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size= 11, stride= 4, bias= 0),
                nn.ReLU(inplace= True),
                nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 0),
                nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5, stride= 1, padding = 2, bias= 0),
                nn.ReLU(inplace= True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size= 3, stride=1, padding= 1, bias=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size= 3, stride= 1, padding= 1, bias=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 0)
            )
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features= 256*6*6, out_features= 4096),
                nn.ReLU(inplace= True),
                nn.Dropout(0.5),
                nn.Linear(in_features= 4096, out_features= 4096),
                nn.ReLU(inplace= True),
                nn.Linear(in_features= 4096, out_features= class_num)
            )

        def forward(self, x):
            x = self.featureExtraction(x)
            x = x.view(x.size(0), 256*6*6)
            x = self.fc(x)
            return x
    ```
### 2. 运行模型时图像分割算法实现 `Image segmentation algorithm implementation`

1. 要运行模型实现文字识别，第一步要考虑的就是怎么将每一个字符从图片中切割下来，然后才可以送入我们设计好的模型进行字符识别，切割算法总结为以下两个步骤：
   
    To run the model to achieve text recognition, the first step to consider is how to cut each character from the picture, and then it can be sent to our designed model for character recognition, and the cutting algorithm is summarized into the following two steps:

   1. 对图片进行水平投影，找到每一行的上界限和下界限，进行行切割；

        Horizontal projection of the picture, find the upper and lower limits of each row, and cut the rows;

   2. 对切割出来的每一行，进行垂直投影，找到每一个字符的左右边界，进行单个字符的切割。

        For each line cut out, vertical projection is carried out, the left and right boundaries of each character are found, and a single character is cut.

2. 对于行切割，即水平投影，是对一张图片的每一行元素进行统计，然后根据统计结果画出统计结果图，进而确定每一行的起始点和结束点，统计像素图如下所示：

    For row cutting, that is, horizontal projection, is to count each row element of a picture, and then draw a statistical result map according to the statistical results, and then determine the start and end points of each row, the statistical pixel map is as follows:

<div align=center>
<img decoding="async" src="./Readme_File/split.png" width="80%">

**图7** 水平行切割示意图

**Figure 7** Schematic diagram of horizontal row cutting</div>

<br>

3. 切割完每一行后，我们得到了一行行文本，对这行文本进行垂直投影并根据垂直投影求出来每个字符的边界值进行单个字符切割，实现分割点效果如下：

    After cutting each line, we get a line of text, project this line of text vertically and cut a single character according to the boundary value of each character according to the vertical projection, and achieve the split point effect as follows:

    <div align=center>
    <img decoding="async" src="./Readme_File/split2.png" width="100%">

    **图8** 实现分割效果

    **Figure 8** Achieve split effect</div>

    <br>

## 五、运行结果及分析 `Operation results and analysis`

### 1. 性能评价指标 `Performance evaluation indicators`

1. loss值（交叉熵函数值）
   
   本次实验使用pytorch的交叉熵库函数，其用来判定实际的输出与期望的输出的接近程度。它主要刻画的是实际输出（概率）与期望输出（概率）的距离，交叉熵的值越小，概率分布越接近，交叉熵函数计算公式如下：

   This experiment uses PyTorch's cross-entropy library function, which is used to determine how close the actual output is to the desired output. It mainly describes the distance between the actual output (probability) and the expected output (probability), the smaller the value of cross-entropy, the closer the probability distribution, the cross-entropy function calculation formula is as follows:

   <div align=center>
    <img decoding="async" src="./Readme_File/loss.pytorch.png" width="100%">

    **图9** pytorch中交叉熵函数

    **Figure 9** Cross-entropy function in pytorch</div>

    <br>

    通过语句 `loss = torch.nn.CrossEntropyLoss()` 获取交叉熵对象，在模型训练阶段，每次训练 `32` 张图片，每 `300` 次训练记录一次 `loss` 值（交叉熵函数值），并使用 `matplotlib` 绘制曲线图。

    Obtain the cross-entropy object by statement loss = torch.nn.CrossEntropyLoss(), in the model training phase, 32 pictures per training, record the loss value (cross-entropy function value) every 300 training cycles, and plot the graph using matplotlib.

2. Accuracy值（预测集准确率值）
   
   本次实验训练阶段对预测集进行预测，根据预测结果与真实值比较，将预测准确的数量与总共计算的数量做比值，计算准确率，本次实验每次训练 `32` 张图片，每 `300` 次训练记录一次值 `Accuracy` 值（预测集准确率值），并使用 `matplotlib` 绘制曲线图。

   In the training stage of this experiment, the prediction set is predicted, and the prediction accuracy is compared with the real value according to the prediction results, the number of prediction accuracy is compared with the total calculated quantity, and the accuracy is calculated, this experiment trains 32 pictures each time, records the value Accuracy value (prediction set accuracy value) every 300 training times, and uses matplotlib to draw a curve.

### 2. 模型的训练结果和分析 `Training results and analysis of the model`

1. `loss` 值（交叉熵函数值） `loss value (cross-entropy function value)`

    <div align=center>
    <img decoding="async" src="./Readme_File/loss曲线.png" width="100%">

    **图10** 模型训练的Loss值（交叉熵损失函数值）曲线

    **Figure 10** The loss value (cross-entropy loss function value) curve trained by the model</div>

    <br>

    观察图像可以看到 `Loss` 值下降幅度有震荡，但整体呈下降姿态，经过 `50000` 次训练结果较好，`Loss` 值接近 `0` ，但仍未完全达到 `0` 值，震荡原因考虑可能为学习率设置为 `0.00001` 较高，之后降低学习率为 `0.000001` ， `Loss` 值仍保持小幅度上下振动，但总体趋势不发生变化。

    Observing the image, you can see that the loss value decreases with oscillations, but the overall decline posture, after 50,000 training results are better, the loss value is close to 0, but still not fully reached the 0 value, the reason for the shock may be that the learning rate is set to 0.00001 is higher, and then the learning rate is reduced to 0.000001, the loss value still maintains a small up and down vibration, but the overall trend does not change.

2. `Accuracy` 值（预测集准确率）`Accuracy value`
   
   <div align=center>
    <img decoding="async" src="./Readme_File/Accuracy曲线.png" width="100%">

    **图11** 模型训练的Accuracy值（预测准确率值）曲线

    **Figure 11** The Accuracy value curve of model training</div>

    <br>

    观察图像可以看到 `Accuracy` 值上升幅度出现振荡现象，但整体呈上升姿态，经过 `50000` 次训练结果较好， `Accurarcy` 值接近 `1` ，但仍未完全达到 `1` 值，震荡原因考虑可能为学习率设置为 `0.00001` 较高，之后降低学习率为 `0.000001` ， `Accuracy` 值仍保持小幅度上下振动，但总体趋势不发生变化。

    Observing the image, you can see that the Accuracy value rises in oscillation, but the overall upward attitude, after 50,000 training results are better, the Accurarcy value is close to 1, but it has not fully reached the 1 value, the reason for the oscillation may be that the learning rate is set to 0.00001 is higher, and then the learning rate is reduced to 0.000001, the Accuracy value still maintains a small amplitude up and down vibration, but the overall trend does not change.

### 3. 对未知图像进行分割预测的运行结果和分析 `Run results and analysis of segmentation prediction on unknown images`
1. 对于单个字体为48*48的文件，识别结果将“馨”字识别为“譬”字，其余均识别正确，如下图所示：
   
   For a single file with a font of 48*48, the recognition result recognizes the character "馨" as "譬", and the rest are recognized correctly, as shown in the following figure:

    <div align=center>
    <img decoding="async" src="./Readme_File/predict1.jpg" width="100%">

    **图12** 未知图像进行分割预测结果

    **Figure 12** Unknown images are segmented to predict result</div>

    <br>

2. 对于单个字体为 `48*49` 的文件，识别结果将“亶”字识别为“膏”字，将“枝”字识别为“栈”字，其余均识别正确，准确率较好。

    For a single file with a font of 48*49, the recognition results recognize the character "亶" as the character "膏" and the character "枝" as the character "栈", and the rest are recognized correctly and the accuracy rate is good.

    <div align=center>
    <img decoding="async" src="./Readme_File/predict2.jpg" width="100%">

    **图13** 未知图像进行分割预测结果

    **Figure 13** Unknown images are segmented to predict result</div>

    <br>

3. 对结果进行分析，由于AlexNet对读入的图片要求是 `227*227` ，本次训练数据集直接生成的 `227*227` 的图片，但是如上做未知图像的模型测试时输入的图片由于切割后几乎都比较小，经过放大后比较模糊，和训练的图片尺寸相差较大，可能会导致预测正确率不能达到训练时的预测正确率的问题，主要体现在相似字的识别上和复杂字的识别上。

    The results are analyzed, because AlexNet's requirement for the read picture is 227\*227, the 227\*227 picture directly generated by the training dataset is directly generated, but as shown above, the pictures input during the model test of unknown images are almost small after cutting, blurry after enlargement, and the size of the training picture is quite different, which may lead to the problem that the prediction accuracy rate cannot reach the prediction accuracy rate during training, which is mainly reflected in the recognition of similar words and the recognition of complex words.

### 总结
1. 经过本次实验后，我们对神经网络的使用有了初步的理解，对 `pytorch` 框架有了使用经验，受益良多。

    After this experiment, we have a preliminary understanding of the use of neural networks, and we have experience in using the pytorch framework, which has benefited a lot.

2. 在训练模型初期，由于数据集较大，约 `3664880` 张 `227*227` 的图片无论是图像生成还是模型训练，都需要消耗较大算力，因此取前 `30` 个字体进行模型训练，经过 `3` 小时左右训练后，模型 `loss` 值显著收敛。

    In the early stage of model training, due to the large data set, about 3664880 227*227 pictures need to consume large computing power whether it is image generation or model training, so the first 30 fonts are taken for model training, and after about 3 hours of training, the model loss value converges significantly.

    之后尝试用 `gpu` 加速对 `3755` 个字进行训练，下载了 `nvidia` 的 `cuda` 模块，并安装了 `cuda` 版本的 `pytorch` ，对 `3755` 张图片进行训练，由于图像较多，设置 `batchsize` 为 `512` ，但训练结果并不理想， `loss` 和 `accuracy` 保持几乎不变且表现均较差，之后查阅相关资料设置 `batchsize` 依次为 `128` 、 `64` 、 `32` ，并不断减小学习率，但效果依然较差。

    After that, try to use GPU acceleration to train 3755 words, download NVIDIA's CUDA module, and install the CUDA version of PyTorch, train 3755 pictures, due to more images, set the batchsize to 512, but the training results are not ideal, loss and accuracy remain almost unchanged and the performance is poor, and then consult the relevant information to set the batchsize to 128, 64, 32, and the learning rate is constantly reduced, but the effect is still poor.

    考虑到小样本上可以实现训练，故考虑样本过大的原因，计划将 `3755` 拆成多组进行训练，对 `1000` 字数据进行训练时发现，仍然不能收敛，故继续减小数据集到 `500` 、 `100` ，对 `100` 字训练时发现 `loss` 值和预测集的准确率值可以成功收敛，此时查阅资料发现有更好的优化器adam，而且经查阅资料发现梯度下降法 `sgd` 的表现明显不如 `adam` ， `adam` 优化方法梯度下降较快，故将优化器设置为 `adam` ，对 `100` 字进行训练，发现模型收敛速度相对于之前的训练快了很多，几乎是之前的数倍，欣喜之后将 `adam` 优化器用于 `1000` 字训练，发现模型的 `loss` 值仍收敛，故将 `adam` 优化器用于 `3755` 字的总体训练集，学习率设置为 `0.00001` 时发现可以收敛， `batchsize` 设置为 `32` 的情况下训练了 `50000` 次 `loss` 值降低到较低水平且总体保持不变， `accuracy` 值亦如此，之后将学习率进一步下调，但 `loss` 值和 `accuracy` 值仍保持一定整体水平变化不大，故结束训练。

    Considering that training can be achieved on a small sample, considering the reason why the sample is too large, it is planned to split the 3755 into multiple groups for training, and when training on 1000 words of data, it is found that it still cannot converge, so continue to reduce the data set to 500, 100, and the accuracy value of the loss value and prediction set can be successfully converged when training 100 words, at this time, the data was found to have a better optimizer adam, and after consulting the data, it was found that the performance of gradient descent SGD was significantly inferior to adam. Adam optimization method gradient decline is fast, so set the optimizer to adam, train 100 words, find that the model convergence speed is much faster than the previous training, almost several times the previous one, after rejoicing, the Adam optimizer is used for 1000 words training, and it is found that the loss value of the model is still convergent, so the ADAM optimizer is used for the overall training set of 3755 words. When the learning rate is set to 0.00001, it is found that it can converge, and when the batchsize is set to 32, the loss value is reduced to a low level and remains unchanged overall, and the accuracy value is also the same, and then the learning rate is further reduced, but the loss value and accuracy value still maintain a certain overall level and do not change much, so the training is ended.

3. 在进行图像预测时，意识到模型训练因为考虑 `AlexNet` 对图片的要求使用的图片较大为 `227*227` ，而待预测的图片由于分割普遍较小，故可能存在预测正确率和训练时的 `accuracy` 值有差别的现象。

    When performing image prediction, it is realized that the model training uses a larger image of 227*227 because considering AlexNet's requirements for images, and the image to be predicted is generally smaller due to the generally small segmentation, so there may be a difference between the prediction accuracy rate and the accuracy value during training.
   
4. 本次实验分工明确，图像生成采用的 `python` 自带的 `PIL` 库根据国标文字库进行图像绘制，由宋云龙同学负责；模型训练采用了 `pytorch` 框架实现了 `AlexNet` ，在 `GPU` 上进行模型训练并记录相关数值，由张景赫同学负责；训练完模型后，使用模型识别未知图片时需对图片分割，从而捕获单个字图片，由柳昂同学负责；对未知图像的具体预测工作由张景赫同学负责；文档撰写部分由张景赫和徐鹏馨负责。

    The division of labor in this experiment is clear, and the PIL library that comes with python for image generation is drawn according to the national standard text library, which is responsible for Song Yunlong; The model training adopts the pytorch framework to implement AlexNet, and the model is trained on the GPU and the relevant values are recorded, which is responsible for Zhang Jinghe. After training the model, when using the model to identify unknown pictures, it is necessary to segment the pictures to capture a single word picture, which is responsible for Liu Ang. The specific prediction of unknown images is the responsibility of Zhang Jinghe; The document writing part was handled by Zhang Jinghe and Xu Pengxin.
