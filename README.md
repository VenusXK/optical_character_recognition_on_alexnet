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

## 一、研究意义 `Research significance`
&emsp;&emsp;汉字作为中华民族文明发展的信息载体，已有数千年的历史，也是世界上使用人数最多的文字，它记录了璀璨的民族文化，展示了东方民族独特的思维和认知方法。随着计算机技术的推广应用，尤其是互联网的日益普及，人类越来越多地以计算机获得各种信息，大量的信息处理工作也都转移到计算机上进行。在日常生活和工作中，存在着大量的文字信息处理问题，因而将文字信息快速输入计算机的要求就变得非常迫切。现代社会的信息量空前丰富，其中绝大部分信息又是以印刷体的形式进行保存和传播的，这使得以键盘输入为主要手段的计算机输入设备变得相形见绌，输入速度低已经成为信息进入计算机系统的主要瓶颈，影响着整个系统的效率。

&emsp;&emsp;As the information carrier for the development of Chinese civilization, Chinese characters have a history of thousands of years and are also the most used script in the world, which records the brilliant national culture and shows the unique thinking and cognitive methods of Eastern peoples. With the promotion and application of computer technology, especially the increasing popularity of the Internet, human beings are increasingly using computers to obtain various information, and a large number of information processing work has also been transferred to computers. In daily life and work, there are a large number of text information processing problems, so the requirement to quickly input text information into the computer has become very urgent. The amount of information in modern society is unprecedentedly rich, and most of the information is preserved and disseminated in the form of printed form, which makes the computer input device with keyboard input as the main means dwarfed, and the low input speed has become the main bottleneck of information entering the computer system, affecting the efficiency of the entire system.

&emsp;&emsp;一方面，使用该技术可以提高计算机的使用效率，克服人与机器的矛盾。另一方面，该技术可以应用于快速识别身份证、银行卡、驾驶证等卡证类信息，将证件文字信息直接转换为可编辑文本，可以大大提高山东省相关部门工作效率，减少人力劳动成本，可以实时进行相关人员的身份核验，以便山东省各部门安全管理。

&emsp;&emsp;On the one hand, the use of this technology can improve the efficiency of computer use and overcome the contradiction between man and machine. On the other hand, this technology can be applied to quickly identify ID cards, bank cards, driver's licenses and other card information, and directly convert the text information of the certificate into editable text, which can greatly improve the work efficiency of relevant departments in Shandong Province, reduce labor costs, and verify the identity of relevant personnel in real time for the safety management of various departments in Shandong Province.

## 二、数据描述 `Dataset description`

### 1. 数据获取途径 `Access to dataset`
&emsp;&emsp;利用 `Windows` 自带的字体文件库，用 `Python` 的 `PIL` 库绘图，每张图片上绘制一个文字，总共绘制 `3755` 个汉字，

&emsp;&emsp;Using the font file library that comes with WINDOWS, drawing with Python's PIL library, drawing a text on each picture, drawing a total of 3755 Chinese characters.

### 2. 使用数据内容组成 `Composition of dataset`

&emsp;&emsp;根据汉字国标库绘图，同一字体生成 `976` 张图片作为训练集，生成 `244` 张图片作为训练中准确率的测试集，总共 `3755` 个汉字，训练集（不包含验证集）共 `3664880` 个文件，每个汉字有对应 `876` 张训练集图片和 `244` 张验证集图片，根据 `AlexNet` 要求每张图片大小应为 `227*227` 。

&emsp;&emsp;According to the Chinese character national standard library drawing, the same font generates 976 pictures as a training set, generates `244` pictures as a test set of accuracy in training, a total of 3755 Chinese characters, a training set (excluding verification set) a total of 3664880 files, each Chinese character has a corresponding 876 training set pictures and 244 verification set pictures, according to AlexNet requirements each picture size should be 227*227.

### 3. 数据生成过程 `Dataset generation process`
 1. 首先定义输入参数，其中包括输出目录、字体目录、测试集大小、图像尺寸、图像旋转幅度等等。

    Start by defining the input parameters, which include the output directory, font directory, test set size, image size, image rotation, and so on.

1. 接下来将得到的汉字与序号对应表读入内存，表示汉字序号到汉字的映射，用于后面的字体生成。

    Next, the table corresponding to the obtained Chinese characters and ordinal numbers is read into memory, indicating the mapping of IDs to Chinese characters, which is used for later font generation.

3. 我们对图像进行一定角度的旋转，将旋转角度存储到列表中，旋转角度的范围是 `[-rotate,rotate]` 。

    We rotate the image at an angle and store the rotation angle in a list with the range of rotation angles [-rotate, rotate].

4. 字体图像的生成使用的工具是 `Python` 自带的 `PIL` 库。该库里有图片生成函数，用该函数结合字体文件，可以生成我们想要的图片化的汉字。设定好生成的字体颜色为黑底白色，字体尺寸由输入参数来动态设定。

    The tool used to generate font images is Python's built-in PIL library. There is an image generation function in this library, and with this function combined with font files, we can generate the Chinese characters we want to be picturesque. Set the generated font color to white on black, and the font size is dynamically set by input parameters.

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
1.	AlexNet对图像大小要求为227*227，生成图像时已设置图像大小为227*227；
2.	使用pytorch的ImageFolder库进行图像文件的选择，ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名；
3.	由于使用pytorch框架，需要将图像转换为Tensor（张量）数据结构，利用torchvsion 下面的transforms库将输入图片转换为Tensor格式，语句如下：
transforms = transforms.Compose([transforms.ToTensor()])
data_test = ImageFolder(path_test, transform=transforms)
4.	使用pytorch的DataLoader库进行训练集的导入，此库导入训练集为成批导入，每一批既导入图像又导入标签，训练时根据每一批图片进行训练，通过设置batch_size可以控制每一批导入数据的量，batch_size越小训练的轮次越多，本次实验设置batch_size为32，通过DataLoader导入训练集大致情况如下图所示：
