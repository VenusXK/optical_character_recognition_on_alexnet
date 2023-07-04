<!-- <div align=center>
<img decoding="async" src="./Readme_File/Poster_matplotlib.png" width="40%" >
</div> -->

# my optical character recognition

>### Brief introduction
>This is my assignment of undergraduate education in China University of Petroleum (East China)
>### About the contents of the 'dataset' folder usage
> Because the dataset is too large, a total of about 40G, it is not convenient to upload, here the dataset generation code is uploaded to run the dataset/gen_printed_char.py file to generate the training set image, the specific operation is below:
>
>```shell
>python gen_printed_char.py --out_dir [out_put_dir] --font_dir [windows_font_dir] --width [img_width] --height [img_height] --margin 4 --rotate 30 --rotate_step 1
>``` 
>### About the contents of the 'train' folder usage
>The train directory is the training directory, After executing the train_adam.py, you can train the model, 'train_adam.py' is used for training with the 'adam' optimizer, if training uses this file 
>### About the contents of the 'test' folder usage
>The 'test' directory is the directory of test images, Run /run.py to convert images to text
> 
>'run/run.py' specifies before running:
>
>- A directory of images to be recognized by OCR
>
>- The directory after the image is segmented
> 
>- Catalog of trained models
>
>- The file corresponding to the kanji and 'id' ordinal number
>
>You can then see the recognized text in the console
>### About the contents of the 'res' folder usage
>The res directory is the model training result directory, which contains the raw data of Loss image generation code, Accuracy image generation code, Loss image, Accuracy image, Loss, and Accuracy

### Directory
- [my optical character recognition](#my-optical-character-recognition)
    - [Directory](#directory)
  - [Research significance](#research-significance)
  - [Dataset description](#dataset-description)
    - [1. Access to dataset](#1-access-to-dataset)
    - [2. Composition of dataset](#2-composition-of-dataset)
    - [3. Dataset generation process](#3-dataset-generation-process)
    - [4. Results of data processing](#4-results-of-data-processing)
  - [三、Model description](#三model-description)
    - [1. Data processing processes](#1-data-processing-processes)
    - [2. The algorithm used](#2-the-algorithm-used)
    - [3. Pseudocode](#3-pseudocode)
    - [4. flow chart](#4-flow-chart)
  - [Algorithm implementation](#algorithm-implementation)
    - [1. AlexNet convolutional neural network structure implementation](#1-alexnet-convolutional-neural-network-structure-implementation)
    - [2. Image segmentation algorithm implementation](#2-image-segmentation-algorithm-implementation)
  - [Operation results and analysis](#operation-results-and-analysis)
    - [1. Performance evaluation indicators](#1-performance-evaluation-indicators)
    - [2. Training results and analysis of the model](#2-training-results-and-analysis-of-the-model)
    - [3. Run results and analysis of segmentation prediction on unknown images](#3-run-results-and-analysis-of-segmentation-prediction-on-unknown-images)
    - [summary](#summary)

## Research significance

As the information carrier for the development of Chinese civilization, Chinese characters have a history of thousands of years and are also the most used script in the world, which records the brilliant national culture and shows the unique thinking and cognitive methods of Eastern peoples. With the promotion and application of computer technology, especially the increasing popularity of the Internet, human beings are increasingly using computers to obtain various information, and a large number of information processing work has also been transferred to computers. In daily life and work, there are a large number of text information processing problems, so the requirement to quickly input text information into the computer has become very urgent. The amount of information in modern society is unprecedentedly rich, and most of the information is preserved and disseminated in the form of printed form, which makes the computer input device with keyboard input as the main means dwarfed, and the low input speed has become the main bottleneck of information entering the computer system, affecting the efficiency of the entire system.

On the one hand, the use of this technology can improve the efficiency of computer use and overcome the contradiction between man and machine. On the other hand, this technology can be applied to quickly identify ID cards, bank cards, driver`s licenses and other card information, and directly convert the text information of the certificate into editable text, which can greatly improve the work efficiency of relevant departments in Shandong Province, reduce labor costs, and verify the identity of relevant personnel in real time for the safety management of various departments in Shandong Province.

## Dataset description


### 1. Access to dataset

Using the font file library that comes with WINDOWS, drawing with Python`s PIL library, drawing a text on each picture, drawing a total of 3755 Chinese characters.

### 2. Composition of dataset

According to the Chinese character national standard library drawing, the same font generates 976 pictures as a training set, generates `244` pictures as a test set of accuracy in training, a total of 3755 Chinese characters, a training set (excluding verification set) a total of 3664880 files, each Chinese character has a corresponding 876 training set pictures and 244 verification set pictures, according to AlexNet requirements each picture size should be 227*227.

### 3. Dataset generation process

1. Start by defining the input parameters, which include the output directory, font directory, test set size, image size, image rotation, and so on.
2. Next, the table corresponding to the obtained Chinese characters and ordinal numbers is read into memory, indicating the mapping of IDs to Chinese characters, which is used for later font generation.
3. We rotate the image at an angle and store the rotation angle in a list with the range of rotation angles [-rotate, rotate].
4. The tool used to generate font images is Python`s built-in PIL library. There is an image generation function in this library, and with this function combined with font files, we can generate the Chinese characters we want to be picturesque. Set the generated font color to white on black, and the font size is dynamically set by input parameters.
5. At the same time, we do appropriate bloat and corrosion of the generated fonts to expand the number of training sets.
6. Execute the following command to start generating a set of Chinese characters in printed characters.

    ```shell
    python gen_printed_char.py --out_dir [out_put_dir] --font_dir [windows_font_dir] --width [img_width] --height [img_height] --margin 4 --rotate 30 --rotate_step 1
    ```
    
7. If an image of 227*227 size is generated, the total generation time under 2060 is nearly 16 hours, and the training set has a total of 3664880 files.

### 4. Results of data processing

<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果1.png" width="80%" >


**Figure 1** The training set (excluding the validation set) contains 3755 characters and 3664880 files

</div>



<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果1.png" width="80%">


**Figure 2** A total of 876 pictures of each Chinese character in the training set</div>



<div align=center>
<img decoding="async" src="./Readme_File/数据处理后的结果2.png" width="80%" >


**Figure 3** A total of 244 images for each Chinese character in the verification set</div>

## 三、Model description
### 1. Data processing processes
1.	AlexNet has an image size requirement of 227\*227, and the image size has been set to 227\*227 when generating images;
2.	Use pytorch`s ImageFolder library for image file selection, ImageFolder assumes that all files are saved in folders, and the same category of pictures is stored under each folder, and the folder name is class name;
3.	Since the `pytorch` framework is used, the image needs to be converted to a `tensor` data structure, and the `transforms` library under `torchvsion` is used to convert the input image to `tensor` format with the following statement:

    ```py
    transforms = transforms.Compose([transforms.ToTensor()])
    data_test = ImageFolder(path_test, transform=transforms)
    ```
4.	Use pytorch`s DataLoader library for the import of the training set, this library imports the training set as batch import, each batch both imports images and labels, training according to each batch of pictures for training, through the setting batch_size you can control the amount of data imported into each batch, the smaller the batch_size, the more rounds of training, the batch_size of this experiment is set to 32, The following figure shows the general situation of importing the training set through DataLoader:

<div align=center>
<img decoding="async" src="./Readme_File/DataLoader.png" width="80%">


**Figure 4** An example of an image of a batchsize input</div>


1. This training uses GPU acceleration due to the large data set, configures the PyTorch environment by installing NVIDIA`s CUDA module and downloading the CUDA 11.6 version of PyTorch, determines whether CUDA is available through the torch.cuda.is_available, and sets the GPU hardware through the device = Torch.Device ("CUDA") statement on the basis of availability. For each batch of training set data imported through DataLoader, import the training set data into the GPU through images = images.to (device) and labels = labels.to (device) statements.

### 2. The algorithm used
1. AlexNet
    <div align=center>
    <img decoding="async" src="./Readme_File/AlexNet.png" width="80%">


    **Figure 5** AlexNet convolutional neural network structure</div>

    

    1. The first convolutional layer of the AlexNet`**
    
        The input image size is: 224\*224\*3, for the convenience of subsequent processing, it is generally changed to 227\*227\*3, the first convolutional layer is: 11\*11\*3, the convolution kernel size is 11\*11, there are 96 convolution kernels, the convolution step size is 4, the ReLU activation function is used after the convolutional layer, the output size is (227-11)/4+1=55, and the size of each feature map output is 55\*55\*96;

        The size of the pooled kernel of the maximum pooling layer is 3\*3, the step size is 2, and the output size is (55-3)/2+1=27, so the size of the feature map is: 27\*27\*96.

    2. The second convolutional layer of the AlexNet
        
        The input data is 27\*27\*96, and the data is convolved by 256 convolution kernels of size 5\*5\*96, with a step of 1 and a fill value of 2, which is processed using the ReLU layer after convolution;
    
        The maximum pooling layer has a kernel size of 3\*3 and a step size of 2;

    3. The third to the fifth convolutional layer of the AlexNet`
    
        Each set of data in the third layer is convolutioned by a convolution kernel with a size of 3\*3\*384, with a step size of 1 and a fill value of 1, and the ReLU layer is used for processing after convolution.
    
        Each set of data in the fourth layer is processed by a convolution kernel convolution with a size of 3\*3\*384, with a step of 1 and a fill value of 1, and the ReLU layer is used for processing after convolution;
    
        Each set of data in the fifth layer is convolutioned by a convolution kernel with a size of 3\*3\*256, with a step size of 1 and a fill value of 1, and then processed using the ReLU layer after convolution;
    
        After the 3\*3 pooling window, the step size is 2, and the pixel layer is output after pooling;


    4. layer 6 to layer 8 full connectivity of the AlexNet
    
        The sixth layer first discards the data with a probability of 0.5, after a total of 4096 neuron processing, and then through the ReLU layer;
    
        The seventh and sixth layers first discard the data with a probability of 0.5, input 4096 eigenvalues, output 4096 eigenvalues, and then process it through the ReLU layer;
    
        The eighth layer inputs 4096 eigenvalues and outputs 3755 eigenvalues.

2. Gradient descent SGD
   
    Algorithm principle: Approximate the first-order Taylor expansion of the loss function, find the minimum value of the approximate function, and take the minimum value as the value used for iteration in the next step.
    
3. optimization algorithm Adam
   
   An optimization algorithm that is faster than SGD gradient descent.
 
### 3. Pseudocode
1. AlexNet structure pseudocode
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
2. Model training pseudocode (with 'Adam' as optimizer)
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
### 4. flow chart
<div align=center>
<img decoding="async" src="./Readme_File/AlexNet.drawio.png" width="80%">


**Figure 6** AlexNet implementation flowchart</div>



## Algorithm implementation

### 1. AlexNet convolutional neural network structure implementation

1. Use the torch.nn library to build AlexNet's python classes, and use class Alexnet (nn. Module) statement passes the Module under torch.nn into the class.
2. By 'nn.Sequential' method builds a sequence container for building the modules of the neural network, and the network layers of the convolutional neural network are added to the 'nn.Sequential' container.

3. Similarly, use 'nn. Sequential' constructs fully connected layers
   
4. Define the forward function, in the torch.nn module, the forward function performs the actual messaging and calculations, and the model automatically executes the layers of the network and outputs the final features

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
### 2. Image segmentation algorithm implementation

To run the model to achieve text recognition, the first step to consider is how to cut each character from the picture, and then it can be sent to our designed model for character recognition, and the cutting algorithm is summarized into the following two steps:

   1. Horizontal projection of the picture, find the upper and lower limits of each row, and cut the rows;

   2. For each line cut out, vertical projection is carried out, the left and right boundaries of each character are found, and a single character is cut.

   3. For row cutting, that is, horizontal projection, is to count each row element of a picture, and then draw a statistical result map according to the statistical results, and then determine the start and end points of each row, the statistical pixel map is as follows:

<div align=center>
<img decoding="async" src="./Readme_File/split.png" width="80%">


**Figure 7** Schematic diagram of horizontal row cutting</div>

1. After cutting each line, we get a line of text, project this line of text vertically and cut a single character according to the boundary value of each character according to the vertical projection, and achieve the split point effect as follows:

    <div align=center>
    <img decoding="async" src="./Readme_File/split2.png" width="100%">

    **Figure 8** Achieve split effect</div>

    

## Operation results and analysis

### 1. Performance evaluation indicators

1. loss
   
   This experiment uses PyTorch's cross-entropy library function, which is used to determine how close the actual output is to the desired output. It mainly describes the distance between the actual output (probability) and the expected output (probability), the smaller the value of cross-entropy, the closer the probability distribution, the cross-entropy function calculation formula is as follows:

   <div align=center>
    <img decoding="async" src="./Readme_File/loss.pytorch.png" width="100%">

    **Figure 9** Cross-entropy function in pytorch</div>

    Obtain the cross-entropy object by statement loss = torch.nn.CrossEntropyLoss(), in the model training phase, 32 pictures per training, record the loss value (cross-entropy function value) every 300 training cycles, and plot the graph using matplotlib.

2. Accuracy

   In the training stage of this experiment, the prediction set is predicted, and the prediction accuracy is compared with the real value according to the prediction results, the number of prediction accuracy is compared with the total calculated quantity, and the accuracy is calculated, this experiment trains 32 pictures each time, records the value Accuracy value (prediction set accuracy value) every 300 training times, and uses matplotlib to draw a curve.

### 2. Training results and analysis of the model

1. loss value (cross-entropy function value)

    <div align=center>
    <img decoding="async" src="./Readme_File/Loss曲线.png" width="100%">


    **Figure 10** The loss value (cross-entropy loss function value) curve trained by the model</div>

    
    Observing the image, you can see that the loss value decreases with oscillations, but the overall decline posture, after 50,000 training results are better, the loss value is close to 0, but still not fully reached the 0 value, the reason for the shock may be that the learning rate is set to 0.00001 is higher, and then the learning rate is reduced to 0.000001, the loss value still maintains a small up and down vibration, but the overall trend does not change.

2. Accuracy value
   
   <div align=center>
    <img decoding="async" src="./Readme_File/Accuracy曲线.png" width="100%">

    **Figure 11** The Accuracy value curve of model training</div>

    
    Observing the image, you can see that the Accuracy value rises in oscillation, but the overall upward attitude, after 50,000 training results are better, the Accurarcy value is close to 1, but it has not fully reached the 1 value, the reason for the oscillation may be that the learning rate is set to 0.00001 is higher, and then the learning rate is reduced to 0.000001, the Accuracy value still maintains a small amplitude up and down vibration, but the overall trend does not change.

### 3. Run results and analysis of segmentation prediction on unknown images
1. For a single file with a font of 48*48, the recognition result recognizes the character "馨" as "譬", and the rest are recognized correctly, as shown in the following figure:

    <div align=center>
    <img decoding="async" src="./Readme_File/predict1.jpg" width="100%">

    **Figure 12** Unknown images are segmented to predict result</div>


2. For a single file with a font of 48*49, the recognition results recognize the character "亶" as the character "膏" and the character "枝" as the character "栈", and the rest are recognized correctly and the accuracy rate is good.

    <div align=center>
    <img decoding="async" src="./Readme_File/predict2.jpg" width="100%">

    **Figure 13** Unknown images are segmented to predict result</div>

    

3. The results are analyzed, because AlexNet's requirement for the read picture is 227\*227, the 227\*227 picture directly generated by the training dataset is directly generated, but as shown above, the pictures input during the model test of unknown images are almost small after cutting, blurry after enlargement, and the size of the training picture is quite different, which may lead to the problem that the prediction accuracy rate cannot reach the prediction accuracy rate during training, which is mainly reflected in the recognition of similar words and the recognition of complex words.

### summary
1. After this experiment, we have a preliminary understanding of the use of neural networks, and we have experience in using the pytorch framework, which has benefited a lot.

2. In the early stage of model training, due to the large data set, about 3664880 227*227 pictures need to consume large computing power whether it is image generation or model training, so the first 30 fonts are taken for model training, and after about 3 hours of training, the model loss value converges significantly.

    After that, try to use GPU acceleration to train 3755 words, download NVIDIA's CUDA module, and install the CUDA version of PyTorch, train 3755 pictures, due to more images, set the batchsize to 512, but the training results are not ideal, loss and accuracy remain almost unchanged and the performance is poor, and then consult the relevant information to set the batchsize to 128, 64, 32, and the learning rate is constantly reduced, but the effect is still poor.

    Considering that training can be achieved on a small sample, considering the reason why the sample is too large, it is planned to split the 3755 into multiple groups for training, and when training on 1000 words of data, it is found that it still cannot converge, so continue to reduce the data set to 500, 100, and the accuracy value of the loss value and prediction set can be successfully converged when training 100 words, at this time, the data was found to have a better optimizer adam, and after consulting the data, it was found that the performance of gradient descent SGD was significantly inferior to adam. Adam optimization method gradient decline is fast, so set the optimizer to adam, train 100 words, find that the model convergence speed is much faster than the previous training, almost several times the previous one, after rejoicing, the Adam optimizer is used for 1000 words training, and it is found that the loss value of the model is still convergent, so the ADAM optimizer is used for the overall training set of 3755 words. When the learning rate is set to 0.00001, it is found that it can converge, and when the batchsize is set to 32, the loss value is reduced to a low level and remains unchanged overall, and the accuracy value is also the same, and then the learning rate is further reduced, but the loss value and accuracy value still maintain a certain overall level and do not change much, so the training is ended.

3. When performing image prediction, it is realized that the model training uses a larger image of 227*227 because considering AlexNet's requirements for images, and the image to be predicted is generally smaller due to the generally small segmentation, so there may be a difference between the prediction accuracy rate and the accuracy value during training.
   
4. The division of labor in this experiment is clear, and the PIL library that comes with python for image generation is drawn according to the national standard text library, which is responsible for Song Yunlong; The model training adopts the pytorch framework to implement AlexNet, and the model is trained on the GPU and the relevant values are recorded, which is responsible for Zhang Jinghe. After training the model, when using the model to identify unknown pictures, it is necessary to segment the pictures to capture a single word picture, which is responsible for Liu Ang. The specific prediction of unknown images is the responsibility of Zhang Jinghe; The document writing part was handled by Zhang Jinghe and Xu Pengxin.
