import cv2
import numpy as np
import numpy
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import pickle
import os

#import torchvision

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

def split(a):  # 获取各行起点和终点
    # b是a的非0元素的下标 组成的数组 (np格式),同时也是高度的值
    b = np.transpose(np.nonzero(a))
    # print(b,type(b))
    # print(a,b.tolist())

    star = []
    end = []
    star.append(int(b[0]))
    for i in range(len(b) - 1):
        cha_dic = int(b[i + 1]) - int(b[i])
        if cha_dic > 1:
            # print(cha_dic,int(b[i]),int(b[i+1]))
            end.append(int(b[i]))
            star.append(int(b[i + 1]))
    end.append(int(b[len(b) - 1]))
    # print(star) # [13, 50, 87, 124, 161]
    # print(end)  # [36, 73, 110, 147,184]
    return star, end


def get_horizontal_shadow(img, img_bi):  # 水平投影+分割
    # 1.水平投影
    h, w = img_bi.shape
    shadow_h = img_bi.copy()  # shadow_h画图用(切记！copy后面还有个())

    a = [0 for z in range(0, h)]  # 初始化一个长度为h的数组，用于记录每一行的黑点个数

    for j in range(0, h):  # 遍历一行
        for i in range(0, w):  # 遍历一列
            if shadow_h[j, i] == 0:  # 发现黑色
                a[j] += 1  # a数组这一行的值+1
                shadow_h[j, i] = 255  # 记录好了就变为白色

    for j in range(0, h):  # 遍历一行 画黑条,长度为a[j]
        for i in range(0, a[j]):
            shadow_h[j, i] = 0

    return a


def get_vertical_shadow(img, img_bi):  # 垂直投影+分割
    # 1.垂直投影
    h, w = img_bi.shape
    shadow_v = img_bi.copy()
    a = [0 for z in range(0, w)]
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # print('h = ', h)
    # print('w = ', w)
    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if shadow_v[i, j] == 0:  # 如果该点为黑点(默认白底黑字)
                a[j] += 1  # 该列的计数器加一计数
                shadow_v[i, j] = 255  # 记录完后将其变为白色
                # print (j)
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            shadow_v[i, j] = 0  # 涂黑

    return a


def character_cut(img, img_bi):  # 字符切割
    h, w = img_bi.shape
    # 1.水平投影
    ha = get_horizontal_shadow(img, img_bi)  # 水平投影
    # 2.开始分割
    # step2.1: 获取各行起点和终点
    h_star, h_end = split(ha)
    # step2.2: 切割行[y:y+h, x:x+w]
    for i in range(len(h_star)):  # 就是几行 5  [0 1 2 3 4]
        hs, he = h_star[i], h_end[i]
        img_line = img[hs:he, 0:img.shape[1]]

        # step2.3: 垂直投影
        img_line_gray = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)
        thresh1, img_line_bi = cv2.threshold(img_line_gray, 130, 255, cv2.THRESH_BINARY)
        # cv2.imshow('img_line',img_line)
        # cv2.imshow('img_line_gray',img_line_gray)
        # cv2.imshow('img_line_bi',img_line_bi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        va = get_vertical_shadow(img_line, img_line_bi)

        # step2.4: 获取各列起点和终点
        v_star, v_end = split(va)
        # step2.5: 切割字符[y:y+h, x:x+w]
        # print(v_star)
        # print(v_end)
        v_star_array = numpy.array(v_star)
        v_end_array = numpy.array(v_end)
        c_array=v_end_array-v_star_array
        n = len(v_star)
        # print(n)
        # print('原始每个字长')
        # print(c_array)

        ma=max(c_array)
        u=0
        # dd=[]
        # d_array =numpy.array(dd,dtype='int32')
        d_array=numpy.empty(n,dtype='int32')
        e_array = numpy.empty(n, dtype='int32')
        f_array = numpy.empty(n, dtype='int32')
        g_array = numpy.empty(n, dtype='int32')
        h_array = numpy.empty(n, dtype='int32')
        for p in range(n-1):
            if c_array[p]<=ma-10 and c_array[p]>5 :
                if c_array[p]+c_array[p+1] >=ma-10 and c_array[p]+c_array[p+1]<=ma :
                    e_array[u]=v_star_array[p]
                    f_array[u]=v_end_array[p+1]
                    d_array[u]=c_array[p]+c_array[p+1]
                    u=u+1
                    p=p+1
            else:
                e_array[u]=v_star_array[p]
                f_array[u]=v_end_array[p]
                d_array[u]=c_array[p]
                u=u+1
        if(c_array[n-1]>=ma-10 or c_array[n-1]<=5):
            e_array[u]=v_star_array[n-1]
            f_array[u]=v_end_array[n-1]
            d_array[u]=c_array[n-1]
        e_array=e_array[:u+1]
        f_array=f_array[:u+1]
        # print(u)
        v=1
        for q in range(1,u+1):
            if e_array[q]>f_array[q-1]:
                g_array[v]=e_array[q]
                h_array[v]=f_array[q]
                v=v+1
        g_array[0]=e_array[0]
        h_array[0]=f_array[0]
        g_array = g_array[:v]
        h_array = h_array[:v ]
        # print('处理后字符开始的位置')
        # print(e_array)
        # print(g_array)
        # print('处理后字符结束的位置')
        # print(f_array)
        # print(h_array)
        # print('处理后字长')
        # print(d_array)

        for j in range(len(g_array)):  # 几列
            if h_array[j]<=w and h_array[j]>0:
                # print(g_array[j], h_array[j])
                vs, ve = g_array[j], h_array[j]
                img_char = img_line[0:img_line.shape[0], vs:ve]  # [0:h, vs:ve]
                thresh, img_char = cv2.threshold(img_char, 130, 255, cv2.THRESH_BINARY_INV)
                # step2.6: 保存字符
                # save_name = 'char_' + str(i) + '_' + str(j) + '.jpg'
                cv2.imwrite(r'D:/Project/PythonProject/MachineLearningProject/ocr/test/yanshi/jieguo/char_' + str(i) + '_' + str(j) + '.jpg', img_char)
            else :
                break

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def get_label_dict():
    f = open('D:/Project/PythonProject/MachineLearningProject/ocr/test/yanshi/chinese_labels', 'r')
    label_dict = pickle.load(StrToBytes(f))
    f.close()
    return label_dict

def getfiles(pic_path):
    filenames=os.listdir(pic_path)
    return filenames


if __name__ == "__main__":
    img = cv2.imread(r'D:/Project/PythonProject/MachineLearningProject/ocr/test/yanshi/yanshi.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bi = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
    character_cut(img, img_bi)  # 输入图片 和 二值图, 即可进行字符分割
    
    label_dic = get_label_dict()
    
    model = Alexnet()
    state_dict = torch.load(r'D:/Project/PythonProject/MachineLearningProject/ocr/test/model_final_48000.pth', map_location='cpu')
    model.load_state_dict(state_dict,strict=False)
    
    pic_path = 'D:/Project/PythonProject/MachineLearningProject/ocr/test/yanshi/jieguo'

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor()  
    ])

    filenames = getfiles(pic_path)
    # filenames.sort(key=lambda x: int(x.split(".")[0].split("_")[1])) 
    flag = filenames[0][5]
    for filename in filenames:
        if(flag!=filename[5]):
            print('')
            flag=filename[5]
        img = cv2.imread(pic_path+'/'+filename)
        # print(pic_path+'/'+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (227,227),interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0,0,0))
        img_ = transform_valid(img).unsqueeze(0)
        
        # img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)#/255
        # print(img_.shape)
        outputs = model(img_)
        _, indices = torch.max(outputs,1)
        # print(filename)
        print(label_dic[int(indices[0])], end='')
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
