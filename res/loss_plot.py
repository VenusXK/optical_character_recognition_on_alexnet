from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib
matplotlib.rc('axes', grid = True, linewidth=3)
import numpy as np

train_step_arr = np.array([])
loss_arr = np.array([])
f=open(r'D:\Project\PythonProject\MachineLearningProject\ocr\words3755learnstep0_00001\loss.txt', encoding='gbk')
for line in f:
    line = line.split(' ')
    train_step_arr = np.append(train_step_arr, int(line[0]))
    loss_arr = np.append(loss_arr, float(line[1][:-2]))

# train_step_arr = train_step_arr[::-1]
# loss_arr = loss_arr[::-1]
print(loss_arr)
print(train_step_arr)
plt.suptitle('交叉熵损失函数数值曲线',fontsize = 20)
plt.plot(train_step_arr, loss_arr,'o:r')

# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
# plt.legend(loc="upper right")
plt.xlabel('训练次数/次',fontsize = 15)
plt.ylabel('Loss值（交叉熵损失函数数值）',fontsize = 15)

plt.show()