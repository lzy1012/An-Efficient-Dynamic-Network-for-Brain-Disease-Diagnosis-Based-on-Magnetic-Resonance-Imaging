import os
import numpy as np
import  matplotlib.pyplot as plt

path = 'output/GF-mobilenetv3_large_100_patch-size-96_T4_train-stage3'   # 指定第几阶段模型保存路径

data = []
with open(os.path.join(path, 'indexes.txt'), 'r') as f:      # 读loss记录文件
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        line = [eval(v) for v in line]
        data.append(line)
data = np.array(data)        #第一列为训练precision，第二列为训练recall，第三列为测试集precision，第四列为测试集recall

# 画precision曲线
plt.figure()
plt.plot(np.arange(data.shape[0]), data[:, 0])
plt.plot(np.arange(data.shape[0]), data[:, 2])
plt.title('precision vs. epoches')
plt.xlabel('epoches')
plt.ylabel('precision')
plt.legend(labels=['Train precision', 'val precision'])
plt.savefig(os.path.join(path, "precision.jpg"))   # 曲线保存在path路径下

# 画recall曲线
plt.figure()
plt.plot(np.arange(data.shape[0]), data[:, 1])
plt.plot(np.arange(data.shape[0]), data[:, 3])
plt.title('recall vs. epoches')
plt.xlabel('epoches')
plt.ylabel('recall')
plt.legend(labels=['Train recall', 'val recall'])
plt.savefig(os.path.join(path, "recall.jpg"))
