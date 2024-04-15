import numpy as np

if __name__ == "__main__":
    a = np.arange(0, 4811)
    # 把a划分成7：3的比例，要求不能重复
    # 1. 生成一个随机序列
    np.random.shuffle(a)
    # 2. 划分
    a1 = a[:int(0.7*len(a))]    
    a2 = a[int(0.7*len(a)):]
    # 保存txt文件
    np.savetxt('train.txt', a1, fmt='%s')
    np.savetxt('val.txt', a2, fmt='%s')