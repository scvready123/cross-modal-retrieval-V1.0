import numpy as np
def load_data(filename,filepath):
    x=np.load(filepath+filename,mmap_mode = 'r')
    return x

x = load_data('sim_matrix.npy','./')
print(x)
print(x.shape[0])
print(x.shape[1])

for i in range(len(x)):
    print('第',i,'张图片')
    print('对应的文本',i*5,'到',((i+1)*5-1))
    print(x[i])
