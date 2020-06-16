import numpy as np



ary = np.load('../../data/computer_vision_data/train_x.npy',allow_pickle=True)
print(ary.shape)
ary = np.load('../../data/computer_vision_data/train_y.npy',allow_pickle=True)
print(ary.shape)
print(ary)
# np.savez('savez.npz',x=np1,y=np2)
# ary = np.load('savez.npz')
# print(ary)
# for i in ary:
#     print(ary[i])
