import numpy as np
import torch

pthfile = r'/home/lxx/xpe/XPE/sim_examples/data/best-600.pth'
net = torch.load(pthfile)
print(type(net))    # collections.OrderedDict'
print(len(net))     # 56

arr_0 = {}
Linear = {}
Conv2d = {}
li = 0
Ci = 0
# print(type(arr_0))    # dict


keys = net.iterkeys()
for key in keys:
    print(key)
    # print(type(key))                                          # unicode
    key = key.encode('unicode-escape').decode('string_escape')  # type from unicode to str
    print(type(key))
    if 'conv' in key:
        if 'weight' in key:
            val = net[key]
            # print(type(val))  # torch.nn.parameter.Parameter
            val_arr = val.detach().numpy()
            # print(type(val_arr))    # 'numpy.ndarray'
            print(val_arr.shape)
            Conv2d.update({'weight': val_arr})
        elif 'bias' in key:
            val = net[key]
            # print(type(val))  # torch.nn.parameter.Parameter
            val_arr = val.detach().numpy()
            # print(type(val_arr))    # 'numpy.ndarray'
            Conv2d.update({'bias':val_arr})
            arr_0.update({"Conv2d"+str(Ci): Conv2d})
            Ci = Ci + 1
            Conv2d = {}
    # elif 'bn' in key:
    #     if 'weight' in key:
    #         val = net[key]
    #         # print(type(val))  # torch.nn.parameter.Parameter
    #         val_arr = val.detach().numpy()
    #         # print(type(val_arr))    # 'numpy.ndarray'
    #         Linear.update({'weight': val_arr})
    #     elif 'bias' in key:
    #         val = net[key]
    #         # print(type(val))  # torch.nn.parameter.Parameter
    #         val_arr = val.detach().numpy()
    #         # print(type(val_arr))    # 'numpy.ndarray'
    #         Linear.update({'bias': val_arr})
    #         arr_0.update({"Linear" + str(li): Linear})
    #         li = li + 1
    #         Linear = {}

# print(type(arr_0))              # dict
# print(len(arr_0))               # 8
# print(len(arr_0['Conv2d1']))    # 2
# print(type(arr_0['Conv2d1']))   # dict
# print(type(arr_0['Linear0']['bias']))   # numpy.ndarray
#
# print(arr_0['Linear2']['bias'])
np.savez('star.npz', arr_0)

# test

Weight = np.load('star.npz')
print(Weight.files)                   # ['arr_0']

weight = np.load('star.npz')['arr_0'].item()
print(len(weight))                      # 8
print(type(weight))                     # dict
# print(weight["Linear0"])
# print(len(weight["Linear2"]))           # 2
print(type(weight["Conv2d4"]))          # dict
# print(type(weight["Linear0"]["bias"]))  # numpy.ndarray
# print(weight['Linear0']['bias'].shape)  # (4096,)
# print(weight['Linear0']['weight'].shape)    # (4096,9216)



# a = torch.ones(5)
# b = a.numpy()
# print(type(a))
# print(type(b))







# print(net.shape)
# net.save('./alexnet-owt-4df8aa71.npz')
# print("save npz")

Weight1 = np.load('star.npz')
print(Weight1.files)                       # ['arr_0']
weight1 = np.load('star.npz')['arr_0'].item()
# print(weight1['Linear0']['bias'].shape)     # (500,)
# print(weight1['Linear0']['weight'].shape)     # (784,500)

# weight = np.load('mnist-500-100.npz')['arr_0'].item()
# print(len(weight))                      # 6
# print(type(weight))                     # dict
# # print(weight["Linear0"])
# print(len(weight["Linear0"]))           # 2
# print(type(weight["Linear0"]))          # dict
# print(type(weight["Linear0"]["bias"]))  # numpy.ndarray
