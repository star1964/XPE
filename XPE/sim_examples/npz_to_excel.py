import numpy as np
import pandas as pd

weights_dir = "./data/mnist-500-100.npz"
# weights_dir = "./data/mnist-lenet.npz"
Weight = np.load(weights_dir)['arr_0'].item()

Net = [
    ['Linear'],
    ['Linear'],
    ['Linear']
]

# Net = [
#     ['Conv2d',],
#     ['Conv2d',],
#     ['Linear',],
#     ['Linear',],
#     ['Linear',],
#     ]

arr_0 = {}
arr_1 = {}
Linear_1 = {}
Conv2d_1 = {}

Linear = []
numLinear = 0
Conv = []
numConv = 0
for i in Net:
    if i[0] == "NoiseConv2d" or i[0] == "Conv2d":
        weight = Weight[i[0] + str(numConv)]["kernel"]
        bias = Weight[i[0] + str(numConv)]["bias"]
        Conv.append((weight, bias))
        numConv += 1
    if i[0] == "NoiseLinear" or i[0] == "Linear":
        weight = Weight[i[0] + str(numLinear)]["weight"]
        bias = Weight[i[0] + str(numLinear)]["bias"]
        Linear.append((weight, bias))
        numLinear += 1

WeightBits = 8
RangeMax = 2 ** (WeightBits - 1)  # RangeMax = 2 ** (self.WeightBits - 1)
# print(RangeMax)                   # 128
numCellperWeight = 2
CellBits = 4

for ii in range(numLinear):
    # print("Start to compile fc layer-%d" % (ii + 1))
    weight, bias = Linear[ii]
    # print(weight.shape)       #(784,500)
    # print(bias.shape)         #(500,)
    Weight = np.concatenate((weight, bias.reshape(1, -1)), axis=0)
    # print(Weight.shape)       #(785,500)

    WeightMax = max(Weight.max(), abs(Weight.min()))
    # print(WeightMax)
    Weight = np.round(Weight / WeightMax * RangeMax)
    # print(Weight)
    Weight = np.where(Weight > (RangeMax - 1), RangeMax - 1, Weight)
    # print(Weight)
    # Weight += RangeMax
    # Weight = np.abs(Weight)
    # print(Weight)
    Weight = Weight+0.0
    arr_1.update({"Linear" + str(ii): Weight})
    weight = Weight[0:len(Weight)-1]
    bias = Weight[len(Weight)-1]
    # print(weight.shape)
    # print(bias.shape)
    # print(weight)
    Linear_1.update({'weight': weight})
    Linear_1.update({'bias': bias})
    arr_0.update({"Linear" + str(ii): Linear_1})
    Linear_1 = {}

for iii in range(numConv):
    # print("Start to compile conv layer-%d" % (iii + 1))
    ConvKernel, bias = Conv[iii]
    # print(ConvKernel.shape)       #(20,1,5,5)
    a2 = ConvKernel.shape[1]
    a3 = ConvKernel.shape[2]
    a4 = ConvKernel.shape[3]
    # print(bias.shape)         #(20,)
    ConvKernel = ConvKernel[:, :, ::-1, ::-1].reshape(ConvKernel.shape[0], -1).T  # (25,20)
    Weight = np.concatenate((ConvKernel, bias.reshape(1, -1)), axis=0)
    # print(Weight.shape)       #(26,20)

    WeightMax = max(Weight.max(), abs(Weight.min()))
    # print(WeightMax)
    Weight = np.round(Weight / WeightMax * RangeMax)
    # print(Weight)
    Weight = np.where(Weight > (RangeMax - 1), RangeMax - 1, Weight)
    # print(Weight)
    # Weight += RangeMax
    # Weight = np.abs(Weight)
    # print(Weight)
    Weight = Weight+0.0
    arr_1.update({"Conv2d" + str(iii): Weight})
    ConvKernel = Weight[0:len(Weight)-1]
    bias = Weight[len(Weight)-1]
    ConvKernel = ConvKernel.T
    ConvKernel = ConvKernel.reshape(ConvKernel.shape[0], a2, a3, a4)
    # print(ConvKernel.shape)
    # print(bias.shape)
    # print(ConvKernel)
    Conv2d_1.update({'weight': ConvKernel})
    Conv2d_1.update({'bias': bias})
    arr_0.update({"Conv2d" + str(iii): Conv2d_1})
    Conv2d_1 = {}

# np.savez('mnist_mlp.npz', arr_0)
# np.savez('mnist_lenet.npz', arr_0)

keys = arr_1.iterkeys()
# index = 1
# writer = pd.ExcelWriter("mlp.xlsx")
# writer = pd.ExcelWriter("lenet.xlsx")
for key in keys:
    key = key.encode('unicode-escape').decode('string_escape')  # type from unicode to str
    print(key)
    print(arr_1[key].shape)
    # df = pd.DataFrame(data=arr_1[key])
    # df.to_excel(excel_writer=writer, sheet_name=("sheet"+str(index)))
    # index += 1

# writer.save()
# writer.close()

# print(type(arr_0))              # dict
# print(len(arr_0))               # 5
# print(len(arr_0['Linear1']))    # 2
# print(type(arr_0['Linear1']))   # dict
# print(type(arr_0['Linear0']['bias']))   # numpy.ndarray










