import numpy as np
root='/home/lemon/Documents/python_do_caffe/test/eltwise/'   #根目录
net_path=root + 'test_subtract.prototxt'    #deploy文件
model_path=root + 'test_subtract.caffemodel'   #caffemodel
data_shape = (10, 1, 28, 28)
dtype = 'float32'
data1 = np.random.uniform(size=data_shape).astype(dtype)  # 生成测试数据
data2 = np.random.uniform(size=data_shape).astype(dtype)
# print(data)
import caffe
# 将网络恢复，输入数据，运行网络，得到输出
net = caffe.Net(net_path,model_path,caffe.TEST)
net.blobs['data1'].data[...]= data1
net.blobs['data2'].data[...]= data2
net.forward()
print("net.blobs['data1'].data)",net.blobs['data1'].data)
print("net.blobs['data2'].data)",net.blobs['data2'].data)
print("net.blobs['dataout'].data)",net.blobs['dataout'].data)
