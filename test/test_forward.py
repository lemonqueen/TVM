# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import tvm

from tvm import te
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm import relay
from collections import namedtuple

import pytest
from caffe import params as P
from caffe import layers as L

import os.path as osp
import os
this_dir = osp.dirname(__file__)
zoo_path = this_dir+'/model_zoo'
if not os.path.exists(zoo_path):
    os.mkdir(zoo_path)

    
def get_tvm_output(net_path, 
                   model_path,
                   input_data,
                   target,
                   ctx):
    """ Generic function to execute and get tvm output"""
    # supporting multiple inputs in caffe in a bit tricky,
    # because the input names can appear at the beginning or end of model.predict_net.external_input
    assert isinstance(input_data, np.ndarray)
    '''
    通过tvm测试
    '''
    # here we use the first input blob to the first op to get the input name

    shape_dict = {'data': input_data.shape}
    dtype_dict = {'data': input_data.dtype}
    mod, params = relay.frontend.from_caffe(
        net_path, model_path, shape_dict, dtype_dict)
    print("6.在relay上编译模型")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)

    num_outputs = m.get_num_outputs()
    if num_outputs == 1:
        output_shape = m.get_output(0).shape # 单输出
        output_dtype = m.get_output(0).dtype
    else:     
        output_shape = [] # 多输出
        output_dtype = []
        for i in range(m.get_num_outputs()):
            output_shape.append(m.get_output(i).shape)
            output_dtype.append(m.get_output(i).dtype)

    # import ipdb; ipdb.set_trace()

    # set inputs
    m.set_input('data', tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input(**params)

    # execute
    m.run()
    print("7 已在relay上运行")
    ## get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        # 多输出
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        # 单输出
        tvm_output = m.get_output(0, tvm.nd.empty((output_shape),output_dtype))
        return tvm_output.asnumpy()

def get_caffe_output(net_path, model_path, x, dtype='float32'):
    '''
    直接通过caffe测试
    '''
    import caffe
    # 将网络恢复，输入数据，运行网络，得到输出
    net = caffe.Net(net_path,model_path,caffe.TEST)
    net.blobs['data'].data[...]= x
    #执行测试
    net.forward()
    # import ipdb; ipdb.set_trace()
    # return net.layers[n-1].blobs[0].data 
    return net.blobs['dataout'].data
    # return net.blobs['conv'].data

def get_tvm_output_eltwise(net_path, 
                   model_path,
                   input_data,
                   input_data2,
                   target,
                   ctx):
    """ Generic function to execute and get tvm output"""
    # supporting single inputs in caffe in a bit tricky,
    assert isinstance(input_data, np.ndarray)
    '''
    通过tvm测试
    '''
    shape_dict = {'data': input_data.shape}
    dtype_dict = {'data': input_data.dtype}
    mod, params = relay.frontend.from_caffe(
        net_path, model_path, shape_dict, dtype_dict)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)
    # 得到output_shape和output_dtype
    output_shape = m.get_output(0).shape # 单输出
    output_dtype = m.get_output(0).dtype

    # set inputs
    m.set_input('data1', tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input('data2', tvm.nd.array(input_data2.astype(input_data2.dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((output_shape),
                                                  output_dtype))
    return tvm_output.asnumpy()

def get_caffe_output_eltwise(net_path, model_path, x, y, dtype='float32'):
    '''
    直接通过caffe测试
    '''
    import caffe
    # 将网络恢复，输入数据，运行网络，得到输出
    net = caffe.Net(net_path,model_path,caffe.TEST)
    net.blobs['data1'].data[...]= x
    net.blobs['data2'].data[...]= y
    #执行测试
    net.forward()
    return net.blobs['dataout'].data

def verify_caffe_forward_impl(net_path, model_path, data_shape):
    dtype = 'float32'
    data = np.random.uniform(size=data_shape).astype(dtype)  # 生成测试数据
    c_out = get_caffe_output(net_path, model_path, data, dtype)
    # print("c_out\n",c_out,type(c_out) )
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(net_path, model_path, data, target, ctx)       
        # print("c_out\n",c_out,type(c_out) )
        # print("tvm_out\n",tvm_out,type(tvm_out) )
        # print("c_out\n",c_out.shape)
        # print("tvm_out\n",np.array(tvm_out).shape)
        tvm.testing.assert_allclose(c_out, tvm_out, rtol=1e-5, atol=1e-5)
    print("完成！！！！")

def verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape):
    dtype = 'float32'
    data1 = np.random.uniform(size=data_shape).astype(dtype)  # 生成测试数据
    data2 = np.random.uniform(size=data_shape).astype(dtype)
    c_out = get_caffe_output_eltwise(net_path, model_path, data1, data2, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output_eltwise(net_path, model_path, data1, data2, target, ctx)
        tvm.testing.assert_allclose(c_out, tvm_out, rtol=1e-5, atol=1e-5)
    print("完成！！！！")


eltwise_params = {
    (0),
    (1),
    (2),
}
@pytest.mark.parametrize("_operation",eltwise_params)
def test_eltwise(_operation):
    data_shape = [10, 1, 28, 28]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data1 = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.data2 = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Eltwise(n.data1,n.data2,eltwise_param=dict(operation=_operation))
        return n.to_proto()
    
    op_type = "eltwise"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape)

def test_eltwise_sub():
    data_shape = [10, 1, 28, 28]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data1 = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.data2 = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Eltwise(n.data1,n.data2,eltwise_param=dict(operation=1,coeff=[1,-1]))
        return n.to_proto()

    op_type = "eltwise_sub"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape)

conv_params = {
    (7,2,64,3,1),
    (6,3,56,4,1),
    (5,1,256,2,2)
}
@pytest.mark.parametrize("_kernel_size,_stride,_num_output,_pad,_group",conv_params)
def test_conv(_kernel_size,_stride,_num_output,_pad,_group):
    data_shape = [10, 4, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Convolution(n.data,param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)],kernel_size=_kernel_size,stride=_stride,num_output=_num_output,pad=_pad,group = _group,weight_filler=dict(type="xavier",std = 0.03),
                          bias_filler=dict(type='constant',value=0.2))
        return n.to_proto()

    op_type = "conv"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)

pool_params = {
    (2,2,0,P.Pooling.MAX),
    (2,4,1,P.Pooling.MAX),
    (7,1,0,P.Pooling.AVE),
}
# 不支持P.Pooling.STOCHASTIC
@pytest.mark.parametrize("_kernel_size,_stride,_pad,_pool",pool_params)
def test_pool(_kernel_size,_stride,_pad,_pool):
    data_shape = [10, 3, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Pooling(n.data,kernel_size=_kernel_size,stride=_kernel_size,pad=_pad,pool=_pool)
        return n.to_proto()

    op_type = "pool"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)

innerproduct_params = {
    (5, 0.03, 0.2),
    (3, 0.01, 0)
}
@pytest.mark.parametrize("_num_output,_std,_vaule",innerproduct_params)
def test_InnerProduct(_num_output,_std,_vaule):
    data_shape = [10, 3, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout=L.InnerProduct(n.data,param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=1,decay_mult=1)],inner_product_param=dict(num_output=_num_output,weight_filler=dict(type='xavier',std=_std),bias_filler=dict(type='constant',value=_vaule)))
        return n.to_proto()

    op_type = "InnerProduct"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)

def test_mnist():
    data_shape = [10, 3, 224, 224]
    import caffe
    
    def net():
        n = caffe.NetSpec()
        n.data=L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.ip1 = L.InnerProduct(n.data, num_output=500, weight_filler=dict(type='xavier'))
        n.relu1 = L.ReLU(n.ip1, in_place=True)
        n.dataout = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
        return n.to_proto()

    op_type = "mnist"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)


def test_relu():
    data_shape = [10, 3, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Convolution(n.data,param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)],kernel_size=7,stride=2,num_output=64,weight_filler=dict(type="xavier",std = 0.03))
        n.relu = L.ReLU(n.dataout, in_place=True)
        # n.relu = L.ReLU(n.dataout)
        return n.to_proto()

    op_type = "relu"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)

def test_BatchNorm():
    data_shape = [10, 3, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Convolution(n.data,param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)],kernel_size=7,stride=2,num_output=64,pad=3,weight_filler=dict(type="xavier",std = 0.03),
                          bias_filler=dict(type='constant',value=0.2))
        n.bn= L.BatchNorm(n.dataout, batch_norm_param=dict(moving_average_fraction=0.90),in_place=True)  # top=conv1
        # n.relu = L.ReLU(n.dataout)
        return n.to_proto()

    op_type = "BatchNorm"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)

def test_Scale():
    data_shape = [10, 3, 224, 224]
    import caffe

    def net():
        n = caffe.NetSpec()
        n.data = L.Input(input_param=dict(shape=dict(dim = data_shape)))
        n.dataout = L.Convolution(n.data,param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)],kernel_size=7,stride=2,num_output=64,pad=3,weight_filler=dict(type="xavier",std = 0.03),
                          bias_filler=dict(type='constant',value=0.2))
        n.scale=L.Scale(n.dataout,scale_param=dict(bias_term=True),in_place=True)
        return n.to_proto()

    op_type = "Scale"
    net_path, model_path = saveModel(op_type,str(net()))    
    verify_caffe_forward_impl(net_path, model_path, data_shape)


def saveModel(name,netproto):
    import caffe
    file_path = this_dir+'/model_zoo/'+name
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    net_path=file_path + '/test_'+name+'.prototxt'    #deploy文件
    model_path=file_path + '/test_'+name+'.caffemodel'   #caffemodel
    solver_path = file_path + '/test_'+name+'_solver.prototxt'
    
    with open(net_path, 'w') as f:
        f.write(netproto)
    with open(solver_path, 'w') as f:
        f.write("net:\""+net_path+"\"")

    #载入solver文件
    solver = caffe.SGDSolver(solver_path)
    solver.net.forward()
    solver.net.save(model_path)

    return net_path, model_path

if __name__ == '__main__':
    # test_eltwise(0)
    # test_eltwise_sub()
    # test_conv(7,2,64,3,1)
    # test_pool(2,2,0,P.Pooling.AVE)
    # test_InnerProduct(5, 0.03, 0.2)
    test_mnist()
    # test_relu()