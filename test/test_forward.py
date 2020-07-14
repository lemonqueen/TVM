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

def get_tvm_output(net_path, model_path,
                   input_data,
                   target,
                   ctx,
                   output_shape,
                   output_dtype='float32'):
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
    # print("graph\n",graph,type(graph) )
    # print("lib\n",lib,type(lib) )
    # print("params\n",params,type(params) )

    m = graph_runtime.create(graph, lib, ctx)

    # set inputs
    m.set_input('data', tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input(**params)

    # execute
    m.run()
    print("7 已在relay上运行")
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0, tvm.nd.empty((output_shape),
                                                  output_dtype))
        return tvm_output.asnumpy()


def get_caffe_output(net_path, model_path, x, dtype='float32'):
    '''
    直接通过caffe测试
    '''
    import caffe
    # 将网络恢复，输入数据，运行网络，得到输出
    net = caffe.Net(net_path,model_path,caffe.TEST)
    net.blobs['data1'].data[...]= x
    net.blobs['data2'].data[...]= x
    #执行测试
    net.forward()
    return net.blobs['dataout'].data


def get_tvm_output_eltwise(net_path, model_path,
                   input_data,
                   input_data2,
                   target,
                   ctx,
                   output_shape,
                   output_dtype='float32'):
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

    # set inputs

    m.set_input('data1', tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input('data2', tvm.nd.array(input_data2.astype(input_data2.dtype)))
    m.set_input(**params)

    # execute
    m.run()
    print("7 已在relay上运行")
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
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


def verify_caffe_forward_impl(net_path, model_path, data_shape, out_shape):
    dtype = 'float32'
    data = np.random.uniform(size=data_shape).astype(dtype)  # 生成测试数据
    c_out = get_caffe_output(net_path, model_path, data, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(net_path, model_path, data, target, ctx, out_shape, dtype)       
        print("c_out\n",c_out,type(c_out) )
        print("tvm_out\n",tvm_out,type(tvm_out) )
        tvm.testing.assert_allclose(c_out, tvm_out, rtol=1e-5, atol=1e-5)
    print("完成！！！！")

def verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape, out_shape):
    dtype = 'float32'
    data1 = np.random.uniform(size=data_shape).astype(dtype)  # 生成测试数据
    data2 = np.random.uniform(size=data_shape).astype(dtype)
    c_out = get_caffe_output_eltwise(net_path, model_path, data1, data2, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output_eltwise(net_path, model_path, data1, data2, target, ctx, out_shape, dtype)       
        tvm.testing.assert_allclose(c_out, tvm_out, rtol=1e-5, atol=1e-5)
    print("完成！！！！")

def test_eltwise_add():
    root='/home/lemon/Documents/python_do_caffe/test/eltwise/'   #根目录
    # root='/home/lemon/Documents/Tvm/tvm/tests/python/frontend/caffe/model_zoo/eltwise/'
    net_path=root + 'test_add.prototxt'    #deploy文件
    model_path=root + 'test_add.caffemodel'   #caffemodel
    data_shape = (10, 1, 28, 28)
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape, data_shape)

def test_eltwise_sub():
    root='/home/lemon/Documents/python_do_caffe/test/eltwise/'   #根目录
    net_path=root + 'test_subtract.prototxt'    #deploy文件
    model_path=root + 'test_subtract.caffemodel'   #caffemodel
    data_shape = (10, 1, 28, 28)
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape, data_shape)

def test_eltwise_mul():
    root='/home/lemon/Documents/python_do_caffe/test/eltwise/'   #根目录
    net_path=root + 'test_mul.prototxt'    #deploy文件
    model_path=root + 'test_mul.caffemodel'   #caffemodel
    data_shape = (10, 1, 28, 28)
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape, data_shape)

def test_eltwise_max():
    root='/home/lemon/Documents/python_do_caffe/test/eltwise/'   #根目录
    net_path=root + 'test_max.prototxt'    #deploy文件
    model_path=root + 'test_max.caffemodel'   #caffemodel
    data_shape = (10, 1, 28, 28)
    verify_caffe_forward_impl_eltwise(net_path, model_path, data_shape, data_shape)

if __name__ == '__main__':
    test_eltwise_add()
    # test_eltwise_sub()
    # test_eltwise_mul()
    # test_eltwise_max()

