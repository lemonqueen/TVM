"""Caffe frontend"""
import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_channels

import copy
import numpy as np


__all__ = ['from_caffe']


class CaffeOpConverter(object):
    """ A helper class for holding Caffe op converters.
    """
    # todo:待修改
    @classmethod
    def get_converter(cls):
        """ Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, '_impl'):
            return getattr(cls, '_impl')
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported in frontend Caffe.'.format(cls.__name__))


class Eltwise(CaffeOpConverter):
    """ A helper class for elemwise op converters.
    """
    name = ''
    @classmethod
    def _impl(cls, inputs, layer, params):
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(
            cls.name, len(inputs))
        #判断算子类型
        if layer.eltwise_param.operation == 0:
            print("这是乘法")
            #按元素相乘
            cls.name = 'multiply'
            
        elif layer.eltwise_param.operation == 1:
            if len(layer.eltwise_param.coeff) == 2:
                # 按元素相减
                print("这是减法")
                cls.name = 'subtract'
            else :
                # 按元素相加
                print("这是加法")
                cls.name = 'add'                      

        elif layer.eltwise_param.operation == 2:
            print("这是求最大值")
            #按元素求最大值
            cls.name = 'maximum'
        print("cls.name",cls.name)
        return get_relay_op(cls.name)(*inputs)


# compatible operators that do NOT require any conversion.
_identity_list = []


def _get_convert_map():
    return {
        # caffe common operators
        'Eltwise': Eltwise.get_converter()
    }


class CaffeNetDef(object):
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
    """

    def __init__(self, shape, dtype, net, model):
        #网络和参数
        self._NetLayer = self.__getNetLayer(net)
        self._ModelLayer = self.__getModelLayer(model)

        self._params = {}
        self._nodes = {}
        self._LayerList = []
        self._shape = shape
        self._dtype = dtype
        self._mod = IRModule({})

    #获取网络层
    def __getNetLayer(self,net):
        if len(net.layer)==0 and len(net.layers)!=0:
            return net.layers
        elif len(net.layer)!=0 and len(net.layers)==0:
            return net.layer
        else:
            print("prototxt layer error")
            return -1

    #获取参数层
    def __getModelLayer(self,model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        elif len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        else:
            print("caffemodel layer error")
            return -1

    #将模型输入信息添加到self._nodes中并获取后续层列表
    def __addInputsTVIandGetLayerList(self,net):
        #如果第一个layer的类型为Input,且没有net.input存在
        if net.input == [] and self._NetLayer[0].type == "Input":
            layer_list = []
            #考虑到整个网络会有多输入情况，把输入存起来，其它层放到layer_list
            for lay in self._NetLayer:
                if lay.type == "Input":        
                    input_shape=list(lay.input_param.shape[0].dim)  #将shape转化为list类型
                    # new_var的类型(str,list,str)
                    self._nodes[lay.top[0]] = new_var(lay.top[0], shape=input_shape, dtype='float32')
                    print("2. 添加模型输入信息",self._nodes[lay.top[0]])
                else:
                    layer_list.append(lay)
            return layer_list
        # elif net.input == [] and self._NetLayer[0].type == "DummyData":
        #     layer_list = []
        #     #考虑到整个网络会有多输入情况，把输入存起来，其它层放到layer_list
        #     for lay in self._NetLayer:
        #         if lay.type == "DummyData":   
        #             print("lay.top",lay.top[0])
        #             print("lay.dummy_data_param.num",lay.dummy_data_param.num[0])
        #             print("type(lay.dummy_data_param.num[0])",type(lay.dummy_data_param.num[0]))
        #             input_shape=[lay.dummy_data_param.num[0],lay.dummy_data_param.channels[0],lay.dummy_data_param.height[0],lay.dummy_data_param.width[0]]
        #             print(type(input_shape))
        #             print("lay.shape",input_shape)
        #             print(type(input_shape[0]))
        #             self._nodes[lay.top] = new_var(lay.top, shape=input_shape, dtype='float32')
        #             print("添加模型输入信息")
        #         else:
        #             layer_list.append(lay)
        #     return layer_list

        #如果存在net.input，只存输入，其它层不需要处理
        elif net.input !=[]:
            if bool(net.input_dim):
                input_dim = net.input_dim
            elif bool(net.input_shape):
                input_dim = net.input_shape[0].dim
            else:
                raise RuntimeError("Input shape missing!")
            self._nodes[net.input] = new_var(net.input, shape=input_dim, dtype='float32')
            print("2. 添加模型输入信息")
            return self._NetLayer

        #以上情况都不是,则该caffe模型没有输入,存在问题
        else:
            raise ValueError("the caffe model has no input")

    # 得到layer的参数shape
    def __getParamsShapeandData(self, layer):
        ParamShape = []
        ParamData = []
        #根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN": 
                    if len(ParamShape) == 3:  
                        # 如果是bn层，则不用最后一层的滑动系数
                        ParamShape = ParamShape[:-1]
                        ParamData = ParamData[:-1]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = ParamData
        return ParamShape, ParamData

 
    #判断当前节点是否是输出节点
    def judgeoutput(self,current_layer,layerlist):
        for outname in current_layer.top:
            for layer in layerlist:
                if outname in layer.bottom:
                    return False
        return True


    # 处理op
    def _process_op(self, layer):
        '''根据layer.bottom找到输入
        从self._nodes找到输入
        从_get_convert_map找到转化函数
        调用转化函数得到转化好的tvm_op，并保存到_convert_relay_map中
        '''

        inputs = []
        print("4.处理op")
        for input in layer.bottom:
            inputs.append(self._nodes[input])
        print("inputs",inputs,type(inputs) )
        convert_map = _get_convert_map()
        tvm_op = convert_map[layer.type](inputs, layer, self._params)

        # 将输出节点替换成转化后的tvm_op function,保存到self._nodes
        print("\ntvm_op\n",tvm_op)
        # print("\nlayer.top",layer.top,type(layer.top) )
        # print("\nlayer.top[0]",layer.top[0],type(layer.top[0]) )
        if not isinstance(tvm_op, _expr.TupleWrapper):
            self._nodes[layer.top[0]] = tvm_op
        else:
            for k, i in zip(list(layer.top), range(len(tvm_op))):
                self._nodes[k] = tvm_op[i]


    def from_caffe(self, net, model, outputs=None):
        """Construct Relay expression from caffe graph.

        Parameters
        ----------
        net : protobuf object

        model : caffemodel object

        Returns
        -------
        mod : tvm.IRModule
            The module that optimizations will be performed on.

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # pylint: disable=import-outside-toplevel

        #获取层列表，除去数据层每一层都存在了list格式的self._LayerList里面,将输入放到self._nodes
        self._LayerList = self.__addInputsTVIandGetLayerList(net)

        # Params 待修改
        self._params = {}
        for layer in self._LayerList:
            self._params[layer.name] = _nd.array(0)
        # for layer_name, param in caffenet.params.items():
        #     self._params[layer_name] = _nd.array(param[0].data)
        print("3. 添加para字典啦：self._params",self._params)
              
        # ops
        for layer in self._LayerList: 
            self._process_op(layer)

        # get list of unsupported ops
        convert_map = _get_convert_map()
        unsupported_ops = set()
        for layer in self._LayerList:
            op_name = layer.type
            if op_name not in convert_map and \
               op_name != 'Constant' and \
               op_name not in _identity_list:
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = 'The following operators are not supported for frontend caffe: '
            msg += ', '.join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # Outputs
        out = []
        print("5.处理输出")
        if outputs is None:
            # 寻找默认输出
            for i in range(len(self._LayerList)):
                if self.judgeoutput(self._LayerList[i],self._LayerList):
                    layer = self._LayerList[i]
                    for top in layer.top:
                        out.append(self._nodes[top])
        else:
            # outputs若存在,必须是layer.name里的内容
            for out_name in outputs:
                for layer in self._LayerList:
                    if layer.name == out_name:
                        for top in layer.top:
                            out.append(self._nodes[top])

        print("添加out",out,type(out) )
        out = out[0] if len(out) == 1 else _expr.Tuple(out)
        func = _function.Function(analysis.free_vars(out), out)
        self._mod["main"] = func
        return self._mod, self._params


    def _convert_operator(self,
                          layer,
                          identity_list=None,
                          convert_map=None):
        """Convert from Caffe2 operator to Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_type : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of input inputs.
        args : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take args and return (new_op_type, new_args)

        Returns
        -------
        func : tvm.relay.function.Function
            Converted relay function
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _get_convert_map()
        if layer.type in identity_list:
            pass
            # func = get_relay_op(layer.type)(*inputs, **args)
        elif layer.type in convert_map:
            # Add a sanitizing step to convert all byte strings in args to strings
            func = convert_map[layer.type](layer, self._params)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend Caffe.'.format(layer.type))
        return func

# from google.protobuf import text_format
# from .caffeproto import caffe_upsample_pb2 as caffe_pb2

def loadcaffemodel(net_path,model_path):
    from google.protobuf import text_format
    try:
        import caffe.proto.caffe_pb2 as caffe_pb2
    except:
        from .caffeproto import caffe_upsample_pb2 as caffe_pb2
    # read prototxt
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    print("1.caffe模型加载完成")
    return net,model

def from_caffe(net_path, model_path, shape=None, dtype="float32", outputs=None):
    """Load caffe graph which contains net into Relay Function.

    Parameters
    ----------
    net : protobuf object
        Caffe NetDef containing the graph

    shape : dict of str to tuple
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format
    """
    net, model = loadcaffemodel(net_path, model_path)
    caffedef = CaffeNetDef(shape, dtype, net, model)
    return caffedef.from_caffe(net, model, outputs)
