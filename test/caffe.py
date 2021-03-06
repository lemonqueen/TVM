"""Caffe frontend"""
import tvm
from tvm.ir import IRModule
from tvm import relay

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, infer_value, infer_value_simulated, get_name

import numpy as np
import copy

__all__ = ['from_caffe']

def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]


def dimension_picker(prefix, surfix=''):
    """Check that dimensions are supported."""
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
        raise tvm.error.OpAttributeUnImplemented(
            'Non-2D kernels are not supported for operator {}2d'.format(prefix))

    return _impl


def dimension_constraint():
    def _dim_check(args):
        if len(args['kernel_shape']) == 2:
            return True
        return False

    return _dim_check, "Only 2d kernel supported."


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
    """ A helper class for eltwise op converters.
    """
    name = ''
    @classmethod
    def _impl(cls, inputs, layer, params):
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(
            cls.name, len(inputs))
        #判断算子类型
        if layer.eltwise_param.operation == 0:
            #按元素相乘
            cls.name = 'multiply'            
        elif layer.eltwise_param.operation == 1:
            if len(layer.eltwise_param.coeff) == 2:
                # 按元素相减
                cls.name = 'subtract'
            else :
                # 按元素相加
                cls.name = 'add'                      
        elif layer.eltwise_param.operation == 2:
            #按元素求最大值
            cls.name = 'maximum'
        return get_relay_op(cls.name)(*inputs)


class Convolution(CaffeOpConverter):
    """ Operator converter for Conv.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # 提取超参数
        dilations = 1
        if layer.convolution_param.dilation != []:
            dilations = layer.convolution_param.dilation[0]
        ##填充pads
        pads = [0, 0]  # 默认为0
        if layer.convolution_param.pad != []:  # 若存在pad,则根据pad赋值
            pads = np.array([layer.convolution_param.pad] * 2).flatten().tolist()
        elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:  # 若存在pad_w,pad_h则根据其赋值
            pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w]
        ##步长strides
        strides = [1, 1]  # 默认为1
        if layer.convolution_param.stride != []:
            strides = np.array([layer.convolution_param.stride] * 2).flatten().tolist()
        ##卷积核尺寸kernel_shape
        kernel_size = np.array([layer.convolution_param.kernel_size] * 2).flatten().tolist()
        if layer.convolution_param.kernel_size == []:
            kernel_size = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
        ##分组group
        # import ipdb; ipdb.set_trace()
        group = layer.convolution_param.group

        # 将权重加入到input
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))

        args = {
              'kernel_shape': kernel_size,
              'strides': strides,
              'dilation': dilations,
              'padding': pads,
              'groups': group,
              'data_layout': 'NCHW',
            #   'kernel_layout': 'OIHW'
              }

        out = AttrCvt(
            op_name=dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                # 'dilations': ('dilation', 1),
                # 'pads': ('padding', 0),
                # 'strides': 'strides',
                # 'group': ('groups', 1),
            },
            custom_check=dimension_constraint())(inputs[:2], args, params)

        if layer.convolution_param.bias_term:
            out = _op.nn.bias_add(out, relay.Constant(tvm.nd.array(params[layer.name][1])))
        return out


class Pooling(CaffeOpConverter):
    """ Operator converter for Pooling.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        #判断是池化种类,最大池化、平均池化
        if layer.pooling_param.pool == 0:
            if layer.pooling_param.global_pooling == True:
                cls.name = 'global_max_pool'
            else:
                cls.name = 'max_pool'
        elif layer.pooling_param.pool == 1:
            if layer.pooling_param.global_pooling == True:
                cls.name = 'global_avg_pool'
            else:
                cls.name = 'avg_pool'
        #Layers[i].pooling_param.pool==2为随机池化

        # 提取超参数
        ##池化核尺寸
        kernel_shape = np.array([layer.pooling_param.kernel_size]*2).flatten().tolist()
        if layer.pooling_param.kernel_size == []:
            kernel_shape = [layer.pooling_param.kernel_h,layer.pooling_param.kernel_w]
        ##步长
        strides = [1, 1]#默认为1
        if layer.pooling_param.stride != []:
            strides = np.array([layer.pooling_param.stride]*2).flatten().tolist()
        ##填充
        pads = [0, 0]#默认为0
        # 这里与卷积时一样,有pad,就按其值设置
        if layer.pooling_param.pad != []:
            pads = np.array([layer.pooling_param.pad] * 2).flatten().tolist()
        elif layer.pooling_param.pad_h != 0 or layer.pooling_param.pad_w != 0:
            pads = [layer.pooling_param.pad_h,layer.pooling_param.pad_w]

        args = {
              'kernel_shape': kernel_shape,
              'strides': strides,
              'padding': pads,
              }

        out = AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
            },
            custom_check=dimension_constraint())(inputs, args, params)
        return out


class InnerProduct(CaffeOpConverter):
    """ Operator converter for InnerProduct.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # 将权重加入到input
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))

        inputs[0] = _op.nn.batch_flatten(inputs[0])
        units = infer_channels(inputs[1])
        out = _op.nn.dense(inputs[0], inputs[1], units=units)
        if layer.inner_product_param.bias_term:
            out = _op.nn.bias_add(out, relay.Constant(tvm.nd.array(params[layer.name][1])))
        return out


class ReLU(CaffeOpConverter):
    """ Operator converter for ReLU.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # relu
        if layer.relu_param.negative_slope == 0:
            return _op.nn.relu(inputs[0])
        # LeakyRelu 
        else:
            return _op.nn.leaky_relu(inputs[0], alpha=float(layer.relu_param.negative_slope))


class PReLU(CaffeOpConverter):
    """ Operator converter for PReLU.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        # 将权重加入到input
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))
        assert len(inputs) == 2, "Prelu need 2 inputs, {} given".format(len(inputs))
        return _op.nn.prelu(inputs[0], inputs[1])


class Concat(CaffeOpConverter):
    """ Operator converter for Concat.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        return _op.concatenate(_as_list(inputs), axis = layer.concat_param.axis)


class BatchNorm(CaffeOpConverter):
    """ Operator converter for BatchNorm.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][1])))
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][2])))
        args = {
                "epsilon": layer.batch_norm_param.eps,  # 滑动系数
                "momentum": layer.batch_norm_param.moving_average_fraction
                }
        return AttrCvt(
            op_name='batch_norm',
            disables=['momentum'],
            ignores=[
                'order', 'spatial', 'is_test', 'consumed_inputs', 'num_batches'
            ])(inputs, args, params)


class Deconvolution(CaffeOpConverter):
    """ Operator converter for Deconvolution.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # 提取超参数
        ##膨胀系数dilations
        dilations = [1, 1]
        if layer.convolution_param.dilation != []:
            dilation = layer.convolution_param.dilation[0]
            dilations = [dilation, dilation]
        ##填充pads
        pads = [0, 0]  # 默认为0
        if layer.convolution_param.pad != []:  # 若存在pad,则根据pad赋值
            pads = np.array([layer.convolution_param.pad] * 2).reshape(1, -1)[0].tolist()
        elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:  # 若存在pad_w,pad_h则根据其赋值
            pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w]
        ##步长strides
        strides = [1, 1]  # 默认为1
        if layer.convolution_param.stride != []:
            strides = np.array([layer.convolution_param.stride] * 2).reshape(1, -1)[0].tolist()
        ##卷积核尺寸kernel_shape
        kernel_shape = np.array([layer.convolution_param.kernel_size] * 2).reshape(1, -1)[0].tolist()
        if layer.convolution_param.kernel_size == []:
            kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
        ##分组group
        # 只支持group=1的情况
        group = layer.convolution_param.group

        # 将权重加入到input
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))

        args = {
              'kernel_shape': kernel_shape,
              'strides': strides,
              'dilation': dilations,
              'padding': pads,
              'groups': group,
              'data_layout': 'NCHW',
            #   'kernel_layout': 'OIHW'
              }

        out = AttrCvt(
            op_name=dimension_picker('conv', '_transpose'),
            transforms={
                'kernel_shape': 'kernel_size',
                # 'dilations': ('dilation', 1),
                # 'pads': ('padding', 0),
                # 'strides': 'strides',
                'group': ('groups', 1),
            },
            custom_check=dimension_constraint())(inputs[:2], args, params)

        if layer.convolution_param.bias_term:
            out = _op.nn.bias_add(out, relay.Constant(tvm.nd.array(params[layer.name][1])))
        return out


class Scale(CaffeOpConverter):
    """ Operator converter for Scale.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        import ipdb; ipdb.set_trace()
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][1])))
        scale = float(attr.get('scale', 1.0))
        return inputs[0] * _expr.const(scale)


class Softmax(CaffeOpConverter):
    """ Operator converter for Softmax.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args={'axis':layer.softmax_param.axis}
        return AttrCvt('softmax', transforms={'axis': ('axis', args['axis'])})(inputs, args, params)


class LRN(CaffeOpConverter):
    """ Operator converter for LRN.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args = {
              'size': layer.lrn_param.local_size,
              'axis': 1,
              'bias': 1.0,
              'alpha': layer.lrn_param.alpha,
              'beta': layer.lrn_param.beta,
              }
        return AttrCvt('lrn')(inputs, args)


class Dropout(CaffeOpConverter):
    """ Operator converter for Dropout.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args={'rate':layer.dropout_param.dropout_ratio}
        return AttrCvt('dropout')(inputs, args)


class TanH(CaffeOpConverter):
    """ Operator converter for TanH.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args={}
        return AttrCvt('tanh')(inputs, args)


class Sigmoid(CaffeOpConverter):
    """ Operator converter for Sigmoid.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args={}
        return AttrCvt('sigmoid')(inputs, args)


class ELU(CaffeOpConverter):
    """ Operator converter for ELU.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        alpha = layer.elu_param.alpha
        return _expr.const(-alpha) * _op.nn.relu(_expr.const(1.) - _op.exp(inputs[0])) + \
                                     _op.nn.relu(inputs[0])


class Exp(CaffeOpConverter):
    """ Operator converter for Exp.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        base = layer.exp_param.base
        scale = layer.exp_param.scale
        shift = layer.exp_param.shift
        if base == -1:
            # y = exp(shift + scale * x)
            return _op.exp(_expr.const(shift) + _expr.const(scale) * inputs[0])
        else:
            # y = base ^ (shift + scale * x)
            return _op.power( _expr.const(base) , (_expr.const(shift) + _expr.const(scale) * inputs[0]))


class Power(CaffeOpConverter):
    """ Operator converter for Power.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        power = layer.power_param.power
        scale = layer.power_param.scale
        shift = layer.power_param.shift
        # y = (shift + scale * x) ^ power
        return _op.power( _expr.const(shift) + (_expr.const(scale) * inputs[0]), _expr.const(power) )


class Log(CaffeOpConverter):
    """ Operator converter for Log.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        base = layer.log_param.base
        scale = layer.log_param.scale
        shift = layer.log_param.shift
        if base == -1:
            # y = ln(shift + scale * x) = log_e(shift + scale * x)
            return _op.log(_expr.const(shift) + _expr.const(scale) * inputs[0])
        else:
            # y = log_base(shift + scale * x)
            return _op.log( _expr.const(shift) + _expr.const(scale) * inputs[0])/_op.log(_expr.const(base))


class BNLL(CaffeOpConverter):
    """ Operator converter for BNLL.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # y = log(1 + exp(x))
        return _op.log(  _expr.const(1.0) + _op.exp(inputs[0]))


class AbsVal(CaffeOpConverter):
    """ Operator converter for AbsVal.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # y = log(1 + exp(x))
        args={}
        return AttrCvt('abs')(inputs, args)


class ArgMax(CaffeOpConverter):
    """ Operator converter for ArgMax.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        args = {
                "k": layer.argmax_param.top_k,
                "axis": layer.argmax_param.axis
                }
        if layer.argmax_param.out_max_val :
            # If true produce pairs (argmax, maxval),如果为真，产生 (argmax, maxval)数据对
            return AttrCvt(op_name='topk')(inputs, args, params)                            
        else:
            return AttrCvt(op_name='topk',extras={"ret_type": "indices"})(inputs, args, params)    


class Crop(CaffeOpConverter):
    """ Operator converter for Crop.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        import ipdb; ipdb.set_trace()

        reshaped_permuted_shape = _infer_shape(reshaped_permuted)
        cropped = reshaped_permuted
        for axis in range(1, M+1):
            crop = crops[axis - 1]
            if crop != [0, 0]:
                indices = tvm.relay.arange(
                    _expr.const(crop[0]),
                    _expr.const(reshaped_permuted_shape[axis] - crop[1]),
                    dtype='int32'
                )
                cropped = tvm.relay.take(cropped, indices=indices, axis=axis)

        return cropped


class flatten(CaffeOpConverter):
    """ Operator converter for flatten.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        axis = layer.flatten_param.axis
        if axis == 1:
            out = _op.nn.batch_flatten(inputs[0])
        else:
            newshape = [0] * (axis + 1)
            newshape[axis] = -1
            out = _op.reshape(inputs[0], list(newshape))
        return out


class Embed(CaffeOpConverter):
    """ Operator converter for Embed.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        import ipdb; ipdb.set_trace()
        # 将权重加入到input
        inputs.append(relay.Constant(tvm.nd.array(params[layer.name][0])))
        if layer.embed_param.bias_term:
            out = _op.nn.bias_add(out, relay.Constant(tvm.nd.array(params[layer.name][1])))
        

        return _op.take(weight, indices.astype('int32'), axis=0)


class Reduction(CaffeOpConverter):
    """ Operator converter for Reduction.
        SUM = 1;ASUM = 2;SUMSQ = 3;MEAN = 4;
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        axis = layer.reduction_param.axis
        # import ipdb; ipdb.set_trace()        
        if layer.reduction_param.operation == 1:
            # SUM = 1;
            return _op.sum(inputs[0], axis=axis, keepdims='False', exclude='False')

        elif layer.reduction_param.operation == 2:
            # ASUM = 2;
            pass
        elif layer.reduction_param.operation == 3:
            # SUMSQ = 3;
            pass
        else:
            # MEAN = 4;
            return _op.mean(inputs[0], axis=axis, keepdims='False', exclude='False')


class Tile(CaffeOpConverter):
    """ Operator converter for Tile.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        # import ipdb; ipdb.set_trace()
        repeats = layer.tile_param.tiles
        axis = layer.tile_param.axis
        return _op.repeat(inputs[0], repeats, axis)


class Slice(CaffeOpConverter):
    """ Operator converter for Slice.
    """

    @classmethod
    def _impl(cls, inputs, layer, params):
        
        new_attrs = {}
        # begin = list(layer.slice_param.slice_point)
        # begin.insert(0,0)
        # import ipdb; ipdb.set_trace()
        input_shape = inputs[0].type_annotation.shape
        stride = layer.slice_param.slice_point
        axis = layer.slice_param.axis
        new_attrs = {'begin': [0], 'end': [input_shape[axis]]}
        if stride is not None:
            new_attrs['strides'] = stride
        return _op.strided_slice(inputs[0], **new_attrs)


_identity_list = []


def _get_convert_map():
    return {
        # caffe common operators
        'Eltwise': Eltwise.get_converter(),
        'Convolution': Convolution.get_converter(),
        'Pooling': Pooling.get_converter(),
        'InnerProduct':InnerProduct.get_converter(),
        'ReLU': ReLU.get_converter(),
        'PReLU': PReLU.get_converter(),
        'Concat': Concat.get_converter(),
        'BatchNorm': BatchNorm.get_converter(),
        'Scale': Scale.get_converter(),
        'Softmax': Softmax.get_converter(),
        'LRN': LRN.get_converter(),
        'Dropout': Dropout.get_converter(),
        'Deconvolution': Deconvolution.get_converter(),
        'TanH': TanH.get_converter(),
        'Sigmoid': Sigmoid.get_converter(),
        'ELU': ELU.get_converter(),
        'Exp': Exp.get_converter(),
        'Power': Power.get_converter(),
        'Log': Log.get_converter(),
        'BNLL': BNLL.get_converter(),
        'AbsVal': AbsVal.get_converter(),
        'ArgMax': ArgMax.get_converter(),
        'Crop': Crop.get_converter(),
        'Flatten': flatten.get_converter(),
        'Embed': Embed.get_converter(),
        'Reduction': Reduction.get_converter(),
        'Tile': Tile.get_converter(),
        'Slice': Slice.get_converter(),

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
    def __addInputsandGetLayerList(self,net):
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

    # 得到layer的参数ParamData并reshape
    def __getParamsData(self, layer):
        ParamShape = []
        ParamData = []
        #根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                # ParamShape = [(p.shape.dim) for p in Params]
                # ParamData = [(p.data) for p in Params]

                ParamShape = [np.array(p.shape.dim) for p in Params]
                # ParamData = [np.array(p.data) for p in Params]
                for i in range(len(Params)):
                    # 第i个paramdata重排成规定的shape
                    npParam = np.array(Params[i].data,dtype='float32')
                    paramdata = npParam.reshape((ParamShape[i]))
                    # paramdata = np.array(Params[i].data,dtype='float32').reshape((ParamShape[i]))
                    ParamData.append(paramdata)
                # if layer.type == "BatchNorm" or layer.type == "BN": 
                #     # import ipdb; ipdb.set_trace()
                #     if len(ParamShape) == 3:  
                #         # 如果是bn层，则不用最后一层的滑动系数
                #         ParamShape = ParamShape[:-1]
                #         ParamData = ParamData[:-1]
                #     elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                #         ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                #         ParamData = ParamData
        return ParamData


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
        调用转化函数得到转化好的tvm_op，并保存到self._nodes中
        '''

        inputs = []
        print("4.处理op",layer.type)
        for input in layer.bottom:
            inputs.append(self._nodes[input])
        tvm_op = self._convert_operator(layer, inputs)

        # 将输出节点替换成转化后的tvm_op function,保存到self._nodes
        print("\ntvm_op\n",tvm_op)

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
        self._LayerList = self.__addInputsandGetLayerList(net)

        # Params 待修改 batchnorm quanlianjie juanji 
        self._params = {}
        for layer in self._LayerList:
            ParamData = self.__getParamsData(layer)
            ParamNum = len(ParamData)
            if ParamNum != 0:
                ParamDataList = []
                for i in range(ParamNum):
                    ParamDataList.append(ParamData[i])
                self._params[layer.name]=ParamDataList

        # for layer_name, param in caffenet.params.items():
        #     self._params[layer_name] = _nd.array(param[0].data)

        print("3. 添加para字典啦：self._params")
              
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
            import ipdb; ipdb.set_trace()
            # for i in range(len(self._LayerList)):
            #     if self.judgeoutput(self._LayerList[i],self._LayerList):
            #         layer = self._LayerList[i]
            #         for top in layer.top:
            #             out.append(self._nodes[top])
            layer = self._LayerList[-1]
            for top in layer.top:
                out.append(self._nodes[top])

            
        else:
            # outputs若存在,必须是layer.name里的内容
            for out_name in outputs:
                for layer in self._LayerList:
                    if layer.name == out_name:
                        for top in layer.top:
                            out.append(self._nodes[top])

        print("添加out",type(out))
        out = out[0] if len(out) == 1 else _expr.Tuple(out)
        func = _function.Function(analysis.free_vars(out), out)
        # import ipdb; ipdb.set_trace()
        self._mod["main"] = func
        return self._mod, None
        # return self._mod, self._params


    def _convert_operator(self,
                          layer,
                          inputs,
                          identity_list=None,
                          convert_map=None):
        """Convert from Caffe operator to Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        layer : caffenet.layer
            The Caffe layer to be converted.
        inputs : list of tvm.relay.function.Function
            List of input inputs.
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
            # import ipdb; ipdb.set_trace()
            func = convert_map[layer.type](inputs, layer, self._params)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend Caffe.'.format(layer.type))
        return func

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
    net_path : the protobuf path

    model_path : the caffemodel path

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
