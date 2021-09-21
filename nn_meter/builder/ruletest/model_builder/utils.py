from tensorflow import keras
from . import layers
from .simple_model import SingleOpModel, TwoOpModel


def get_layer_by_name(name, input_shape):
    return getattr(layers, name)(input_shape)

def get_inputs_by_shapes(shapes):
    if len(shapes) == 1:
        return keras.Input(shape=shapes[0])
    else:
        return [keras.Input(shape=shape) for shape in shapes]

def get_model_by_ops(op1, op2, input_shape):
    layer1, op1_output_shape, op1_is_two_inputs = get_layer_by_name(op1, input_shape)
    layer2, _, op2_is_two_inputs = get_layer_by_name(op2, op1_output_shape)

    op1_model = SingleOpModel(layer1)
    op1_shapes = [input_shape] * (1 + op1_is_two_inputs)
    op1_model(get_inputs_by_shapes(op1_shapes))

    op2_model = SingleOpModel(layer2)
    op2_shapes = [op1_output_shape] * (1 + op2_is_two_inputs)
    op2_model(get_inputs_by_shapes(op2_shapes))

    block_model = TwoOpModel(layer1, layer2, op1_is_two_inputs, op2_is_two_inputs)
    block_shapes = [input_shape] * (1 + op1_is_two_inputs) + [op1_output_shape] * op2_is_two_inputs
    block_model(get_inputs_by_shapes(block_shapes))

    return op1_model, op2_model, block_model, op1_shapes, op2_shapes, block_shapes
