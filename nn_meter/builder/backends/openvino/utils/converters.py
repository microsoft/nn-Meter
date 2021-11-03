# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os


def keras_model_to_frozenpb(model, frozen_out_path, frozen_graph_filename, shapes, dtype=tf.dtypes.float32):
    full_model = tf.function(lambda x: model(x))
    if len(shapes) == 1:
        tensor_specs = tf.TensorSpec([1] + shapes[0], dtype)
    else:
        tensor_specs = [tf.TensorSpec([1] + shape, dtype) for shape in shapes]
    full_model = full_model.get_concrete_function(tensor_specs)

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    frozen_graph_filename_bin = frozen_graph_filename + '.pb'
    frozen_graph_filename_txt = frozen_graph_filename + '.pbtxt'

    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=frozen_graph_filename_bin,
        as_text=False
    )
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=frozen_graph_filename_txt,
        as_text=True
    )

    return (
        os.path.join(frozen_out_path, frozen_graph_filename_bin),
        os.path.join(frozen_out_path, frozen_graph_filename_txt),
    )
