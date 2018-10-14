# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'input_layer', None, 'name of input tensor')

tf.app.flags.DEFINE_string(
    'output_layer', None, 'name of output tensor')

tf.app.flags.DEFINE_string(
    'graph_file', None, 'path to the graph file to be exported')

tf.app.flags.DEFINE_string(
    'output_dir', None, 'path to save the graph file to')

FLAGS = tf.app.flags.FLAGS

def build_prediction_signature(graph):
    input_name = "import/" + FLAGS.input_layer
    output_name = "import/" + FLAGS.output_layer
    input_tensor = graph.get_tensor_by_name(input_name + ':0')
    output_tensor = graph.get_tensor_by_name(output_name + ':0')
    tensor_info_input = utils.build_tensor_info(input_tensor)
    tensor_info_output = utils.build_tensor_info(output_tensor)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs = {'input': tensor_info_input},
        outputs = {'output': tensor_info_output},
        method_name = signature_constants.PREDICT_METHOD_NAME
    )

    return prediction_signature

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def export(graph_file, output_dir):
    graph = load_graph(graph_file)
    with tf.Session(graph = graph) as sess:
        # Build and save prediction meta graph and trained variable values.
        prediction_signature = build_prediction_signature(graph)

        # Create a saver for writing SavedModel training checkpoints.
        build = builder.SavedModelBuilder(
            os.path.join(output_dir, 'saved_model'))
        build.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
        build.save()

def main(_):
    export(FLAGS.graph_file, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()

