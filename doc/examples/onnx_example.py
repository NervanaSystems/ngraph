# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import onnx
from ngraph.frontends.onnx.onnx_importer.model_wrappers import ModelWrapper


def import_onnx_model(onnx_protobuf):  # type: (onnx.ModelProto) -> List[Dict]
    """
    Import an ONNX Protocol Buffers model and convert it into a list of ngraph operations.

    An ONNX model defines a set of output nodes. Each output node will be added to the
    returned list as a dict with the following fields:

    * 'name' - name of the output, as specified in the imported ONNX model
    * 'inputs' - a list of ngraph placeholder ops, used to feed data into the model
    * 'output' - ngraph Op representing the output of the model

    Usage example:

    >>> onnx_protobuf = onnx.load('y_equals_a_plus_b.onnx.pb')
    >>> import_onnx_model(onnx_protobuf)
    [{
        'name': 'Y',
        'inputs': [<AssignableTensorOp(placeholder):4552991464>,
                   <AssignableTensorOp(placeholder):4510192360>],
        'output': <Add(Add_0):4552894504>
    }]

    >>> ng_model = import_onnx_model(model)[0]
    >>> transformer = ng.transformers.make_transformer()
    >>> computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    >>> computation(4, 6)
    array([ 10.], dtype=float32)

    :param onnx_protobuf: ONNX Protocol Buffers model (onnx_pb2.ModelProto object)
    :return: list of dicts representing ngraph Ops and their inputs
    """
    model = ModelWrapper(onnx_protobuf)
    return model.graph.get_ng_model()


def import_onnx_file(filename):  # type: (str) -> List[Dict]
    """
    Import ONNX model from a Protocol Buffers file and convert to ngraph operations.

    :param filename: path to an ONNX file
    :return: List of imported ngraph Ops (see docs for import_onnx_model).
    """
    onnx_protobuf = onnx.load(filename)
    return import_onnx_model(onnx_protobuf)

