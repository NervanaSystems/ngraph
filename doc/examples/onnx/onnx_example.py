# ******************************************************************************
# Copyright 2018 Intel Corporation
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

onnx_protobuf = onnx.load('/path/to/model/cntk_ResNet20_CIFAR10/model.onnx')

# Convert a serialized ONNX model to an ngraph model
from ngraph_onnx.onnx_importer.importer import import_onnx_model
ng_model = import_onnx_model(onnx_protobuf)[0]


# Using an ngraph runtime (CPU backend), create a callable computation
import ngraph as ng
runtime = ng.runtime(backend_name='CPU')
resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])

# Load or create an image
import numpy as np
picture = np.ones([1, 3, 32, 32])

# Run ResNet inference on picture
resnet(picture)

