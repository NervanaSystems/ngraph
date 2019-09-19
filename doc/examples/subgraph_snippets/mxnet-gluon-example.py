# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

import mxnet as mx

# Convert gluon model to a static model
from mxnet.gluon.model_zoo import vision
import time

batch_shape = (1, 3, 224, 224)

input_data = mx.nd.zeros(batch_shape)

resnet_gluon = vision.resnet50_v2(pretrained=True)
resnet_gluon.hybridize()
resnet_gluon.forward(input_data)
resnet_gluon.export('resnet50_v2')
resnet_sym, arg_params, aux_params = mx.model.load_checkpoint('resnet50_v2', 0)

# Load the model into nGraph as a static graph
model = resnet_sym.simple_bind(ctx=mx.cpu(), data=batch_shape, grad_req='null')
model.copy_params_from(arg_params, aux_params)

# To test the model's performance, we've provided this helpful code snippet
# customizable

dry_run = 5
num_batches = 100
for i in range(dry_run + num_batches):
   if i == dry_run:
       start_time = time.time()
   outputs = model.forward(data=input_data, is_train=False)
   for output in outputs:
       output.wait_to_read()
print("Average Latency = ", (time.time() - start_time)/num_batches * 1000, "ms")
