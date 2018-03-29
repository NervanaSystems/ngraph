/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn_v7.h>

using namespace ngraph;
using namespace ngraph::runtime;

int CUDNNEmitter::build_test() {

    m_cudnn_primitives.emplace_back([=] () {
            // cudnnTensorDescriptor_t desc;
            // cudnnCreateTensorDescriptor(&desc);
            std::cout << "Hi" << std::endl;
        });
    return 0;
}
