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
#pragma once

#include <list>
#include <memory>

#include <cudnn.h>

#include "ngraph/log.hpp"
#include "ngraph/runtime/gpu/gpu_host_parameters.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            /// \brief A factory which builds cuDNN host parameters
            /// and manages their creation and destruction.
            class CUDNNHostParameters
            {
            public:
                CUDNNHostParameters(const std::shared_ptr<GPUHostParameters> params)
                    : m_host_parameters(params)
                {
                }
                ~CUDNNHostParameters() = default;

                void* allocate_by_datatype(const cudnnDataType_t data_type, const double value)
                {
                    void* r = nullptr;
                    switch (data_type)
                    {
                    case CUDNN_DATA_FLOAT:
                        r = m_host_parameters->cache(static_cast<float>(value));
                        break;
                    case CUDNN_DATA_DOUBLE:
                        r = m_host_parameters->cache(static_cast<double>(value));
                        break;
                    case CUDNN_DATA_INT8:
                        r = m_host_parameters->cache(static_cast<int8_t>(value));
                        break;
                    case CUDNN_DATA_INT32:
                        r = m_host_parameters->cache(static_cast<int32_t>(value));
                        break;
                    case CUDNN_DATA_HALF:
                    case CUDNN_DATA_INT8x4:
#if CUDNN_VERSION >= 7100
                    case CUDNN_DATA_UINT8:
                    case CUDNN_DATA_UINT8x4:
#endif
                        std::string err = "datatype is not supported by cuDNN";
                        throw std::runtime_error(err);
                    }
                    return r;
                }

            private:
                std::shared_ptr<GPUHostParameters> m_host_parameters;
            };
        }
    }
}
