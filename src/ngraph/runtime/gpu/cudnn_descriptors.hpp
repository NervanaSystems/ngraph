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

#include <memory>
#include <vector>

#include <cudnn.h>

#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <typename T>
            struct cudnn_descriptor;

            /// \brief A factory which builds cuDNN descriptors
            /// and manages their creation and destruction.
            class CUDNNDescriptors
            {
                class Descriptor
                {
                public:
                    Descriptor() = default;
                    virtual ~Descriptor() = default;
                };

            public:
                CUDNNDescriptors() = default;
                ~CUDNNDescriptors() = default;

                template <typename T>
                T& build()
                {
                    // A function-local class for type erased storage of
                    // various cudnn descriptor types which is templated
                    // over function `build`'s specialization type
                    class descriptor_ : public Descriptor
                    {
                    public:
                        descriptor_() { cudnn_descriptor<T>::create(stored_descriptor); }
                        ~descriptor_() { cudnn_descriptor<T>::destroy(stored_descriptor); }
                        T& get() { return stored_descriptor; }
                    private:
                        T stored_descriptor;
                    };

                    m_descriptors.emplace_back(new descriptor_);
                    return static_cast<descriptor_*>(m_descriptors.back().get())->get();
                }

            private:
                std::vector<std::unique_ptr<Descriptor>> m_descriptors;
            };

            template <>
            struct cudnn_descriptor<cudnnLRNDescriptor_t>
            {
                static void create(cudnnLRNDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateLRNDescriptor(&desc));
                }
                static void destroy(cudnnLRNDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyLRNDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnTensorDescriptor_t>
            {
                static void create(cudnnTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&desc));
                }
                static void destroy(cudnnTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyTensorDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnSpatialTransformerDescriptor_t>
            {
                static void create(cudnnSpatialTransformerDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateSpatialTransformerDescriptor(&desc));
                }
                static void destroy(cudnnSpatialTransformerDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroySpatialTransformerDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnReduceTensorDescriptor_t>
            {
                static void create(cudnnReduceTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateReduceTensorDescriptor(&desc));
                }
                static void destroy(cudnnReduceTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyReduceTensorDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnRNNDescriptor_t>
            {
                static void create(cudnnRNNDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateRNNDescriptor(&desc));
                }
                static void destroy(cudnnRNNDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyRNNDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnPoolingDescriptor_t>
            {
                static void create(cudnnPoolingDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreatePoolingDescriptor(&desc));
                }
                static void destroy(cudnnPoolingDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyPoolingDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnOpTensorDescriptor_t>
            {
                static void create(cudnnOpTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateOpTensorDescriptor(&desc));
                }
                static void destroy(cudnnOpTensorDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyOpTensorDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnFilterDescriptor_t>
            {
                static void create(cudnnFilterDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&desc));
                }
                static void destroy(cudnnFilterDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyFilterDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnDropoutDescriptor_t>
            {
                static void create(cudnnDropoutDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateDropoutDescriptor(&desc));
                }
                static void destroy(cudnnDropoutDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyDropoutDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnConvolutionDescriptor_t>
            {
                static void create(cudnnConvolutionDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&desc));
                }
                static void destroy(cudnnConvolutionDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyConvolutionDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnCTCLossDescriptor_t>
            {
                static void create(cudnnCTCLossDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateCTCLossDescriptor(&desc));
                }
                static void destroy(cudnnCTCLossDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyCTCLossDescriptor(desc));
                }
            };

            template <>
            struct cudnn_descriptor<cudnnActivationDescriptor_t>
            {
                static void create(cudnnActivationDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL(cudnnCreateActivationDescriptor(&desc));
                }
                static void destroy(cudnnActivationDescriptor_t& desc)
                {
                    CUDNN_SAFE_CALL_NO_THROW(cudnnDestroyActivationDescriptor(desc));
                }
            };
        }
    }
}
