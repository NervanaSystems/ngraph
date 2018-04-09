/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class ExternalFunction;
        class CallFrame;
        class TensorView;

        /// @brief Interface to a generic backend.
        ///
        /// Backends are responsible for function execution and value allocation.
        class Backend
        {
        public:
            virtual ~Backend() {}
            /// @brief Make a call frame that can support one concurrent call of an external function.
            ///
            /// If more than one concurrent execution is needed, each execution will require its own call frame.
            /// DEPRECATED
            virtual std::shared_ptr<ngraph::runtime::CallFrame>
                make_call_frame(const std::shared_ptr<ExternalFunction>& external_function) = 0;

            /// @brief Return a handle for a tensor on the backend device.
            /// DEPRECATED
            virtual std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const ngraph::element::Type& element_type,
                                         const Shape& shape) = 0;

            /// DEPRECATED
            template <typename T>
            std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const Shape& shape)
            {
                return make_primary_tensor_view(element::from<T>(), shape);
            }

            /// @brief Return a handle for a tensor for given mem on backend device
            virtual std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const ngraph::element::Type& element_type,
                                         const Shape& shape,
                                         void* memory_pointer) = 0;

            /// DEPRECATED
            virtual bool call(const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                              const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) = 0;

            /// @brief Create a new Backend object
            /// @param type The name of a registered backend, such as "CPU" or "GPU".
            ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
            /// @returns shared_ptr to a new Backend or nullptr if the named backend
            ///   does not exist.
            static std::shared_ptr<Backend> create(const std::string& type);

            /// @brief Query the list of registered devices
            /// @returns A vector of all registered devices.
            static std::vector<std::string> get_registered_devices();

            /// @brief Query the list of available subdevices of a particular device.
            /// @param type The name of a registered backend, such as "CPU" or "GPU"
            /// @returns A vector of available devices of the specified type.
            static std::vector<size_t> get_subdevices(const std::string& type);

            virtual std::shared_ptr<ngraph::runtime::TensorView>
                create_tensor(const ngraph::element::Type& element_type, const Shape& shape) = 0;

            template <typename T>
            std::shared_ptr<ngraph::runtime::TensorView> create_tensor(const Shape& shape)
            {
                return create_tensor(element::from<T>(), shape);
            }

            virtual bool compile(const ngraph::Function& func) = 0;

            virtual bool call(const ngraph::Function& func,
                              const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                              const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) = 0;

            virtual void remove_compiled_function(const ngraph::Function& func);
        };
    }
}
