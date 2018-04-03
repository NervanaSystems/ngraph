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

#include <string>

#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPUTensorView : public ngraph::runtime::TensorView
            {
            public:
                CPUTensorView(const ngraph::element::Type& element_type,
                              const Shape& shape,
                              const std::string& name = "external");
                virtual ~CPUTensorView() override;

                char* get_data_ptr();
                const char* get_data_ptr() const;

                size_t get_size() const;
                const element::Type& get_element_type() const;

                /// @brief Write bytes directly into the tensor
                /// @param p Pointer to source of data
                /// @param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
                /// @param n Number of bytes to write, must be integral number of elements.
                void write(const void* p, size_t tensor_offset, size_t n) override;

                /// @brief Read bytes directly from the tensor
                /// @param p Pointer to destination for data
                /// @param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
                /// @param n Number of bytes to read, must be integral number of elements.
                void read(void* p, size_t tensor_offset, size_t n) const override;

            private:
                static const size_t BufferAlignment;

                char* buffer;
                char* aligned_buffer;
                size_t buffer_size;
            };
        }
    }
}
