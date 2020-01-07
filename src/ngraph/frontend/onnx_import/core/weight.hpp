//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief Weight for an input
        class Weight
        {
        public:
            Weight(const Weight&) = default;
            Weight& operator=(const Weight&) = delete;

            Weight() = delete;

            Weight(Weight&&) = default;
            Weight& operator=(Weight&&) = delete;

            Weight(const element::Type& type, const Shape& shape, std::vector<char> data)
                : m_shape{shape}
                , m_type{type}
                , m_data{std::move(data)}
            {
                for (const auto& value : m_shape)
                {
                    m_size *= value;
                }
            }

            const Shape& shape() const { return m_shape; }
            std::size_t size() const { return m_size; }
            const element::Type& type() const { return m_type; }
            std::shared_ptr<runtime::Tensor> to_tensor(runtime::Backend& backend)
            {
                return backend.create_tensor(
                    m_type, m_shape, reinterpret_cast<void*>(m_data.data()));
            }

            const void* data() const { return reinterpret_cast<const void*>(m_data.data()); }
        private:
            Shape m_shape{};
            const element::Type& m_type;
            std::size_t m_size{1};
            std::vector<char> m_data{};
        };

        using Weights = std::unordered_map<std::string, Weight>;
    }
}
