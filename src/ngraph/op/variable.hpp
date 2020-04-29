//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace v3
    {
        class NGRAPH_API Variable
        {
        public:
            Variable() = default;

            Variable(const PartialShape& data_shape,
                    const element::Type& data_type,
                      std::string variable_id)
                      :
                      m_data_shape(data_shape),
                      m_data_type(data_type),
                      m_variable_id(std::move(variable_id))
            {
            }

            PartialShape get_shape() { return m_data_shape; }
            element::Type get_type() { return m_data_type; }
            std::string get_id() { return m_variable_id; }

            void update(const PartialShape& data_shape,
                        const element::Type& data_type,
                        std::string variable_id)
            {
                m_data_shape = data_shape;
                m_data_type = data_type;
                m_variable_id = std::move(variable_id);
            }

        private:
            PartialShape m_data_shape;
            element::Type m_data_type;
            std::string m_variable_id;
        };
    }
}
