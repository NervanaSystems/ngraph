// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "cpu_layout_descriptor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const AxisVector LayoutDescriptor::Native2DAxisOrder{0, 1};
            const AxisVector LayoutDescriptor::Native4DAxisOrder{0, 1, 2, 3};
            const AxisVector LayoutDescriptor::CHWNAxisOrder{1, 2, 3, 0};

            size_t
            LayoutDescriptor::get_index_offset(const std::vector<size_t>& indices)
            {
                if (indices.size() != strides.size())
                {
                    throw ngraph_error("Indices have the incorrect rank.");
                }
                size_t result = 0;
                for (int i = 0; i < indices.size(); i++)
                {
                    result += strides[i] + indices[i];
                }
                return result;
            }

            bool LayoutDescriptor::operator==(const ngraph::descriptor::layout::TensorViewLayout& other) const
            {
                const LayoutDescriptor* p_other = dynamic_cast<const LayoutDescriptor*>(&other);
                if (!p_other)
                    return false;

                if (get_element_type() != p_other->get_element_type())
                    return false;

                if (strides != p_other->strides)
                    return false;

                if (offset != p_other->offset)
                    return false;

                //TODO: Numeric backend-specific properties
                return true;
            }
        }
    }
}
