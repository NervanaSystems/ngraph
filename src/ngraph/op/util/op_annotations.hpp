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

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for annotations added to graph ops
            class OpAnnotations
            {
            public:
                void set_in_place_oi_pairs(const std::map<size_t, size_t>& oi_pairs)
                {
                    m_in_place_oi_pairs = oi_pairs;
                }

                const std::map<size_t, size_t>& get_in_place_oi_pairs() const
                {
                    return m_in_place_oi_pairs;
                }

            private:
                std::map<size_t, size_t> m_in_place_oi_pairs;
            };
        }
    }
}
