//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        class RegionYolo : public Op
        {
        public:
            /// \brief Constructs a RegionYolo operation
            ///
            /// \param input          Input
            /// \param num_coords     Number of coordinates for each region
            /// \param num_classes    Number of classes for each region
            /// \param num_regions    Number of regions
            /// \param do_softmax     Compute softmax
            /// \param mask           Mask
            /// \param axis           Axis to begin softmax on
            /// \param end_axis       Axis to end softmax on
            RegionYolo(const std::shared_ptr<Node>& input,
                       const size_t num_coords,
                       const size_t num_classes,
                       const size_t num_regions,
                       const bool do_softmax,
                       const std::vector<int64_t>& mask,
                       const int axis,
                       const int end_axis);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_num_coords() const { return m_num_coords; }
            size_t get_num_classes() const { return m_num_classes; }
            size_t get_num_regions() const { return m_num_regions; }
            bool get_do_softmax() const { return m_do_softmax; }
            const std::vector<int64_t>& get_mask() const { return m_mask; }
            int get_axis() const { return m_axis; }
            int get_end_axis() const { return m_end_axis; }
        private:
            size_t m_num_coords;
            size_t m_num_classes;
            size_t m_num_regions;
            bool m_do_softmax;
            std::vector<int64_t> m_mask;
            int m_axis;
            int m_end_axis;
        };
    }
}
