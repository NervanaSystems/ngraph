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

#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        //brief Computes indices of top k maximum/minimum index along a specified axis for a given tensor
        class TopK : public Op
        {
        public:
            enum class SortType
            {
                // Returned values are not sorted
                NONE,
                // Sort result based on element indices
                SORT_INDICES,
                // Sort result based on element values
                SORT_VALUES,
            };

            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a TopK operation
            TopK() = default;
            /// \brief Constructs a TopK operation.
            ///
            /// \param arg The input tensor
            /// \param top_k_axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param compute_max Compute top k max or top k min?
            /// \param sort SortType for sorting results, default - NONE
            TopK(const Output<Node>& arg,
                 size_t top_k_axis,
                 const element::Type& index_element_type,
                 size_t k = 0,
                 bool compute_max = true,
                 SortType sort = SortType::NONE);
            /// \brief Constructs a TopK operation.
            ///
            /// \param arg The input tensor
            /// \param k Number of top indices to compute. Compute all indices if k = 0
            /// \param top_k_axis The axis along which to compute top k indices
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            /// \param compute_max Compute top k max or top k min?
            /// \param sort SortType for sorting results, default - NONE
            TopK(const Output<Node>& arg,
                 const Output<Node>& k,
                 size_t top_k_axis,
                 const element::Type& index_element_type,
                 bool compute_max = true,
                 SortType sort = SortType::NONE);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_k() const;
            void set_k(size_t k);

            size_t get_top_k_axis() const { return m_top_k_axis; }
            element::Type get_index_element_type() const { return m_index_element_type; }
            bool get_compute_max() const { return m_compute_max; }
            SortType get_sort() const { return m_sort; }
        protected:
            size_t m_top_k_axis{0};
            element::Type m_index_element_type;
            bool m_compute_max{false};
            SortType m_sort;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
