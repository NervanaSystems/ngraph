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
        namespace v0
        {
            /// \brief Dynamic Tiling operation which repeats a tensor multiple times
            ///        along each dimension
            class NGRAPH_API Tile : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Tile", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Tile() = default;
                /// \brief Perform dynamic padding of a tensor
                ///
                /// \param data The node producing input tensor to be padded.
                /// \param repeats The node producing the per-dimension replication factor
                Tile(const Output<Node>& data, const Output<Node>& repeats);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
        using v0::Tile;
    }
}
