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
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Dropout : public Op
        {
        public:
            static constexpr NodeTypeInfo type_info{"Dropout", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            Dropout(const Output<Node>& input,
                    const Output<Node>& gm_const,
                    const Output<Node>& use_seed,
                    const Output<Node>& seed,
                    const Output<Node>& keep_prob); // keep_prob = 1 - dropout_prob

            bool get_use_seed() const;
            uint64_t get_seed() const;
            double get_keep_prob() const;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
