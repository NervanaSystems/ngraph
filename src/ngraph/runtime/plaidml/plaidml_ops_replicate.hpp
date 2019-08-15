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

#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace op
            {
                class Replicate;
            }
        }
    }
}

// Replicate works like Concat, but only over identical inputs.  This
// restriction allows it to be substantially more efficient.
class ngraph::runtime::plaidml::op::Replicate final : public ngraph::op::Op
{
public:
    static const std::string type_name;
    const std::string& description() const override { return type_name; }
    Replicate(const Output<Node>& arg, std::size_t replication_axis, std::size_t replication_count);

    Replicate(const Output<Node>& arg, std::vector<std::size_t> replication_axes);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    /// \return The replication axes: axis index -> the replication count along that axis.
    const std::vector<std::size_t>& get_replication_axes() const { return m_replication_axes; }
private:
    std::vector<std::size_t> m_replication_axes;
};
