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

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "interpreter/int_executable.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/broadcast.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace gcpu
        {
            class GCPUBackend;
            class GCPUExecutable;

            namespace
            {
                // This expands the op list in op_tbl.hpp into a list of enumerations that look like
                // this:
                // Abs,
                // Acos,
                // ...
                enum class OP_TYPEID
                {
#define NGRAPH_OP(NAME, NAMESPACE) NAME,
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
                    UnknownOp
                };
            }
        }

class gcpu::GCPUExecutable : public interpreter::INTExecutable
{
    friend class GCPUBackend;

public:
    GCPUExecutable(const std::shared_ptr<ngraph::Function>& function,
                   bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& intputs) override;

private:
    int get_alignment() const { return 64; }
    void generate_calls(const ngraph::element::Type& type,
                        const ngraph::Node& op,
                        const std::vector<std::shared_ptr<ngraph::runtime::HostTensor>>& outputs,
                        const std::vector<std::shared_ptr<ngraph::runtime::HostTensor>>& inputs) override;

    template <typename T>
    void gop_engine(const ngraph::Node& node,
                    const std::vector<std::shared_ptr<ngraph::runtime::HostTensor>>& out,
                    const std::vector<std::shared_ptr<ngraph::runtime::HostTensor>>& args)
    {
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
        switch (INTExecutable::get_typeid(node))
        {
        case interpreter::OP_TYPEID::Broadcast:
        {
            const ngraph::op::Broadcast* broadcast = static_cast<const ngraph::op::Broadcast*>(&node);
            ngraph::Shape in_shape = node.get_input_shape(0);
            ngraph::Shape out_shape = node.get_output_shape(0);
            ngraph::AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            ngraph::runtime::reference::broadcast<T>(args[0]->get_data_ptr<const T>(),
                                    out[0]->get_data_ptr<T>(),
                                    in_shape,
                                    out_shape,
                                    broadcast_axes);
            break;
        }
        case interpreter::OP_TYPEID::Reshape:
        {
            const ngraph::op::Reshape* reshape = static_cast<const ngraph::op::Reshape*>(&node);
            ngraph::runtime::reference::reshape(args[0]->get_data_ptr<const T>(),
                               out[0]->get_data_ptr<T>(),
                               node.get_input_shape(0),
                               reshape->get_input_order(),
                               node.get_output_shape(0));
            break;
        }
        default: op_engine<T>(node, out, args); break;
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
    }
};
