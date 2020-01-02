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

#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/generic_cpu/kernel/broadcast.hpp"
#include "ngraph/runtime/generic_cpu/kernel/dot.hpp"
#include "ngraph/runtime/generic_cpu/kernel/reshape.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/interpreter/int_executable.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/allreduce.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/argmax.hpp"
#include "ngraph/runtime/reference/argmin.hpp"
#include "ngraph/runtime/reference/asin.hpp"
#include "ngraph/runtime/reference/atan.hpp"
#include "ngraph/runtime/reference/atan2.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_mat_mul.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/broadcast_distributed.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/constant.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/runtime/reference/cosh.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/erf.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "ngraph/runtime/reference/generate_mask.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/log.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/power.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/recv.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/scatter_add.hpp"
#include "ngraph/runtime/reference/scatter_nd_add.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/send.hpp"
#include "ngraph/runtime/reference/shape_of.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sin.hpp"
#include "ngraph/runtime/reference/sinh.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/tan.hpp"
#include "ngraph/runtime/reference/tanh.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/runtime/reference/xor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/bernoulli_rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
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
    }
}

class ngraph::runtime::gcpu::GCPUExecutable : public runtime::interpreter::INTExecutable
{
    friend class GCPUBackend;

public:
    GCPUExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& intputs) override;

private:
    int get_alignment() const { return 64; }

    void generate_calls(const element::Type& type,
                        const Node& op,
                        const std::vector<std::shared_ptr<HostTensor>>& outputs,
                        const std::vector<std::shared_ptr<HostTensor>>& inputs) override;

    template <typename T>
    void gop_engine(const Node& node,
                   const std::vector<std::shared_ptr<HostTensor>>& out,
                   const std::vector<std::shared_ptr<HostTensor>>& args)
    {
        switch (INTExecutable::get_typeid(node))
        {
        case ngraph::runtime::interpreter::OP_TYPEID::Broadcast:
        {
            const op::Broadcast* broadcast = static_cast<const op::Broadcast*>(&node);
            Shape in_shape = node.get_input_shape(0);
            Shape out_shape = node.get_output_shape(0);
            AxisSet broadcast_axes = broadcast->get_broadcast_axes();
            reference::broadcast<T>(args[0]->get_data_ptr<const T>(),
                                    out[0]->get_data_ptr<T>(),
                                    in_shape,
                                    out_shape,
                                    broadcast_axes);
            break;
        }
        case ngraph::runtime::interpreter::OP_TYPEID::Reshape:
        {
            const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
            reference::reshape(args[0]->get_data_ptr<const T>(),
                               out[0]->get_data_ptr<T>(),
                               node.get_input_shape(0),
                               reshape->get_input_order(),
                               node.get_output_shape(0));
            break;
        }
        default: op_engine<T>(node, out, args); break;
        }
    }
};
