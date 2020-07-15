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
#include "ngraph/runtime/host_tensor.hpp"
#ifdef INTERPRETER_USE_HYBRID
#include "ngraph/runtime/hybrid/op/function_call.hpp"
#endif
#include "ngraph/runtime/mlir/mlir_backend_visibility.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/allreduce.hpp"
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
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/runtime/reference/cosh.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"
#include "ngraph/runtime/reference/erf.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "ngraph/runtime/reference/generate_mask.hpp"
#include "ngraph/runtime/reference/log.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/matmul.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/mod.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/random_uniform.hpp"
#include "ngraph/runtime/reference/recv.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/round.hpp"
#include "ngraph/runtime/reference/scatter_add.hpp"
#include "ngraph/runtime/reference/scatter_nd_add.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/send.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sin.hpp"
#include "ngraph/runtime/reference/sinh.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/tan.hpp"
#include "ngraph/runtime/reference/tanh.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/bernoulli_rng_state.hpp"
#include "ngraph/state/uniform_rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace mlir
        {
            class MlirBackend;
            class MlirExecutable;

            // This expands the op list in op_tbl.hpp into a list of enumerations that look like
            // this:
            // Abs,
            // Acos,
            // ...
            enum class OP_TYPEID
            {
#define NGRAPH_OP(NAME, NAMESPACE) ID_SUFFIX(NAME),
#include "ngraph/runtime/mlir/opset_mlir_tbl.hpp"
#undef NGRAPH_OP
                UnknownOp
            };
        }
    }
}

class MLIR_BACKEND_API ngraph::runtime::mlir::MlirExecutable : public Executable
{
    friend class MlirBackend;

public:
    MlirExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index) override;

    std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_input_tensor(size_t input_index, size_t pipeline_depth) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_output_tensor(size_t output_index, size_t pipeline_depth) override;

protected:
    MlirExecutable(const std::string& model_string);

    std::shared_ptr<ngraph::op::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ngraph::op::Result> get_result(size_t index) const;
    int get_alignment() const { return 64; }
    std::shared_ptr<Function> m_function;
    NodeVector m_nodes;
    std::unordered_map<const Node*, std::shared_ptr<State>> m_states;

    static OP_TYPEID get_typeid(const Node& node);
};
