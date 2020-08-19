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

#include "contrib/mlir/runtime/cpu/cpu_runtime.hpp"
#include "contrib/mlir/runtime/runtime.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/mlir/mlir_backend_visibility.hpp"
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
#define NGRAPH_OP(NAME, VERSION) NAME##_v##VERSION,
#include "ngraph/op_version_tbl.hpp"
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
    std::shared_ptr<ngraph::op::v0::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ngraph::op::v0::Result> get_result(size_t index) const;
    int get_alignment() const { return 64; }
    std::shared_ptr<Function> m_function;
    NodeVector m_nodes;
    std::unordered_map<const Node*, std::shared_ptr<State>> m_states;
    runtime::ngmlir::MLIRCPURuntime m_mlir_runtime;
    bool m_first_iteration = true;

    static OP_TYPEID get_typeid(const Node& node);
};
