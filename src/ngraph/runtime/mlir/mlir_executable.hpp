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

#include <memory>
#include <string>
#include <vector>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/mlir/mlir_backend_visibility.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace mlir
        {
            class MlirBackend;
            class MlirExecutable;
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
    std::shared_ptr<ngraph::op::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ngraph::op::Result> get_result(size_t index) const;
    void init();
    llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
        create_default_target_machine(unsigned optLevel);

    int get_alignment() const { return 64; }
    std::shared_ptr<Function> m_function;
    NodeVector m_nodes;
    bool m_first_iteration = true;
    std::unique_ptr<::mlir::ExecutionEngine> m_engine;
};
