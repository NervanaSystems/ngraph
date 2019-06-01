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

#include "ngraph/function.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Tensor;
        class Executable;
        class Backend;
    }
}

class ngraph::runtime::Executable
{
public:
    Executable(const std::shared_ptr<runtime::Backend>& backend);
    virtual ~Executable();

    /// \brief Create a tensor specific to this backend
    /// \param index The position of the input tensor in the inputs
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<runtime::Tensor> create_input_tensor(size_t index,
                                                                 void* memory_pointer = nullptr);
    virtual std::shared_ptr<runtime::Tensor> create_output_tensor(size_t index,
                                                                  void* memory_pointer = nullptr);
    virtual std::shared_ptr<runtime::Tensor>
        create_parameter_tensor(const op::Parameter& parameter);
    virtual std::shared_ptr<runtime::Tensor> create_result_tensor(const Node& result);
    std::shared_ptr<runtime::Tensor>
        create_parameter_tensor(const std::shared_ptr<op::Parameter>& parameter);
    std::shared_ptr<runtime::Tensor> create_result_tensor(const std::shared_ptr<Node>& result);

    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    virtual bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) = 0;

    /// \brief Executes a single iteration of a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    bool call_with_validate(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                            const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Collect performance information gathered on a Function.
    /// \returns Vector of PerformanceCounter information.
    virtual std::vector<PerformanceCounter> get_performance_data() const;

    /// \brief Validates a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    void validate(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                  const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Query the input Parameters
    /// \returns an ngraph::op::ParameterVector of all input parameters
    const ngraph::ParameterVector& get_parameters() const;

    /// \brief Query the output Results
    /// \returns an ngraph::ResultVector of all input parameters
    const ngraph::ResultVector& get_results() const;

    /// \brief Save this compiled Executable to an output stream.
    ///    Saved stream may be read with Backend::load
    virtual void save(std::ostream& output_stream);

protected:
    /// \brief Called at the end of compile to the values to be returned by get_parameters
    ///     and get_results
    /// \param func The function with Results fully resolved.
    void set_parameters_and_results(const Function& func);

    ngraph::ParameterVector m_parameters;
    ngraph::ResultVector m_results;
    std::shared_ptr<runtime::Backend> m_backend;
};
