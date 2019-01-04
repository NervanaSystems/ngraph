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
#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/parameter_vector.hpp"
#include "ngraph/result_vector.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Executable;
        class Tensor;
    }
}

class ngraph::runtime::Executable
{
public:
    Executable(runtime::Backend* backend);
    virtual ~Executable();

    /// \deprecated use execute method
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    DEPRECATED virtual bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                 const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Executes a single iteration of a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    virtual bool execute(const std::vector<runtime::Tensor*>& outputs,
                         const std::vector<runtime::Tensor*>& inputs) = 0;

    /// \brief Executes a single iteration of a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    DEPRECATED bool call_with_validate(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                       const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Validates and then Executes a single iteration of a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    bool validate_and_execute(const std::vector<runtime::Tensor*>& outputs,
                              const std::vector<runtime::Tensor*>& inputs);

    /// \brief Collect performance information gathered on a Function.
    /// \returns Vector of PerformanceCounter information.
    virtual std::vector<PerformanceCounter> get_performance_data() const;

    /// \brief Validates a Function.
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    void validate(const std::vector<runtime::Tensor*>& outputs,
                  const std::vector<runtime::Tensor*>& inputs);

    /// \brief Query the input Parameters for a given Handle
    /// \returns an ngraph::op::ParameterVector of all input parameters
    const ngraph::ParameterVector& get_parameters() const;

    /// \brief Query the output Results for a given Handle
    /// \returns an ngraph::ResultVector of all input parameters
    const ngraph::ResultVector& get_results() const;

protected:
    /// \brief Called at the end of compile to the the values to be returned by get_parameters
    ///     and get_results
    /// \param func The function with Results fully resolved.
    void set_parameters_and_results(const Function& func);

    runtime::Backend* get_backend();

private:
    ngraph::ParameterVector m_parameters;
    ngraph::ResultVector m_results;
    runtime::Backend* m_backend;
};
