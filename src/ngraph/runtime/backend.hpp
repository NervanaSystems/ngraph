//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/deprecated.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class ExternalFunction;
        class Tensor;
        class Backend;
        using Handle = void*;
    }
}

/// \brief Interface to a generic backend.
///
/// Backends are responsible for function execution and value allocation.
class ngraph::runtime::Backend
{
public:
    virtual ~Backend();

    /// \brief Create a new Backend object
    /// \param type The name of a registered backend, such as "CPU" or "GPU".
    ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
    /// \returns unique_ptr to a new Backend or nullptr if the named backend
    ///   does not exist.
    static std::unique_ptr<Backend> create(const std::string& type);

    /// \brief Query the list of registered devices
    /// \returns A vector of all registered devices.
    static std::vector<std::string> get_registered_devices();

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor> create_tensor(
        const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer) = 0;

    /// \brief Create a tensor of C type T specific to this backend
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend specific tensor
    template <typename T>
    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const Shape& shape)
    {
        return create_tensor(element::from<T>(), shape);
    }

    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \param enable_performance_collection Flag to add per-op performance profiling.
    /// \returns Handle or nullptr on failure
    virtual Handle compile(std::shared_ptr<Function> func,
                           bool enable_performance_collection = false) = 0;

    /// \deprecated use execute method
    /// \brief Executes a single iteration of a Function.
    /// \param handle The Handle returned from compile or load
    /// \returns true if iteration is successful, false otherwise
    DEPRECATED virtual bool call(Handle handle,
                                 const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                 const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Executes a single iteration of a Function.
    /// \param handle The Handle returned from compile or load
    /// \param outputs vector of runtime::Tensor used as outputs
    /// \param inputs vector of runtime::Tensor used as inputs
    /// \returns true if iteration is successful, false otherwise
    virtual bool execute(Handle handle,
                         const std::vector<runtime::Tensor*>& outputs,
                         const std::vector<runtime::Tensor*>& inputs) = 0;

    /// \brief Executes a single iteration of a Function.
    /// \param handle The Handle returned from compile or load
    /// \returns true if iteration is successful, false otherwise
    DEPRECATED bool call_with_validate(Handle handle,
                                       const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                       const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

    /// \brief Validates and then Executes a single iteration of a Function.
    /// \param handle The Handle returned from compile or load
    /// \returns true if iteration is successful, false otherwise
    bool validate_and_execute(Handle handle,
                              const std::vector<runtime::Tensor*>& outputs,
                              const std::vector<runtime::Tensor*>& inputs)
    {
        validate(handle, outputs, inputs);
        return execute(handle, outputs, inputs);
    }

    /// \brief Compiled functions may be cached. This function removes a compiled function
    ///     from the cache.
    /// \param handle The Handle returned from compile or load
    virtual void remove_compiled_function(Handle handle);

    /// \brief Collect performance information gathered on a Function.
    /// \param handle The Handle returned from compile or load
    /// \returns Vector of PerformanceCounter information.
    virtual std::vector<PerformanceCounter> get_performance_data(Handle handle) const;

    /// \brief Test if a backend is capable of supporting an op
    /// \param node is the op to test.
    /// \returns true if the op is supported, false otherwise.
    virtual bool is_supported(const Node& node) const;

    void validate(Handle handle,
                  const std::vector<runtime::Tensor*>& outputs,
                  const std::vector<runtime::Tensor*>& inputs);

    /// \brief Query the input Parameters for a given Handle
    /// \param handle The Handle returned from compile or load
    /// \returns an ngraph::op::ParameterVector of all input parameters
    const ngraph::ParameterVector& get_parameters(Handle handle) const;

    /// \brief Query the output Results for a given Handle
    /// \param handle The Handle returned from compile or load
    /// \returns an ngraph::ResultVector of all input parameters
    const ngraph::ResultVector& get_results(Handle handle) const;

protected:
    /// \brief Called at the end of compile to the the values to be returned by get_parameters
    ///     and get_results
    /// \param func The function with Results fully resolved.
    void set_parameters_and_results(Handle handle, const Function& func);

private:
    std::unordered_map<Handle, ngraph::ParameterVector> m_parameters;
    std::unordered_map<Handle, ngraph::ResultVector> m_results;
};
