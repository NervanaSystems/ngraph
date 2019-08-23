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
#include <mutex>

#include "ngraph/function.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Tensor;
        class Backend;
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
    /// \param must_support_dynamic If `true`, the returned `Backend` object
    ///    will support dynamic tensors. If the underlying backend has native
    ///    support for dynamic tensors, then that backend object will be
    ///    returned directly. Otherwise, it will be wrapped with
    ///    DynamicWrapperBackend. This feature is EXPERIMENTAL.
    /// \returns shared_ptr to a new Backend or nullptr if the named backend
    ///   does not exist.
    static std::shared_ptr<Backend> create(const std::string& type,
                                           bool must_support_dynamic = false);

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

    /// \brief Create a dynamic tensor specific to this backend, if the backend supports dynamic
    ///        tensors.
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    /// \throws std::invalid_argument if the backend does not support dynamic tensors
    virtual std::shared_ptr<ngraph::runtime::Tensor>
        create_dynamic_tensor(const ngraph::element::Type& element_type, const PartialShape& shape);

    /// \returns `true` if this backend supports dynamic tensors, else `false`.
    virtual bool supports_dynamic_tensors() { return false; }
    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \returns compiled function or nullptr on failure
    virtual std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                                bool enable_performance_data = false) = 0;

    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \param pass_config Configuration object for defining compilation options
    /// \returns compiled function or nullptr on failure
    virtual std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                                ngraph::pass::PassConfig& pass_config,
                                                bool enable_performance_data = false);

    /// \brief Loads a previously saved Executable object from a stream.
    /// \param input_stream the opened input stream containing the saved Executable
    /// \returns A compiled function or throws an exception on error
    virtual std::shared_ptr<Executable> load(std::istream& input_stream);

    /// \brief Test if a backend is capable of supporting an op
    /// \param node is the op to test.
    /// \returns true if the op is supported, false otherwise.
    virtual bool is_supported(const Node& node) const;

    /// \brief A set of properties supported by a backend
    enum class Property
    {
        memory_attach /// New tensor can use attached memory
    };

    /// \brief Test if a backend particular property is supported
    /// \param prop is the feature to test.
    /// \returns true if the property is supported, false otherwise.
    virtual bool is_supported_property(const Property prop) const;

    virtual void remove_compiled_function(std::shared_ptr<Executable> exec);

    /// \brief Return a backend specific op (that is not a core ngraph op).
    ///     The string op_name is the requested op, which a backend may or may not implement.
    ///     If unsupported, nullptr is returned, else a backend op is returned.
    ///     The variadic input is used to pass inputs that the op constructor might take
    /// \param op_name is the name of the backend specific op
    /// \returns a shared pointer to the op if found, else nullptr
    virtual std::shared_ptr<ngraph::Node> get_backend_op(const std::string& op_name, ...);

    /// \brief Allows sending backend specific configuration. The map contains key, value pairs
    ///     specific to a particluar backend. The definition of these key, value pairs is
    ///     defined by each backend.
    /// \param config The configuration map sent to the backend
    /// \param error An error string describing any error encountered
    /// \returns true if the configuration is supported, false otherwise. On false the error
    ///     parameter value is valid.
    virtual bool set_config(const std::map<std::string, std::string>& config, std::string& error);

    static void set_backend_shared_library_search_directory(const std::string& path);
    static const std::string& get_backend_shared_library_search_directory();

    /// \brief Returns memory allocator used by backend for host allocations
    virtual Allocator* get_host_memory_allocator() { return nullptr; }
    /// \brief Set the host memory allocator to be used by the backend
    /// \param allocator is pointer to host memory allocator object
    virtual void set_host_memory_allocator(Allocator* allocator) {}
    /// \brief Returns memory allocator used by backend for device allocations
    virtual Allocator* get_device_memory_allocator()
    {
        // override this method from each supported backend to return
        // its own device memory allocator
        return nullptr;
    }

    /// \brief method for each supported backend to determine if the passed pointer is in device
    ///        pinned memory or not
    /// \param ptr pointer to the memory to determine if its in device memory or not
    virtual bool is_device_memory(void* ptr);

    virtual bool executable_can_create_tensors();

    /// \brief Get the version of the backend
    /// The default value of 0.0.0 is chosen to be a parsable version number
    virtual std::string get_version() const { return "0.0.0"; }
private:
    // mutex to modify s_backend_shared_library_search_directory thread safe
    static std::mutex m_mtx;
    static std::string s_backend_shared_library_search_directory;
};
