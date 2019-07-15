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

#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "ngraph/runtime/plaidml/plaidml_build.hpp"
#include "ngraph/runtime/plaidml/plaidml_builder.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // PlaidML Operation implementation support.
            //
            // To add a new operation:
            //  1) Include the operation header (this file)
            //  2) Use NGRAPH_PLAIDML_OP_CLASS() to instantiate the
            //     operation classes (or roll your own, based on the
            //     macro).
            //  3) Write the per-operation implementation definition

            // The OpImpl template is generally used as the base class
            // for operation implementations.
            template <class O>
            class OpImpl
            {
            public:
                using Op = O;

                virtual ~OpImpl() {}
                // Apply the operation implementation to the current
                // build.
                //
                // set_build() and set_op() must be called before
                // applying the operation implementation.  (Typically,
                // OpImplDescriptor is used to handle this.)
                virtual void Apply() = 0;

                Build* build() const { return m_build; }
                void set_build(Build* build) { m_build = build; }
                const Op& op() const { return *m_op; }
                void set_op(const Op* op) { m_op = op; }
                // Returns the indicated operation input as a PlaidML variable.
                vertexai::plaidml::variable op_input(std::size_t idx) const
                {
                    const auto& ti =
                        m_build->bindings.at(&op().input(idx).get_source_output().get_tensor());
                    return ti.var;
                }

                // Returns the 0th operation input as a PlaidML variable.
                vertexai::plaidml::variable op_input() const { return op_input(0); }
                // Validates that the number of operation inputs matches the expected operation
                // input count.
                void check_inputs(std::size_t expected_input_count) const
                {
                    if (op().get_input_size() < expected_input_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << op().description()
                           << " operations with an input count >= " << expected_input_count
                           << " (got " << op().get_input_size() << " inputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Validates that the number of operation inputs is greater than or equal to the
                // expected operation input count.
                void check_inputs_ge(std::size_t minimum_input_count) const
                {
                    if (op().get_input_size() < minimum_input_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << op().description()
                           << " operations with an input count >= " << minimum_input_count
                           << " (got " << op().get_input_size() << " inputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Validates that the number of operation outputs matches the expected operation
                // output count.
                void check_outputs(std::size_t expected_output_count) const
                {
                    if (op().get_output_size() != expected_output_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << op().description()
                           << " operations with an output count == " << expected_output_count
                           << " (got " << op().get_output_size() << " outputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Binds the indicated operation output to the supplied PlaidML variable.
                void set_output(std::size_t idx,
                                vertexai::plaidml::variable var,
                                TensorContents contents = TensorContents::DATA)
                {
                    m_build->bindings.emplace(op().get_output_tensor_ptr(idx).get(),
                                              TensorInfo{std::move(var), contents});
                }

                // Binds the 0th operation output to the supplied PlaidML variable.
                void set_output(vertexai::plaidml::variable var,
                                TensorContents contents = TensorContents::DATA)
                {
                    m_build->bindings.emplace(op().get_output_tensor_ptr().get(),
                                              TensorInfo{std::move(var), contents});
                }

                // Gets a useful name for the current op.
                std::string get_op_name() const { return op().description(); }
                // Starts a Tile function builder.
                builder::Function start_tile_function(bool debug = false) const
                {
                    return builder::Function{get_op_name(), debug || m_build->config->debug};
                }

            private:
                Build* m_build = nullptr;
                const Op* m_op = nullptr;
            };

            // OpImplDescriptorBase is the non-template virtual base
            // class for OpImplDescriptor.  It solely exists to lever
            // up to a typed OpImplDescriptor in a safe way.
            class OpImplDescriptorBase
            {
            public:
                virtual ~OpImplDescriptorBase() {}
                virtual void Apply(Build* build, const ngraph::Node* op) = 0;
            };

            // OpImplDescriptor describes an operation implementation class,
            // and can be used to apply the implementation to a build in progress.
            template <class OI>
            class OpImplDescriptor final : public OpImplDescriptorBase
            {
            public:
                using Impl = OI;

                void Apply(Build* build, const ngraph::Node* op) final
                {
                    Impl impl;
                    impl.set_build(build);
                    impl.set_op(static_cast<const typename Impl::Op*>(op));
                    impl.Apply();
                }
            };

            using OpImplMap =
                std::unordered_map<std::type_index, std::unique_ptr<OpImplDescriptorBase>>;

            // The accessor for the global operation handler map.
            OpImplMap* GlobalOpImplMap();

            // OpImplRegistration handles the registration of a particular operation implementation.
            // To use it, instantiate a variable of type OpImplRegistration<OpImplDescriptor> at
            // global scope.
            template <class OID>
            class OpImplRegistration final
            {
            public:
                OpImplRegistration()
                {
                    GlobalOpImplMap()->emplace(std::type_index{typeid(typename OID::Impl::Op)},
                                               std::unique_ptr<OpImplDescriptorBase>{new OID()});
                }
            };
        }
    }
}

// A macro to make the usual case of declaring operations a little simpler.
#define NGRAPH_PLAIDML_OP_CLASS(_Impl, _Parent)                                                    \
    class _Impl final : public _Parent                                                             \
    {                                                                                              \
    public:                                                                                        \
        void Apply() final;                                                                        \
    };                                                                                             \
                                                                                                   \
    namespace                                                                                      \
    {                                                                                              \
        OpImplRegistration<OpImplDescriptor<_Impl>> register_##_Impl;                              \
    }
