// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "element_type.hpp"

namespace ngraph
{
    class ExecutionState;

    class Op;
    // class TensorDescription;
    class ComputationOp;

    using computation_op_ptr = std::shared_ptr<ComputationOp>;
    using op_ptr             = std::shared_ptr<Op>;
    using scalar_t           = float;

    //================================================================================================
    // TensorInterface
    //================================================================================================

    class TensorInterface
    {
    public:
        virtual ~TensorInterface() {}
        virtual const ElementType& element_type() const = 0;
        virtual std::string        value_string() const = 0;
    };

    //================================================================================================
    // Tensor
    //================================================================================================

    template <typename T>
    class Tensor : public TensorInterface
    {
    public:
        Tensor(const T& val)
            : m_value{val}
            , m_element_type{element_type_float}
        {
        }

        virtual ~Tensor() {}
        const ElementType& element_type() const override { return m_element_type; }
        std::string        value_string() const override
        {
            std::string rc = "WTF";
            if (std::is_floating_point<T>::value)
            {
                std::stringstream ss;
                ss << m_value;
                rc = ss.str();
            }
            return rc;
        }

    private:
        T           m_value;
        ElementType m_element_type;
    };

    //================================================================================================
    // Transformer
    //================================================================================================

    class Transformer
    {
    public:
        virtual ~Transformer() {}
        virtual ExecutionState& execution_state() = 0;
    };

    //================================================================================================
    // TensorDescription
    //================================================================================================

    // class TensorDescription
    // {
    // public:
    //     virtual ~TensorDescription();
    //     virtual axes_key_t axes_key() const = 0;
    //     virtual std::string name() const = 0;
    //     virtual std::vector<size_t> shape() const = 0;
    //     virtual std::shared_ptr<TensorDescription> base() = 0;
    //     virtual ElementType element_type() const = 0;
    //     virtual size_t tensor_size() = 0;
    //     virtual bool is_persistent() = 0;
    //     virtual bool is_input() = 0;
    // };

    //================================================================================================
    // Op
    //================================================================================================

    // class Op
    // {
    //     // Any operation that can be in an AST.

    //     // Arguments:
    //     //     args: Values used by this node.
    //     //     const: The value of a constant Op, or None,
    //     //     constant (bool): The Op is constant.  Default False.
    //     //     forward: If not None, the node to use instead of this node.
    //     //     metadata: String key value dictionary for frontend metadata.
    //     //     kwargs: Args defined in related classes.

    //     // Attributes:
    //     //     const: The value of a constant.
    //     //     constant (bool): The value is constant.
    //     //     control_deps (OrderedSet): Ops in addtion to args that must run before this op.
    //     //     persistent (bool): The value will be retained from computation to computation and
    //     //         not shared.  Always True if reference is set.
    //     //     metadata: Dictionary with of string keys and values used for attaching
    //     //         arbitrary metadata to nodes.
    //     //     trainable: The value is trainable.
    // public:
    //     virtual ~Op() {}

    // virtual std::string name() const = 0;
    // virtual tensor_description_ptr tensor_description() = 0;
    // virtual op_ptr tensor() = 0;

    // virtual bool is_tensor_op() = 0;
    // virtual bool is_state_op() const = 0;
    // virtual bool is_sequencing_op() const = 0;
    // virtual op_ptr effective_tensor_op() = 0;
    // virtual const std::vector<op_ptr>& all_deps() const = 0;

    //     // ops

    //     // TODO support multiple types
    //     static op_ptr constant(float value)
    //     {
    //         op_ptr = make_shared<LiteralScalarOp>(value);
    //     }
    // };

    //================================================================================================
    // TensorOp
    //================================================================================================

    // class TensorOp : public Op
    // {
    // public:
    //     std::string name() const override { return "TensorOp"; }
    //     tensor_description_ptr tensor_description() override { return nullptr; }
    //     op_ptr tensor() override { return nullptr; }
    //     bool is_tensor_op() override { return true; }
    //     bool is_state_op() const override { return false; }
    //     op_ptr effective_tensor_op() override { return nullptr; }
    //     const std::vector<op_ptr>& all_deps() const override { return m_all_deps; }

    // private:
    //     std::vector<op_ptr> m_all_deps;
    // };

} // end of namespace ngraph
