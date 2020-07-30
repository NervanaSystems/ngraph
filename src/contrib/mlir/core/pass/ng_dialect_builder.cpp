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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming
// convention.

#include "ng_dialect_builder.hpp"
#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/type/element_type.hpp"

// Defines a new LLVM debug type for this file to be used by LLVM_DEBUG macro.
#define DEBUG_TYPE "mlir-compiler"

using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::StringRef;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

namespace
{
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

    OP_TYPEID get_typeid(const Node& node)
    {
        const NodeTypeInfo& type_info = node.get_type_info();
        // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
        // {Abs::type_info, OP_TYPEID::Abs},
        // {Acos::type_info, OP_TYPEID::Acos},
        // ...
        static const std::map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, VERSION)                                                                   \
    {ngraph::op::v##VERSION::NAME::type_info, OP_TYPEID::NAME##_v##VERSION},
#include "ngraph/op_version_tbl.hpp"
#undef NGRAPH_OP
        };
        OP_TYPEID rc = OP_TYPEID::UnknownOp;

        auto it = type_info_map.find(type_info);
        if (it != type_info_map.end())
        {
            rc = it->second;
        }
        return rc;
    }
}

pass::NgDialectConversionPass::NgDialectConversionPass(const NgDialectConversionPass& obj)
    : m_function(obj.m_function)
    , m_context(obj.m_context)
    , m_builder(obj.m_builder)
    , m_tensorToValueMap(obj.m_tensorToValueMap)
{
}

void pass::NgDialectConversionPass::runOnOperation()
{
    TypeList argsTypeList, resultTypeList;

    mlir::ModuleOp module = getOperation();
    // Retrieve input and output tensors.
    ParameterVector kernelInputs = m_function->get_parameters();
    ResultVector kernelOutput = m_function->get_results();
    // NGRAPH_CHECK(kernelInputs.size() != 0, "Cannot have empty inputs list");
    // NGRAPH_CHECK(kernelOutput.size() != 0, "Cannot have empty outputs list");

    for (auto input : kernelInputs)
    {
        argsTypeList.push_back(getMlirType(&input->get_output_tensor(0)));
    }

    for (auto output : kernelOutput)
    {
        resultTypeList.push_back(getMlirType(&output->get_input_tensor(0)));
    }

    auto funcType = mlir::FunctionType::get(argsTypeList, resultTypeList, m_context);
    auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(m_context), "main", funcType);
    function.addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto p : m_function->get_parameters())
    {
        auto paramNode = p;
        auto argValue = function.getArgument(i);
        m_tensorToValueMap.insert(
            TensorToInfo(paramNode->get_output_tensor_ptr(0).get(), {argValue}));
        i++;
    }

    // create builder
    buildNgDialect(function);
    module.push_back(function);
}

ngraph::Node* pass::NgDialectConversionPass::getOriginArg(ngraph::Node* node) const
{
    return node;
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion
// of the Tensor's
// element type.
mlir::Type pass::NgDialectConversionPass::getMlirType(const descriptor::Tensor* tensor)
{
    llvm::SmallVector<int64_t, 4> mlirShape;
    getMlirShape(tensor->get_shape(), mlirShape);
    return mlir::NGTensorType::get(m_context, getMlirType(tensor->get_element_type()), mlirShape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type pass::NgDialectConversionPass::getMlirType(const element::Type& type)
{
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

    auto sign = mlir::NGIntegerType::SignednessSemantics::Signed;
    auto unsign = mlir::NGIntegerType::SignednessSemantics::Unsigned;

    switch (type)
    {
    case ngraph::element::Type_t::undefined:
    case ngraph::element::Type_t::dynamic:
    case ngraph::element::Type_t::u1:
    default: NGRAPH_CHECK(false, "MLIR: Unsupported NGraph types"); break;
    case ngraph::element::Type_t::bf16: return mlir::NGFloatType::getBF16(m_context);
    case ngraph::element::Type_t::f16: return mlir::NGFloatType::getF16(m_context);
    case ngraph::element::Type_t::f32: return mlir::NGFloatType::getF32(m_context);
    case ngraph::element::Type_t::f64: return mlir::NGFloatType::getF64(m_context);
    case ngraph::element::Type_t::i8: return mlir::NGIntegerType::get(8, sign, m_context);
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::boolean: return mlir::NGIntegerType::get(8, unsign, m_context);
    case ngraph::element::Type_t::i16: return mlir::NGIntegerType::get(16, sign, m_context);
    case ngraph::element::Type_t::u16: return mlir::NGIntegerType::get(16, unsign, m_context);
    case ngraph::element::Type_t::i32: return mlir::NGIntegerType::get(32, sign, m_context);
    case ngraph::element::Type_t::u32: return mlir::NGIntegerType::get(32, unsign, m_context);
    case ngraph::element::Type_t::i64: return mlir::NGIntegerType::get(64, sign, m_context);
    case ngraph::element::Type_t::u64: return mlir::NGIntegerType::get(64, unsign, m_context);
    }
    NGRAPH_CHECK(false, "Unreachable");
    return mlir::Type();

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

void pass::NgDialectConversionPass::updateTensorValue(descriptor::Tensor* tensor, mlir::Value value)
{
    NGRAPH_CHECK(m_tensorToValueMap.find(tensor) == m_tensorToValueMap.end(),
                 "tensor value already defined");
    TensorInfo tensorInfo{value};
    m_tensorToValueMap.insert(TensorToInfo(tensor, tensorInfo));
}

pass::NgDialectConversionPass::TensorInfo
    pass::NgDialectConversionPass::getTensorValue(descriptor::Tensor* tensor)
{
    auto it = m_tensorToValueMap.find(tensor);

    NGRAPH_CHECK(it != m_tensorToValueMap.end(), "Undefined tensor");

    return it->second;
}

void pass::NgDialectConversionPass::buildNgDialect(mlir::FuncOp function)
{
    auto& region = function.getBody();
    m_builder.setInsertionPoint(&region.front(), region.front().begin());

    for (auto np : m_function->get_ordered_ops())
    {
        NGRAPH_INFO << *np;
        if (is_type<op::Parameter>(np) || is_type<op::Result>(np))
        {
            continue;
        }
        mlir::Operation* op = nullptr;

        switch (get_typeid(*np))
        {
#define NGRAPH_OP(NAME, VER)                                                                       \
    case OP_TYPEID::NAME##_v##VER:                                                                 \
        op = createOp(*this, static_cast<const ngraph::op::v##VER::NAME*>(np.get()));              \
        break;
#include "ngraph/op_version_tbl.hpp"
#undef NGRAPH_OP
        }
        NGRAPH_INFO;

        // This assumes simple 1:1 mapping between output edges and generated MLIR
        // op results
        // If the mapping is more complex, the create_op helper can return null
        // operation
        // and handles populating the value map itself
        if (op)
        {
            for (auto i = 0; i < op->getNumResults(); i++)
            {
                auto result = op->getResult(i);
                if (result)
                {
                    updateTensorValue(np->get_output_tensor_ptr(i).get(), result);
                }
            }
        }
    }
    createReturn();
}

void pass::NgDialectConversionPass::createReturn()
{
    std::vector<mlir::Value> valueList;
    for (auto output : m_function->get_results())
    {
        valueList.push_back(getTensorValue(&output->get_input_tensor(0)).m_value);
    }
    m_builder.create<mlir::NGReturnOp>(mlir::UnknownLoc::get(m_context), valueList);
}

std::unique_ptr<mlir::Pass>
    ngraph::pass::createNgDialectConversionPass(std::shared_ptr<ngraph::Function> function,
                                                mlir::MLIRContext* context)
{
    return std::make_unique<NgDialectConversionPass>(function, context);
}
