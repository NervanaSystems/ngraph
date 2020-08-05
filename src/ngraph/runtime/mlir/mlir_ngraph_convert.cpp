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

#include "ngraph/runtime/mlir/mlir_ngraph_convert.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "ngraph/runtime/mlir/mlir_ngraph_ops.hpp"

#include "Ngraph/NgraphDialect.h"
#include "Ngraph/NgraphOps.h"

using namespace std;
using namespace ngraph;

runtime::mlir::NgraphToMlir::NgraphToMlir(::mlir::MLIRContext* context)
    : m_context(context)
    , m_builder(context)
{
}

// /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
// llvm::SmallVectorImpl<int64_t> runtime::mlir::NgraphToMlir::get_mlir_shape(const ngraph::Shape&
// ngraph_shape)
// {
//     llvm::SmallVectorImpl<int64_t> mlir_shape;
//     for (auto dim : ngraph_shape)
//     {
//         mlir_shape.push_back(dim);
//     }
//     return mlir_shape;
// }

// ::mlir::Type runtime::mlir::NgraphToMlir::get_mlir_type(const descriptor::Tensor* tensor)
// {
//     llvm::SmallVector<int64_t, 4> mlir_shape = get_mlir_shape(tensor->get_shape());
//     return mlir::NGTensorType::get(m_context, get_mlir_type(tensor->get_element_type()),
//     mlir_shape);
// }

::mlir::Type runtime::mlir::NgraphToMlir::get_mlir_type(const element::Type& type)
{
    ::mlir::Type rc;
    auto sign = ::mlir::IntegerType::SignednessSemantics::Signed;
    auto unsign = ::mlir::IntegerType::SignednessSemantics::Unsigned;

    switch (type)
    {
    case ngraph::element::Type_t::bf16: rc = ::mlir::FloatType::getBF16(m_context); break;
    case ngraph::element::Type_t::f16: rc = ::mlir::FloatType::getF16(m_context); break;
    case ngraph::element::Type_t::f32: rc = ::mlir::FloatType::getF32(m_context); break;
    case ngraph::element::Type_t::f64: rc = ::mlir::FloatType::getF64(m_context); break;
    case ngraph::element::Type_t::i8: rc = ::mlir::IntegerType::get(8, sign, m_context); break;
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::boolean:
        rc = ::mlir::IntegerType::get(8, unsign, m_context);
        break;
    case ngraph::element::Type_t::i16: rc = ::mlir::IntegerType::get(16, sign, m_context); break;
    case ngraph::element::Type_t::u16: rc = ::mlir::IntegerType::get(16, unsign, m_context); break;
    case ngraph::element::Type_t::i32: rc = ::mlir::IntegerType::get(32, sign, m_context); break;
    case ngraph::element::Type_t::u32: rc = ::mlir::IntegerType::get(32, unsign, m_context); break;
    case ngraph::element::Type_t::i64: rc = ::mlir::IntegerType::get(64, sign, m_context); break;
    case ngraph::element::Type_t::u64: rc = ::mlir::IntegerType::get(64, unsign, m_context); break;
    case ngraph::element::Type_t::undefined:
    case ngraph::element::Type_t::dynamic:
    case ngraph::element::Type_t::u1: throw runtime_error("MLIR: Unsupported nGraph type");
    }
    return rc;
}

void runtime::mlir::NgraphToMlir::convert_function(const ngraph::Function* ngraph_function)
{
    ::mlir::MLIRContext context;
    NgraphToMlir obj(&context);
    obj.convert(ngraph_function);
}

map<Output<Node>, ::mlir::Value>& runtime::mlir::NgraphToMlir::get_tensor_value_map()
{
    static map<Output<Node>, ::mlir::Value> ngraph_output_to_mlir_value;
    return ngraph_output_to_mlir_value;
}

void runtime::mlir::NgraphToMlir::set_tensor_value(const Output<Node>& t, ::mlir::Value value)
{
    get_tensor_value_map()[t] = value;
}

::mlir::Value runtime::mlir::NgraphToMlir::get_tensor_value(const Output<Node>& t)
{
    return get_tensor_value_map().at(t);
}

::mlir::Type runtime::mlir::NgraphToMlir::get_tensor_type(const Output<Node>& t)
{
    element::Type_t type = t.get_element_type();
    PartialShape pshape = t.get_partial_shape();
    ::mlir::Type mlir_element_type = ::mlir::FloatType::getF32(m_context);
    ::mlir::Type tensor;
    if (pshape.rank().is_static())
    {
        // Rank is static so can create a RankedTensorType
        vector<int64_t> dims;
        for (size_t i=0; i<pshape.rank().get_length(); ++i)
        {
            if (pshape[i] == Dimension::dynamic())
            {
                dims.push_back(-1);
            }
            else
            {
                dims.push_back(pshape[i].get_length());
            }
        }
        NGRAPH_INFO << "resulting shape " << join(dims);
        tensor = ::mlir::RankedTensorType::get(dims, mlir_element_type);
        // tensor = ::mlir::RankedTensorType::get(static_cast<int64_t>(pshape.rank().get_length()),
        //                                        mlir_element_type);
    }
    else
    {
        tensor = ::mlir::UnrankedTensorType::get(mlir_element_type);
    }

    return tensor;
}

void runtime::mlir::NgraphToMlir::convert(const ngraph::Function* ngraph_function)
{
    using TypeList = llvm::SmallVector<::mlir::Type, 4>;

    TypeList function_input_types;
    TypeList function_output_types;

    // mlir::ModuleOp module = getOperation();

    for (auto input : ngraph_function->get_parameters())
    {
        NGRAPH_INFO;
        function_input_types.push_back(get_mlir_type(input->get_output_element_type(0)));
        NGRAPH_INFO;
    }

    for (auto output : ngraph_function->get_results())
    {
        function_output_types.push_back(get_mlir_type(output->get_input_element_type(0)));
    }

    auto funcType =
        ::mlir::FunctionType::get(function_input_types, function_output_types, m_context);
    auto mlir_function =
        ::mlir::FuncOp::create(::mlir::UnknownLoc::get(m_context), "main", funcType);
    mlir_function.addEntryBlock();

    auto& region = mlir_function.getBody();
    m_builder.setInsertionPoint(&region.front(), region.front().begin());

    // Fill the ngraph tensor to mlir value map with all Parameters so that all inputs are available
    for (size_t i = 0; i < ngraph_function->get_parameters().size(); i++)
    {
        Output<Node> output = ngraph_function->get_parameters()[i]->output(0);
        ::mlir::Value value = mlir_function.getArgument(i);
        set_tensor_value(output, value);
    }

    ::mlir::edsc::ScopedContext scope(m_builder, ::mlir::UnknownLoc::get(m_context));
    for (shared_ptr<Node> ngraph_op : ngraph_function->get_ordered_ops())
    {
        ::mlir::OperationState ods_state(::mlir::UnknownLoc::get(m_context), ngraph_op->get_name());
        vector<::mlir::Value> input_values;
        vector<::mlir::Type> output_types;
        for (auto input : ngraph_op->input_values())
        {
            input_values.push_back(get_tensor_value(input));
        }
        for (auto output : ngraph_op->outputs())
        {
            output_types.push_back(get_tensor_type(output));
        }
        switch (get_typeid(*ngraph_op))
        {
        case runtime::mlir::OP_TYPEID::Parameter_v0: break;
        case runtime::mlir::OP_TYPEID::Result_v0: break;
        case runtime::mlir::OP_TYPEID::Add_v0:
        {
            NGRAPH_INFO << *ngraph_op;
            ::mlir::Value result = ::mlir::edsc::ValueBuilder<::mlir::ngraph::AddOp>(
                                       output_types[0], input_values[0], input_values[1])
                                       .value;
            set_tensor_value(ngraph_op->output(0), result);
            // if (mlir_op)
            // {
            //     for (auto i = 0; i < mlir_op->getNumResults(); i++)
            //     {
            //         auto result = mlir_op->getResult(i);
            //         if (result)
            //         {
            //             set_tensor_value(ngraph_op->output(i), result);
            //         }
            //     }
            // }

            break;
        }
        default: NGRAPH_INFO << "Unsupported " << *ngraph_op; break;
        }
    }

    // Create return
    vector<::mlir::Value> value_list;
    for (auto output : ngraph_function->get_results())
    {
        value_list.push_back(get_tensor_value(output->input(0).get_source_output()));
    }
    m_builder.create<::mlir::ReturnOp>(::mlir::UnknownLoc::get(m_context), value_list);

    mlir_function.dump();
}
