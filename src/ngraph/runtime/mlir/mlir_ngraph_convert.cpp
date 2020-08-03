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

#include "Ngraph/NgraphDialect.h"

using namespace std;
using namespace ngraph;

// /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
// llvm::SmallVectorImpl<int64_t> runtime::mlir::NgraphToMlir::get_mlir_shape(const ngraph::Shape& ngraph_shape)
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
//     return mlir::NGTensorType::get(m_context, get_mlir_type(tensor->get_element_type()), mlir_shape);
// }

// ::mlir::Type runtime::mlir::NgraphToMlir::get_mlir_type(const element::Type& type)
// {
// #if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
// #pragma GCC diagnostic push
// #pragma GCC diagnostic error "-Wswitch"
// #pragma GCC diagnostic error "-Wswitch-enum"
// #endif

//     auto sign = mlir::NGIntegerType::SignednessSemantics::Signed;
//     auto unsign = mlir::NGIntegerType::SignednessSemantics::Unsigned;

//     switch (type)
//     {
//     case ngraph::element::Type_t::undefined:
//     case ngraph::element::Type_t::dynamic:
//     case ngraph::element::Type_t::u1:
//     default: NGRAPH_CHECK(false, "MLIR: Unsupported NGraph types"); break;
//     case ngraph::element::Type_t::bf16: return mlir::NGFloatType::getBF16(m_context);
//     case ngraph::element::Type_t::f16: return mlir::NGFloatType::getF16(m_context);
//     case ngraph::element::Type_t::f32: return mlir::NGFloatType::getF32(m_context);
//     case ngraph::element::Type_t::f64: return mlir::NGFloatType::getF64(m_context);
//     case ngraph::element::Type_t::i8: return mlir::NGIntegerType::get(8, sign, m_context);
//     case ngraph::element::Type_t::u8:
//     case ngraph::element::Type_t::boolean: return mlir::NGIntegerType::get(8, unsign, m_context);
//     case ngraph::element::Type_t::i16: return mlir::NGIntegerType::get(16, sign, m_context);
//     case ngraph::element::Type_t::u16: return mlir::NGIntegerType::get(16, unsign, m_context);
//     case ngraph::element::Type_t::i32: return mlir::NGIntegerType::get(32, sign, m_context);
//     case ngraph::element::Type_t::u32: return mlir::NGIntegerType::get(32, unsign, m_context);
//     case ngraph::element::Type_t::i64: return mlir::NGIntegerType::get(64, sign, m_context);
//     case ngraph::element::Type_t::u64: return mlir::NGIntegerType::get(64, unsign, m_context);
//     }
//     NGRAPH_CHECK(false, "Unreachable");
//     return mlir::Type();

// #if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
// #pragma GCC diagnostic pop
// #endif
// }

void runtime::mlir::NgraphToMlir::convert_function(const ngraph::Function* ngraph_function)
{
    NgraphToMlir obj;
    obj.convert(ngraph_function);
}

void runtime::mlir::NgraphToMlir::convert(const ngraph::Function* ngraph_function)
{
    using TypeList = llvm::SmallVector<::mlir::Type, 4>;

    ::mlir::MLIRContext context_obj;
    ::mlir::MLIRContext* context = &context_obj;

    TypeList argsTypeList;
    TypeList resultTypeList;

    // mlir::ModuleOp module = getOperation();

    // for (auto input : ngraph_function->get_parameters())
    // {
    //     argsTypeList.push_back(get_mlir_type(&input->get_output_tensor(0)));
    // }

    // for (auto output : ngraph_function->get_results())
    // {
    //     resultTypeList.push_back(get_mlir_type(&output->get_input_tensor(0)));
    // }

    // auto funcType = ::mlir::FunctionType::get(argsTypeList, resultTypeList, context);
    // auto mlir_function = ::mlir::FuncOp::create(::mlir::UnknownLoc::get(context), "main", funcType);
    // mlir_function.addEntryBlock();

    // ::mlir::OpBuilder op_builder(context);



    // Create return
    // std::vector<mlir::Value> valueList;
    // for (auto output : function->get_results())
    // {
    //     valueList.push_back(getTensorValue(&output->get_input_tensor(0)).m_value);
    // }
    // m_builder.create<mlir::NGReturnOp>(mlir::UnknownLoc::get(m_context), valueList);
}
