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

#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/pass/ng_dialect_builder.hpp"
#include "ngraph/ops.hpp"

template <>
mlir::Operation* ngraph::pass::NgDialectConversionPass::createOp<ngraph::op::v0::Gemm>(
    NgDialectConversionPass& NgDialectObj, const ngraph::Node* ngNode)
{
    auto gemmNode = dynamic_cast<const ngraph::op::v0::Gemm*>(ngNode);
    NGRAPH_CHECK(
        ngNode, gemmNode != nullptr, "ngNode ", ngNode->description(), " is not a v0::Gemm");

    auto op = NgDialectObj.createGenericOp<mlir::NGGemmOp>(ngNode);
    auto gemmOp = llvm::cast<mlir::NGGemmOp>(op);
    gemmOp.setTransA(NgDialectObj.m_builder.getBoolAttr(gemmNode->get_transA()));
    gemmOp.setTransB(NgDialectObj.m_builder.getBoolAttr(gemmNode->get_transB()));
    gemmOp.setAlpha(NgDialectObj.m_builder.getF32FloatAttr(gemmNode->get_alpha()));
    gemmOp.setBeta(NgDialectObj.m_builder.getF32FloatAttr(gemmNode->get_beta()));
    return op;
}
