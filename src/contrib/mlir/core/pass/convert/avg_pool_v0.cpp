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

mlir::Operation*
    ngraph::pass::NgDialectConversionPass::createOp(NgDialectConversionPass& NgDialectObj,
                                                    const ngraph::op::v0::AvgPool* ngNode)
{
    auto avgPoolNode = dynamic_cast<const ngraph::op::v0::AvgPool*>(ngNode);
    NGRAPH_CHECK(
        ngNode, avgPoolNode != nullptr, "ngNode ", ngNode->description(), " is not a v0::AvgPool");

    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGAvgPoolOp>(ngNode);
    auto avgPoolOp = llvm::cast<mlir::NGAvgPoolOp>(op);

    mlir::BoolAttr boolAttr =
        NgDialectObj.m_builder.getBoolAttr(avgPoolNode->get_include_padding_in_avg_computation());
    avgPoolOp.setIncludePadding(boolAttr);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(avgPoolNode->get_window_shape());
    avgPoolOp.setWindowShape(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolNode->get_window_movement_strides());
    avgPoolOp.setWindowMovementStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolNode->get_padding_below());
    avgPoolOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolNode->get_padding_above());
    avgPoolOp.setPadAbove(attr);
    return op;
}
