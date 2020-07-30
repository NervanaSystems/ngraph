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
                                                    const ngraph::op::v0::Softmax* ngNode)
{
    auto node = dynamic_cast<const ngraph::op::v0::Softmax*>(ngNode);
    NGRAPH_CHECK(
        ngNode, node != nullptr, "ngNode ", ngNode->description(), " is not a v0::Softmax");

    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGSoftMaxOp>(ngNode, 1);
    auto softmaxOp = llvm::cast<mlir::NGSoftMaxOp>(op);

    auto originArg = NgDialectObj.getOriginArg(ngNode->input_value(1).get_node());
    auto const_op = as_type<ngraph::op::Constant>(originArg);
    NGRAPH_INFO << "**********************************";
    NGRAPH_CHECK(ngNode, const_op != nullptr, "Input to softmax is not a Constant");

    AxisSet axes = const_op->get_axis_set_val();
    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(axes);
    softmaxOp.setAxes(attr);
    return op;
}
