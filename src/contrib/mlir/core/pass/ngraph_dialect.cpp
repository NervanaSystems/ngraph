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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "ngraph_dialect.hpp"
#include "contrib/mlir/utils.hpp"
using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

ngraph::pass::NgDialectConversionPass::NgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel)
    :m_compiledKernel(compiledKernel)
{

}

void ngraph::pass::NgDialectConversionPass::buildNgDialect()
{
    const NodeVector& subGraph = m_compiledKernel->get_node_list();

    for (auto np : subGraph)
    {
        auto it = opDispatcher.find(TI(*np));
        if (it == opDispatcher.end())
        {
            throw unsupported_op{std::string{"The MLIR backend doesn't currently implement the '"} +
                                 np->description() + "' operation"};
        }
        mlir::Operation* op = it->second(*this, np.get());
        // This assumes simple 1:1 mapping between output edges and generated MLIR op results
        // If the mapping is more complex, the create_op helper can return null operation
        // and handles populating the value map itself
        if (op)
        {
            for (auto i = 0; i < op->getNumResults(); i++)
            {
                mlir::Value* result = op->getResult(i);
                if (result)
                {
                    updateTensorValue(np->get_output_tensor_ptr(i).get(), result);
                }
            }
        }
    }
    createReturn();
}

void ngraph::pass::NgDialectConversionPass::runOnModule()
{
    TypeList argsTypeList, resultTypeList;

    // Retrieve input and output tensors.
    const auto& kernelInputs = m_compiledKernel->get_arguments();
    const auto& kernelOutput = m_compiledKernel->get_kernel_outputs();
    NGRAPH_CHECK(kernelInputs.size() != 0, "Cannot have empty inputs list");
    NGRAPH_CHECK(kernelOutput.size() != 0, "Cannot have empty outputs list");

    for (auto input : kernelInputs)
    {
        argsTypeList.push_back(getMlirType(input.get()));
    }

    for (auto output : kernelOutput)
    {
        resultTypeList.push_back(getMlirType(output.get()));
    }

    auto funcType = mlir::FunctionType::get(argsTypeList, resultTypeList, &m_context);
    auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(&m_context), "main", funcType);
    function.addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto input : kernelInputs)
    {
        mlir::Value* arg = function.getArgument(i);
        TensorInfo tensorInfo{arg};
        m_tensorToValueMap.insert(TensorToInfo(input->get_output_tensor_ptr().get(), tensorInfo));
        i++;
    }

    // create builder
    m_builder = std::unique_ptr<mlir::OpBuilder>(new mlir::OpBuilder(function.getBody()));
    ngraph::pass::NgDialectConversionPass::buildNgDialect();
}

std::unique_ptr<mlir::Pass> ngraph::pass::CreateNgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel)
{
	return std::make_unique<ngraph::pass::NgDialectConversionPass>(compiledKernel);
}
