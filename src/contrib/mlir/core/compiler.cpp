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

#include "compiler.hpp"

#include "ngraph_dialect/dialect.hpp"
#include "ngraph_dialect/ops.hpp"
#include "ngraph_dialect/type.hpp"

#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "ngraph/type/element_type.hpp"

#include "contrib/mlir/utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <mutex>

// Defines a new LLVM debug type for this file to be used by LLVM_DEBUG macro.
#define DEBUG_TYPE "mlir-compiler"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

#define COMPILE_OP_DECL(op_name)                                                                   \
    createOp<op_name>(MLIRCompiler & compiler, const ngraph::Node* ngNode)

bool MLIRCompiler::initialized = false;

void MLIRCompiler::init()
{
    // Mutex to safely initialize MLIR.
    static std::mutex mlirInitMutex;

    std::unique_lock<std::mutex> lock(mlirInitMutex);

    if (!initialized)
    {
        initializeNGraphMLIR();

        // Register MLIR command line options in the pool of supported flags and and process flags
        // from environment variable to be used by nGraph, MLIR and LLVM.
        mlir::registerPassManagerCLOptions();
        llvm::cl::ParseEnvironmentOptions("ngraph", "NGRAPH_MLIR_OPTIONS", "");

        initialized = true;
    }
}

void MLIRCompiler::compile()
{
    buildNgDialectModule();
    // Free MLIR function builder.
    if (m_builder)
    {
        m_builder.reset(nullptr);
    }
}

// Creates an MLIR module and function with nGraph dialect ops from the input CompiledKernel.
void MLIRCompiler::buildNgDialectModule()
{
    // initialize an empty module
    m_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&m_context));

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
    buildNgDialect();
    m_module->push_back(function);
    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after lowering to NG dialect");
    }

    dumpMlirModule("nGraph Dialect Construction", m_module.get());
}

template <typename T>
void MLIRCompiler::getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape)
{
    for (auto dim : ngShape)
    {
        mlirShape.push_back(dim);
    }
}

template <typename T>
mlir::ArrayAttr MLIRCompiler::getShapeAsAttr(T ngShape)
{
    SmallVector<int64_t, 4> mlirShape;
    getMlirShape(ngShape, mlirShape);
    return m_builder->getI64ArrayAttr(mlirShape);
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion of the Tensor's
// element type.
mlir::Type MLIRCompiler::getMlirType(const descriptor::Tensor* tensor)
{
    llvm::SmallVector<int64_t, 4> mlirShape;
    getMlirShape(tensor->get_shape(), mlirShape);
    return mlir::NGTensorType::get(&m_context, getMlirType(tensor->get_element_type()), mlirShape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type MLIRCompiler::getMlirType(const element::Type& type)
{
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

    switch (type)
    {
    case ngraph::element::Type_t::undefined:
    case ngraph::element::Type_t::dynamic:
    default: NGRAPH_CHECK(false, "MLIR: Unsupported NGraph types"); break;
    case ngraph::element::Type_t::bf16: return mlir::NGFloatType::getBF16(&m_context);
    case ngraph::element::Type_t::f16: return mlir::NGFloatType::getF16(&m_context);
    case ngraph::element::Type_t::f32: return mlir::NGFloatType::getF32(&m_context);
    case ngraph::element::Type_t::f64: return mlir::NGFloatType::getF64(&m_context);
    case ngraph::element::Type_t::i8: return mlir::NGIntegerType::getInt8(&m_context);
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::boolean: return mlir::NGIntegerType::getUInt8(&m_context);
    case ngraph::element::Type_t::i16: return mlir::NGIntegerType::getInt16(&m_context);
    case ngraph::element::Type_t::u16: return mlir::NGIntegerType::getInt16(&m_context);
    case ngraph::element::Type_t::i32: return mlir::NGIntegerType::getInt32(&m_context);
    case ngraph::element::Type_t::u32: return mlir::NGIntegerType::getUInt32(&m_context);
    case ngraph::element::Type_t::i64: return mlir::NGIntegerType::getInt64(&m_context);
    case ngraph::element::Type_t::u64: return mlir::NGIntegerType::getUInt64(&m_context);
    }
    NGRAPH_CHECK(false, "Unreachable");
    return mlir::Type();

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

mlir::Type MLIRCompiler::getMlirType(const ngraph::Node* node)
{
    descriptor::Tensor* outTensor = node->get_output_tensor_ptr().get();
    return getMlirType(outTensor);
}

void MLIRCompiler::updateTensorValue(descriptor::Tensor* tensor, mlir::Value* value)
{
    NGRAPH_CHECK(m_tensorToValueMap.find(tensor) == m_tensorToValueMap.end(),
                 "tensor value already defined");
    TensorInfo tensorInfo{value};
    m_tensorToValueMap.insert(TensorToInfo(tensor, tensorInfo));
}

MLIRCompiler::TensorInfo MLIRCompiler::getTensorValue(descriptor::Tensor* tensor)
{
    auto it = m_tensorToValueMap.find(tensor);

    NGRAPH_CHECK(it != m_tensorToValueMap.end(), "Undefined tensor");

    return it->second;
}

// MLIR builders
#define TI(x) std::type_index(typeid(x))

void MLIRCompiler::buildNgDialect()
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

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Add)
            {
                return compiler.createGenericOp<mlir::NGAddOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Subtract)
            {
                return compiler.createGenericOp<mlir::NGSubOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Multiply)
            {
                return compiler.createGenericOp<mlir::NGMulOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Divide)
            {
                return compiler.createGenericOp<mlir::NGDivOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Greater)
            {
                return compiler.createGenericOp<mlir::NGGreaterOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Less)
            {
                return compiler.createGenericOp<mlir::NGLessOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Maximum)
            {
                return compiler.createGenericOp<mlir::NGMaxOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Minimum)
            {
                return compiler.createGenericOp<mlir::NGMinOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMax)
            {
                return compiler.createIndexReduction<mlir::NGArgMaxRedOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMin)
            {
                return compiler.createIndexReduction<mlir::NGArgMinRedOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Dot)
            {
                return compiler.createGenericOp<mlir::NGDotOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Concat)
            {
                auto concat = static_cast<const ngraph::op::Concat*>(ngNode);
                auto op = compiler.createGenericOp<mlir::NGConcatOp>(ngNode);
                op->setAttr(
                    "concatenation_axis",
                    compiler.m_builder->getI64IntegerAttr(concat->get_concatenation_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Gather)
            {
                auto gather = static_cast<const ngraph::op::Gather*>(ngNode);
                auto op = compiler.createGenericOp<mlir::NGGatherOp>(ngNode);
                op->setAttr("axis", compiler.m_builder->getI64IntegerAttr(gather->get_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Relu)
            {
                return compiler.createGenericOp<mlir::NGReluOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Negative)
            {
                return compiler.createGenericOp<mlir::NGNegOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Convolution)
            {
                mlir::Operation* op = compiler.createGenericOp<mlir::NGConvolutionOp>(ngNode);
                auto convNode = static_cast<const ngraph::op::Convolution*>(ngNode);
                auto convOp = llvm::cast<mlir::NGConvolutionOp>(op);

                mlir::ArrayAttr attr =
                    compiler.getShapeAsAttr(convNode->get_window_movement_strides());
                convOp.setStrides(attr);

                attr = compiler.getShapeAsAttr(convNode->get_padding_below());
                convOp.setPadBelow(attr);

                attr = compiler.getShapeAsAttr(convNode->get_padding_above());
                convOp.setPadAbove(attr);
                return op;
            }
        }
    }
}

template <typename Op>
mlir::Operation* MLIRCompiler::createGenericOp(const ngraph::Node* ngNode)
{
    std::vector<mlir::Value*> argValues;
    std::vector<mlir::Type> resTypes;
    auto inputMap = m_compiledKernel->get_input_map();
    std::shared_ptr<descriptor::Tensor> argTensor;
    for (auto& argOutput : ngNode->input_values())
    {
        auto argOutputNode = argOutput.get_node();
        if (as_type<op::Parameter>(argOutputNode))
        {
            auto it = inputMap.find(argOutputNode->shared_from_this());
            NGRAPH_CHECK(it != inputMap.end(), "Parameter not in CK input map");

            argTensor = m_compiledKernel->input_values().at(it->second).get_tensor_ptr();
        }
        else
        {
            argTensor = argOutput.get_tensor_ptr();
        }

        auto argV = getTensorValue(argTensor.get()).m_value;
        argValues.push_back(argV);
    }

    for (auto& output : ngNode->outputs())
    {
        resTypes.push_back(getMlirType(output.get_tensor_ptr().get()));
    }

    return (m_builder->create<Op,
                              ArrayRef<mlir::Type>,
                              ArrayRef<mlir::Value*>,
                              ArrayRef<mlir::NamedAttribute>>(
                mlir::UnknownLoc::get(&m_context), resTypes, argValues, {/* no attrs */}))
        .getOperation();
}

const MLIRCompiler::MLIRCompOpMap MLIRCompiler::opDispatcher{
#define MLIR_OP(OP) {TI(ngraph::op::OP), &MLIRCompiler::createOp<ngraph::op::OP>},
#include "ops_supported.inc"
};

void MLIRCompiler::createReturn()
{
    std::vector<mlir::Value*> valueList;
    for (auto output : m_compiledKernel->get_kernel_outputs())
    {
        valueList.push_back(getTensorValue(output->get_output_tensor_ptr().get()).m_value);
    }
    m_builder->create<mlir::NGReturnOp>(mlir::UnknownLoc::get(&m_context), valueList);
}

template <typename RedOp>
mlir::Operation* MLIRCompiler::createIndexReduction(const ngraph::Node* ngNode)
{
    auto* idxRed = static_cast<const ngraph::op::util::IndexReduction*>(ngNode);
    auto op = createGenericOp<RedOp>(ngNode);
    mlir::ArrayAttr redAxesAttr =
        m_builder->getI64ArrayAttr({(int64_t)idxRed->get_reduction_axis()});
    op->setAttr("axes", redAxesAttr);
    return op;
}
