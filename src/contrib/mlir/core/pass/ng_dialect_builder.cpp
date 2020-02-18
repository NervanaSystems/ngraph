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
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "ng_dialect_builder.hpp"
#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"

#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/type/element_type.hpp"

// Defines a new LLVM debug type for this file to be used by LLVM_DEBUG macro.
#define DEBUG_TYPE "mlir-compiler"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

#define COMPILE_OP_DECL(op_name)                                                                   \
    createOp<op_name>(NgDialectConversionPass & NgDialectObj, const ngraph::Node* ngNode)

namespace
{
    /// NgDialectConversionPass is an MLIR ModulePass Given an nGraph sub-graph, represented as
    /// CompiledKernel node, it
    /// translates the graph down to nGraph dialect

    class NgDialectConversionPass : public mlir::ModulePass<NgDialectConversionPass>
    {
    public:
        using TensorList = std::vector<descriptor::Tensor*>;
        using TypeList = llvm::SmallVector<mlir::Type, 4>;

        NgDialectConversionPass(const ngraph::op::CompiledKernel* compiled_kernel,
                                mlir::MLIRContext* context)
            : m_compiledKernel(compiled_kernel)
            , m_context(context)
            , m_builder(context)
        {
        }

        NgDialectConversionPass(const NgDialectConversionPass& obj);

    private:
        struct TensorInfo
        {
            // MLIR values this tensor maps to.
            mlir::Value m_value;
        };

    private:
        // Converts an nGraph sub-graph to MLIR nGraph dialect.
        void buildNgDialectModule();
        void buildNgDialect(mlir::FuncOp function);
        void runOnModule() override;
        // Applies any nGraph dialect optimizations
        void optimizeNgDialect() { /*TODO: Add Core NG dialect optimizations */}

        mlir::Type getMlirType(const descriptor::Tensor* tensor);
        mlir::Type getMlirType(const element::Type& type);
        mlir::Type getMlirType(const ngraph::Node* node);

        TensorInfo getTensorValue(descriptor::Tensor* tensor);
        void updateTensorValue(descriptor::Tensor* tensor, mlir::Value value);

        template <typename Op>
        static mlir::Operation* createOp(NgDialectConversionPass& NgDialectObj,
                                         const ngraph::Node* ngNode)
        {
            throw std::runtime_error("Unimplemented op '" + ngNode->description() +
                                     "' in MLIR Compiler");
        }

        // Generic op lowerer to ng dialect.
        // Simply maps ngraph tensors to values and generate an OP. No op-specific logic.
        // Use inNum when mlir OP needs less input than its corresponding ngraph OP.
        template <typename Op>
        mlir::Operation* createGenericOp(const ngraph::Node* ngNode, int inNum = -1);

        template <typename RedOp>
        mlir::Operation* createIndexReduction(const ngraph::Node* ngNode);

        void createReturn();

        /// Converts nGraph shape-like types \p ng_shape to MLIR shape \p mlir_shape.
        template <typename T>
        void getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape);

        /// Converts an ngraph shape to an I64 array attribute
        template <typename T>
        mlir::ArrayAttr getShapeAsAttr(T ngShape);
        /// Returns the builder
        mlir::OpBuilder& getBuilder() { return m_builder; }
        /// Return the real input node corresponding to the fake node
        ngraph::Node* getOriginArg(ngraph::Node* node) const;

    private:
        // Sub-graph to be compiled and executed with MLIR.
        const ngraph::op::CompiledKernel* m_compiledKernel;

        // MLIR context that holds all the MLIR information related to the sub-graph
        // compilation.
        mlir::MLIRContext* m_context;
        mlir::OpBuilder m_builder;

        using TensorToInfo = std::pair<descriptor::Tensor*, TensorInfo>;
        using TensorToInfoMap = std::unordered_map<descriptor::Tensor*, TensorInfo>;
        using MLIRCompOpFunction = std::function<mlir::Operation*(
            NgDialectConversionPass& NgDialectObj, const ngraph::Node*)>;
        using MLIRCompOpMap = std::unordered_map<Node::type_info_t, MLIRCompOpFunction>;

        // Maps tensor to the value it represents in the IR
        // use for MLIR dialect gen
        TensorToInfoMap m_tensorToValueMap;
        static const MLIRCompOpMap& getOpDispatcher();
    };

} // end of namespace
NgDialectConversionPass::NgDialectConversionPass(const NgDialectConversionPass& obj)
    : m_compiledKernel(obj.m_compiledKernel)
    , m_context(obj.m_context)
    , m_builder(obj.m_builder)
    , m_tensorToValueMap(obj.m_tensorToValueMap)
{
}

void NgDialectConversionPass::runOnModule()
{
    TypeList argsTypeList, resultTypeList;

    mlir::ModuleOp module = getModule();
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

    auto funcType = mlir::FunctionType::get(argsTypeList, resultTypeList, m_context);
    auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(m_context), "main", funcType);
    function.addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto input : kernelInputs)
    {
        auto arg = function.getArgument(i);
        TensorInfo tensorInfo{arg};
        m_tensorToValueMap.insert(TensorToInfo(input->get_output_tensor_ptr().get(), tensorInfo));
        i++;
    }

    // create builder
    buildNgDialect(function);
    module.push_back(function);
}

template <typename T>
void NgDialectConversionPass::getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape)
{
    for (auto dim : ngShape)
    {
        mlirShape.push_back(dim);
    }
}

template <typename T>
mlir::ArrayAttr NgDialectConversionPass::getShapeAsAttr(T ngShape)
{
    SmallVector<int64_t, 4> mlirShape;
    getMlirShape(ngShape, mlirShape);
    return m_builder.getI64ArrayAttr(mlirShape);
}

ngraph::Node* NgDialectConversionPass::getOriginArg(ngraph::Node* node) const
{
    auto inputMap = m_compiledKernel->get_input_map();
    auto it = inputMap.find(node->shared_from_this());
    NGRAPH_CHECK(it != inputMap.end(), "Parameter not in CK input map");
    return m_compiledKernel->input_values().at(it->second).get_node();
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion of the Tensor's
// element type.
mlir::Type NgDialectConversionPass::getMlirType(const descriptor::Tensor* tensor)
{
    llvm::SmallVector<int64_t, 4> mlirShape;
    getMlirShape(tensor->get_shape(), mlirShape);
    return mlir::NGTensorType::get(m_context, getMlirType(tensor->get_element_type()), mlirShape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type NgDialectConversionPass::getMlirType(const element::Type& type)
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
    case ngraph::element::Type_t::u1:
    default: NGRAPH_CHECK(false, "MLIR: Unsupported NGraph types"); break;
    case ngraph::element::Type_t::bf16: return mlir::NGFloatType::getBF16(m_context);
    case ngraph::element::Type_t::f16: return mlir::NGFloatType::getF16(m_context);
    case ngraph::element::Type_t::f32: return mlir::NGFloatType::getF32(m_context);
    case ngraph::element::Type_t::f64: return mlir::NGFloatType::getF64(m_context);
    case ngraph::element::Type_t::i8: return mlir::NGIntegerType::getInt8(m_context);
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::boolean: return mlir::NGIntegerType::getUInt8(m_context);
    case ngraph::element::Type_t::i16: return mlir::NGIntegerType::getInt16(m_context);
    case ngraph::element::Type_t::u16: return mlir::NGIntegerType::getInt16(m_context);
    case ngraph::element::Type_t::i32: return mlir::NGIntegerType::getInt32(m_context);
    case ngraph::element::Type_t::u32: return mlir::NGIntegerType::getUInt32(m_context);
    case ngraph::element::Type_t::i64: return mlir::NGIntegerType::getInt64(m_context);
    case ngraph::element::Type_t::u64: return mlir::NGIntegerType::getUInt64(m_context);
    }
    NGRAPH_CHECK(false, "Unreachable");
    return mlir::Type();

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

mlir::Type NgDialectConversionPass::getMlirType(const ngraph::Node* node)
{
    descriptor::Tensor* outTensor = node->get_output_tensor_ptr().get();
    return getMlirType(outTensor);
}

void NgDialectConversionPass::updateTensorValue(descriptor::Tensor* tensor, mlir::Value value)
{
    NGRAPH_CHECK(m_tensorToValueMap.find(tensor) == m_tensorToValueMap.end(),
                 "tensor value already defined");
    TensorInfo tensorInfo{value};
    m_tensorToValueMap.insert(TensorToInfo(tensor, tensorInfo));
}

NgDialectConversionPass::TensorInfo
    NgDialectConversionPass::getTensorValue(descriptor::Tensor* tensor)
{
    auto it = m_tensorToValueMap.find(tensor);

    NGRAPH_CHECK(it != m_tensorToValueMap.end(), "Undefined tensor");

    return it->second;
}

// MLIR builders
#define TI(x) std::type_index(typeid(x))

void NgDialectConversionPass::buildNgDialect(mlir::FuncOp function)
{
    auto& region = function.getBody();
    m_builder.setInsertionPoint(&region.front(), region.front().begin());
    const NodeVector& subGraph = m_compiledKernel->get_node_list();

    auto& opDispatcher = getOpDispatcher();
    for (auto np : subGraph)
    {
        auto it = opDispatcher.find(np->get_type_info());
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

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Add)
{
    return NgDialectObj.createGenericOp<mlir::NGAddOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Subtract)
{
    return NgDialectObj.createGenericOp<mlir::NGSubOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Multiply)
{
    return NgDialectObj.createGenericOp<mlir::NGMulOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Divide)
{
    return NgDialectObj.createGenericOp<mlir::NGDivOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Greater)
{
    return NgDialectObj.createGenericOp<mlir::NGGreaterOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Less)
{
    return NgDialectObj.createGenericOp<mlir::NGLessOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::GreaterEq)
{
    return NgDialectObj.createGenericOp<mlir::NGGreaterEqOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::LessEq)
{
    return NgDialectObj.createGenericOp<mlir::NGLessEqOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Equal)
{
    return NgDialectObj.createGenericOp<mlir::NGEqOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::NotEqual)
{
    return NgDialectObj.createGenericOp<mlir::NGNotEqOp>(ngNode);
}
template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Maximum)
{
    return NgDialectObj.createGenericOp<mlir::NGMaxOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Minimum)
{
    return NgDialectObj.createGenericOp<mlir::NGMinOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::ArgMax)
{
    return NgDialectObj.createIndexReduction<mlir::NGArgMaxRedOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::ArgMin)
{
    return NgDialectObj.createIndexReduction<mlir::NGArgMinRedOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Dot)
{
    return NgDialectObj.createGenericOp<mlir::NGDotOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Concat)
{
    auto concat = static_cast<const ngraph::op::Concat*>(ngNode);
    auto op = NgDialectObj.createGenericOp<mlir::NGConcatOp>(ngNode);
    op->setAttr("concatenation_axis",
                NgDialectObj.m_builder.getI64IntegerAttr(concat->get_concatenation_axis()));
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Gather)
{
    auto gather = static_cast<const ngraph::op::Gather*>(ngNode);
    auto op = NgDialectObj.createGenericOp<mlir::NGGatherOp>(ngNode);
    op->setAttr("axis", NgDialectObj.m_builder.getI64IntegerAttr(gather->get_axis()));
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Relu)
{
    return NgDialectObj.createGenericOp<mlir::NGReluOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Negative)
{
    return NgDialectObj.createGenericOp<mlir::NGNegOp>(ngNode);
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Convolution)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGConvolutionOp>(ngNode);
    auto convNode = static_cast<const ngraph::op::Convolution*>(ngNode);
    auto convOp = llvm::cast<mlir::NGConvolutionOp>(op);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(convNode->get_window_movement_strides());
    convOp.setStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(convNode->get_padding_below());
    convOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(convNode->get_padding_above());
    convOp.setPadAbove(attr);

    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::GroupConvolution)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGGroupConvOp>(ngNode);
    auto gConvNode = static_cast<const ngraph::op::GroupConvolution*>(ngNode);
    auto gConvOp = llvm::cast<mlir::NGGroupConvOp>(op);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(gConvNode->get_window_movement_strides());
    gConvOp.setStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(gConvNode->get_padding_below());
    gConvOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(gConvNode->get_padding_above());
    gConvOp.setPadAbove(attr);

    gConvOp.setGroups(NgDialectObj.getBuilder().getI64IntegerAttr(gConvNode->get_groups()));
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::AvgPool)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGAvgPoolOp>(ngNode);
    auto avgPoolNode = static_cast<const ngraph::op::AvgPool*>(ngNode);
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

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::AvgPoolBackprop)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGAvgPoolBackpropOp>(ngNode);
    auto avgPoolBackpropNode = static_cast<const ngraph::op::AvgPoolBackprop*>(ngNode);
    auto avgPoolBackpropOp = llvm::cast<mlir::NGAvgPoolBackpropOp>(op);

    mlir::BoolAttr boolAttr = NgDialectObj.m_builder.getBoolAttr(
        avgPoolBackpropNode->get_include_padding_in_avg_computation());
    avgPoolBackpropOp.setIncludePadding(boolAttr);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(avgPoolBackpropNode->get_window_shape());
    avgPoolBackpropOp.setWindowShape(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolBackpropNode->get_window_movement_strides());
    avgPoolBackpropOp.setWindowMovementStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolBackpropNode->get_padding_below());
    avgPoolBackpropOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolBackpropNode->get_padding_above());
    avgPoolBackpropOp.setPadAbove(attr);

    attr = NgDialectObj.getShapeAsAttr(avgPoolBackpropNode->get_forward_arg_shape());
    avgPoolBackpropOp.setForwardArgShape(attr);
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::MaxPool)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGMaxPoolOp>(ngNode);
    auto maxPoolNode = static_cast<const ngraph::op::MaxPool*>(ngNode);
    auto maxPoolOp = llvm::cast<mlir::NGMaxPoolOp>(op);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(maxPoolNode->get_window_shape());
    maxPoolOp.setWindowShape(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolNode->get_window_movement_strides());
    maxPoolOp.setWindowMovementStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolNode->get_padding_below());
    maxPoolOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolNode->get_padding_above());
    maxPoolOp.setPadAbove(attr);
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::MaxPoolBackprop)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGMaxPoolBackpropOp>(ngNode, 2);
    auto maxPoolBackpropNode = static_cast<const ngraph::op::MaxPool*>(ngNode);
    auto maxPoolBackpropOp = llvm::cast<mlir::NGMaxPoolBackpropOp>(op);

    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(maxPoolBackpropNode->get_window_shape());
    maxPoolBackpropOp.setWindowShape(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolBackpropNode->get_window_movement_strides());
    maxPoolBackpropOp.setWindowMovementStrides(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolBackpropNode->get_padding_below());
    maxPoolBackpropOp.setPadBelow(attr);

    attr = NgDialectObj.getShapeAsAttr(maxPoolBackpropNode->get_padding_above());
    maxPoolBackpropOp.setPadAbove(attr);
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::MatMul)
{
    auto matmulNode = static_cast<const ngraph::op::MatMul*>(ngNode);
    auto op = NgDialectObj.createGenericOp<mlir::NGMatMulOp>(ngNode);
    auto matmulOp = llvm::cast<mlir::NGMatMulOp>(op);
    matmulOp.setTransposeA(NgDialectObj.m_builder.getBoolAttr(matmulNode->get_transpose_a()));
    matmulOp.setTransposeB(NgDialectObj.m_builder.getBoolAttr(matmulNode->get_transpose_b()));
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Gemm)
{
    auto gemmNode = static_cast<const ngraph::op::Gemm*>(ngNode);
    auto op = NgDialectObj.createGenericOp<mlir::NGGemmOp>(ngNode);
    auto gemmOp = llvm::cast<mlir::NGGemmOp>(op);
    gemmOp.setTransA(NgDialectObj.m_builder.getBoolAttr(gemmNode->get_transA()));
    gemmOp.setTransB(NgDialectObj.m_builder.getBoolAttr(gemmNode->get_transB()));
    gemmOp.setAlpha(NgDialectObj.m_builder.getF32FloatAttr(gemmNode->get_alpha()));
    gemmOp.setBeta(NgDialectObj.m_builder.getF32FloatAttr(gemmNode->get_beta()));
    return op;
}

template <>
mlir::Operation* NgDialectConversionPass::COMPILE_OP_DECL(ngraph::op::Softmax)
{
    mlir::Operation* op = NgDialectObj.createGenericOp<mlir::NGSoftMaxOp>(ngNode, 1);
    auto softmaxOp = llvm::cast<mlir::NGSoftMaxOp>(op);

    auto originArg = NgDialectObj.getOriginArg(ngNode->input_value(1).get_node());
    auto const_op = static_cast<ngraph::op::Constant*>(originArg);

    AxisSet axes = const_op->get_axis_set_val();
    mlir::ArrayAttr attr = NgDialectObj.getShapeAsAttr(axes);
    softmaxOp.setAxes(attr);
    return op;
}
template <typename Op>
mlir::Operation* NgDialectConversionPass::createGenericOp(const ngraph::Node* ngNode, int inNum)
{
    std::vector<mlir::Value> argValues;
    std::vector<mlir::Type> resTypes;
    auto inputMap = m_compiledKernel->get_input_map();
    std::shared_ptr<descriptor::Tensor> argTensor;
    int i = 0;
    for (auto& argOutput : ngNode->input_values())
    {
        if (inNum != -1 && i == inNum)
        {
            break;
        }
        auto argOutputNode = argOutput.get_node();
        if (is_type<op::Parameter>(argOutputNode))
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
        i++;
    }

    for (auto& output : ngNode->outputs())
    {
        resTypes.push_back(getMlirType(output.get_tensor_ptr().get()));
    }

    return (m_builder.create<Op,
                             ArrayRef<mlir::Type>,
                             ArrayRef<mlir::Value>,
                             ArrayRef<mlir::NamedAttribute>>(
                mlir::UnknownLoc::get(m_context), resTypes, argValues, {/* no attrs */}))
        .getOperation();
}

const NgDialectConversionPass::MLIRCompOpMap& NgDialectConversionPass::getOpDispatcher()
{
    static MLIRCompOpMap opDispatcher{
#define MLIR_OP(OP) {ngraph::op::OP::type_info, &NgDialectConversionPass::createOp<ngraph::op::OP>},
#include "contrib/mlir/core/ops_supported.inc"
    };
    return opDispatcher;
}

void NgDialectConversionPass::createReturn()
{
    std::vector<mlir::Value> valueList;
    for (auto output : m_compiledKernel->get_kernel_outputs())
    {
        valueList.push_back(getTensorValue(output->get_output_tensor_ptr().get()).m_value);
    }
    m_builder.create<mlir::NGReturnOp>(mlir::UnknownLoc::get(m_context), valueList);
}

template <typename RedOp>
mlir::Operation* NgDialectConversionPass::createIndexReduction(const ngraph::Node* ngNode)
{
    auto* idxRed = static_cast<const ngraph::op::util::IndexReduction*>(ngNode);
    auto op = createGenericOp<RedOp>(ngNode);
    mlir::ArrayAttr redAxesAttr =
        m_builder.getI64ArrayAttr({(int64_t)idxRed->get_reduction_axis()});
    op->setAttr("axes", redAxesAttr);
    return op;
}

std::unique_ptr<mlir::Pass>
    ngraph::pass::createNgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel,
                                                mlir::MLIRContext* context)
{
    return std::make_unique<NgDialectConversionPass>(compiledKernel, context);
}
