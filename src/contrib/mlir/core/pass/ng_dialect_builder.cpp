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

    auto& opDispatcher = getOpDispatcher();
    for (auto np : m_function->get_ordered_ops())
    {
        NGRAPH_INFO << *np;
        if (is_type<op::Parameter>(np) || is_type<op::Result>(np))
        {
            continue;
        }
        mlir::Operation* op = nullptr;
        NGRAPH_INFO << static_cast<int>(get_typeid(*np));
        NGRAPH_INFO << static_cast<int>(OP_TYPEID::Subtract_v0);
        NGRAPH_INFO << static_cast<int>(OP_TYPEID::Subtract_v1);
        NGRAPH_INFO << static_cast<int>(OP_TYPEID::StridedSlice_v1);
        // #define NGRAPH_OP(NAME, VER)                                                                       \
//     case OP_TYPEID::NAME##_v##VER:                                                                 \
//         NGRAPH_INFO << "ngraph::op::v##VER::NAME"; \
//         op = createOp_NAME_v##VER(*xxxxthis, np.get());                                  \
//         break;
        // #include "ngraph/op_version_tbl.hpp"
        // #undef NGRAPH_OP

        switch (get_typeid(*np))
        {
        case OP_TYPEID::Abs_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Acos_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Acosh_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Add_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Add_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::All_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::AllReduce_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::And_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Any_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ArgMax_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ArgMin_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Asin_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Asinh_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Atan_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Atan2_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Atanh_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::AvgPool_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::AvgPool_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::AvgPoolBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchMatMul_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchMatMulTranspose_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchNormInference_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchNormTraining_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchNormTrainingBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BatchToSpace_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BinaryConvolution_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Broadcast_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Broadcast_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Broadcast_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BroadcastDistributed_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::BroadcastLike_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Bucketize_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::CTCGreedyDecoder_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Ceiling_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Clamp_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Concat_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Constant_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Convert_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvertLike_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Convolution_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Convolution_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBackpropData_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBackpropData_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBackpropFilters_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBias_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBiasAdd_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Cos_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Cosh_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::CropAndResize_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::CrossEntropy_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::CrossEntropyBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::CumSum_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DeformableConvolution_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DeformablePSROIPooling_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DepthToSpace_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Dequantize_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DetectionOutput_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Divide_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Divide_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Dot_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DynBroadcast_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DynPad_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DynReplaceSlice_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::DynSlice_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Elu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::EmbeddingBagOffsetsSum_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::EmbeddingBagPackedSum_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::EmbeddingLookup_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::EmbeddingSegmentsSum_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Equal_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Equal_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Erf_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Exp_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ExtractImagePatches_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::FakeQuantize_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Floor_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::FloorMod_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GRN_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GRUCell_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Gather_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Gather_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GatherND_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GatherTree_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Gelu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GeluBackpropFactor_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Gemm_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GenerateMask_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Greater_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Greater_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GreaterEq_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GreaterEqual_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GroupConvolution_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GroupConvolution_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GroupConvolutionBackpropData_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GroupConvolutionBackpropData_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::GroupConvolutionBackpropFilters_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::HardSigmoid_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Interpolate_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Interpolate_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LRN_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LSTMCell_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LSTMSequence_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LayerNorm_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LayerNormBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Less_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Less_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LessEq_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LessEqual_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Log_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LogicalAnd_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LogicalNot_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LogicalOr_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::LogicalXor_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::MVN_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::MatMul_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Max_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::MaxPool_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::MaxPool_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::MaxPoolBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Maximum_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Maximum_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Min_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Minimum_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Minimum_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Mod_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Multiply_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Multiply_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Negative_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NonMaxSuppression_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NonMaxSuppression_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NonZero_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NormalizeL2_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Not_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NotEqual_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::NotEqual_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::OneHot_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::OneHot_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Or_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PRelu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PSROIPooling_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Pad_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Pad_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Parameter_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PartialSlice_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PartialSliceBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Passthrough_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Power_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Power_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PriorBox_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::PriorBoxClustered_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Product_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Proposal_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Quantize_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedConvolution_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedConvolutionBias_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedConvolutionBiasAdd_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedConvolutionRelu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedDot_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::QuantizedDotBias_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::RNNCell_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ROIAlign_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ROIPooling_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::RandomUniform_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Range_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Recv_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceLogicalAnd_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceLogicalOr_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceMax_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceMean_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceMin_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceProd_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReduceSum_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::RegionYolo_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Relu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReluBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReorgYolo_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReplaceSlice_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Reshape_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Reshape_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Result_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Reverse_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Reverse_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ReverseSequence_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Round_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScalarConstantLike_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScaleShift_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScatterAdd_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScatterElementsUpdate_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScatterND_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScatterNDAdd_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ScatterUpdate_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Select_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Select_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Selu_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Send_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ShapeOf_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ShapeOf_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::ShuffleChannels_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sigmoid_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SigmoidBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sign_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sin_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sinh_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Slice_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Softmax_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Softmax_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SoftmaxCrossEntropy_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SoftmaxCrossEntropyBackprop_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SpaceToBatch_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SpaceToDepth_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Split_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Split_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sqrt_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::SquaredDifference_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Squeeze_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Stack_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::StopGradient_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::StridedSlice_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Subtract_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Subtract_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Sum_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Tan_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Tanh_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::TensorIterator_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Tile_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::TopK_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::TopK_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::TopK_v3: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Transpose_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Unsqueeze_v0: NGRAPH_INFO << *np; break;
        case OP_TYPEID::VariadicSplit_v1: NGRAPH_INFO << *np; break;
        case OP_TYPEID::Xor_v0: NGRAPH_INFO << *np; break;
        }

        switch (get_typeid(*np))
        {
        case OP_TYPEID::Abs_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Abs*>(np.get()));
            break;
        }
        case OP_TYPEID::Acos_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Acos*>(np.get()));
            break;
        }
        case OP_TYPEID::Acosh_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Acosh*>(np.get()));
            break;
        }
        case OP_TYPEID::Add_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Add*>(np.get()));
            break;
        }
        case OP_TYPEID::Add_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Add*>(np.get()));
            break;
        }
        case OP_TYPEID::All_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::All*>(np.get()));
            break;
        }
        case OP_TYPEID::AllReduce_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::AllReduce*>(np.get()));
            break;
        }
        case OP_TYPEID::And_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::And*>(np.get()));
            break;
        }
        case OP_TYPEID::Any_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Any*>(np.get()));
            break;
        }
        case OP_TYPEID::ArgMax_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ArgMax*>(np.get()));
            break;
        }
        case OP_TYPEID::ArgMin_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ArgMin*>(np.get()));
            break;
        }
        case OP_TYPEID::Asin_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Asin*>(np.get()));
            break;
        }
        case OP_TYPEID::Asinh_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Asinh*>(np.get()));
            break;
        }
        case OP_TYPEID::Atan_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Atan*>(np.get()));
            break;
        }
        case OP_TYPEID::Atan2_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Atan2*>(np.get()));
            break;
        }
        case OP_TYPEID::Atanh_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Atanh*>(np.get()));
            break;
        }
        case OP_TYPEID::AvgPool_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::AvgPool*>(np.get()));
            break;
        }
        case OP_TYPEID::AvgPool_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::AvgPool*>(np.get()));
            break;
        }
        case OP_TYPEID::AvgPoolBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::AvgPoolBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchMatMul_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BatchMatMul*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchMatMulTranspose_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BatchMatMulTranspose*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchNormInference_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BatchNormInference*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchNormTraining_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BatchNormTraining*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BatchNormTrainingBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::BatchToSpace_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::BatchToSpace*>(np.get()));
            break;
        }
        case OP_TYPEID::BinaryConvolution_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::BinaryConvolution*>(np.get()));
            break;
        }
        case OP_TYPEID::Broadcast_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Broadcast*>(np.get()));
            break;
        }
        case OP_TYPEID::Broadcast_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Broadcast*>(np.get()));
            break;
        }
        case OP_TYPEID::Broadcast_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Broadcast*>(np.get()));
            break;
        }
        case OP_TYPEID::BroadcastDistributed_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BroadcastDistributed*>(np.get()));
            break;
        }
        case OP_TYPEID::BroadcastLike_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::BroadcastLike*>(np.get()));
            break;
        }
        case OP_TYPEID::Bucketize_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Bucketize*>(np.get()));
            break;
        }
        case OP_TYPEID::CTCGreedyDecoder_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::CTCGreedyDecoder*>(np.get()));
            break;
        }
        case OP_TYPEID::Ceiling_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Ceiling*>(np.get()));
            break;
        }
        case OP_TYPEID::Clamp_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Clamp*>(np.get()));
            break;
        }
        case OP_TYPEID::Concat_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Concat*>(np.get()));
            break;
        }
        case OP_TYPEID::Constant_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Constant*>(np.get()));
            break;
        }
        case OP_TYPEID::Convert_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Convert*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvertLike_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ConvertLike*>(np.get()));
            break;
        }
        case OP_TYPEID::Convolution_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Convolution*>(np.get()));
            break;
        }
        case OP_TYPEID::Convolution_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Convolution*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ConvolutionBackpropData*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ConvolutionBackpropData*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ConvolutionBackpropFilters*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBias_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ConvolutionBias*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBiasAdd_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ConvolutionBiasAdd*>(np.get()));
            break;
        }
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ConvolutionBiasBackpropFiltersBias*>(np.get()));
            break;
        }
        case OP_TYPEID::Cos_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Cos*>(np.get()));
            break;
        }
        case OP_TYPEID::Cosh_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Cosh*>(np.get()));
            break;
        }
        case OP_TYPEID::CropAndResize_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::CropAndResize*>(np.get()));
            break;
        }
        case OP_TYPEID::CrossEntropy_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::CrossEntropy*>(np.get()));
            break;
        }
        case OP_TYPEID::CrossEntropyBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::CrossEntropyBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::CumSum_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::CumSum*>(np.get()));
            break;
        }
        case OP_TYPEID::DeformableConvolution_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::DeformableConvolution*>(np.get()));
            break;
        }
        case OP_TYPEID::DeformablePSROIPooling_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::DeformablePSROIPooling*>(np.get()));
            break;
        }
        case OP_TYPEID::DepthToSpace_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DepthToSpace*>(np.get()));
            break;
        }
        case OP_TYPEID::Dequantize_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Dequantize*>(np.get()));
            break;
        }
        case OP_TYPEID::DetectionOutput_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DetectionOutput*>(np.get()));
            break;
        }
        case OP_TYPEID::Divide_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Divide*>(np.get()));
            break;
        }
        case OP_TYPEID::Divide_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Divide*>(np.get()));
            break;
        }
        case OP_TYPEID::Dot_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Dot*>(np.get()));
            break;
        }
        case OP_TYPEID::DynBroadcast_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DynBroadcast*>(np.get()));
            break;
        }
        case OP_TYPEID::DynPad_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DynPad*>(np.get()));
            break;
        }
        case OP_TYPEID::DynReplaceSlice_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DynReplaceSlice*>(np.get()));
            break;
        }
        case OP_TYPEID::DynSlice_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::DynSlice*>(np.get()));
            break;
        }
        case OP_TYPEID::Elu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Elu*>(np.get()));
            break;
        }
        case OP_TYPEID::EmbeddingBagOffsetsSum_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::EmbeddingBagOffsetsSum*>(np.get()));
            break;
        }
        case OP_TYPEID::EmbeddingBagPackedSum_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::EmbeddingBagPackedSum*>(np.get()));
            break;
        }
        case OP_TYPEID::EmbeddingLookup_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::EmbeddingLookup*>(np.get()));
            break;
        }
        case OP_TYPEID::EmbeddingSegmentsSum_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::EmbeddingSegmentsSum*>(np.get()));
            break;
        }
        case OP_TYPEID::Equal_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Equal*>(np.get()));
            break;
        }
        case OP_TYPEID::Equal_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Equal*>(np.get()));
            break;
        }
        case OP_TYPEID::Erf_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Erf*>(np.get()));
            break;
        }
        case OP_TYPEID::Exp_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Exp*>(np.get()));
            break;
        }
        case OP_TYPEID::ExtractImagePatches_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::ExtractImagePatches*>(np.get()));
            break;
        }
        case OP_TYPEID::FakeQuantize_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::FakeQuantize*>(np.get()));
            break;
        }
        case OP_TYPEID::Floor_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Floor*>(np.get()));
            break;
        }
        case OP_TYPEID::FloorMod_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::FloorMod*>(np.get()));
            break;
        }
        case OP_TYPEID::GRN_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GRN*>(np.get()));
            break;
        }
        case OP_TYPEID::GRUCell_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::GRUCell*>(np.get()));
            break;
        }
        case OP_TYPEID::Gather_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Gather*>(np.get()));
            break;
        }
        case OP_TYPEID::Gather_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Gather*>(np.get()));
            break;
        }
        case OP_TYPEID::GatherND_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GatherND*>(np.get()));
            break;
        }
        case OP_TYPEID::GatherTree_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::GatherTree*>(np.get()));
            break;
        }
        case OP_TYPEID::Gelu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Gelu*>(np.get()));
            break;
        }
        case OP_TYPEID::GeluBackpropFactor_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GeluBackpropFactor*>(np.get()));
            break;
        }
        case OP_TYPEID::Gemm_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Gemm*>(np.get()));
            break;
        }
        case OP_TYPEID::GenerateMask_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GenerateMask*>(np.get()));
            break;
        }
        case OP_TYPEID::Greater_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Greater*>(np.get()));
            break;
        }
        case OP_TYPEID::Greater_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Greater*>(np.get()));
            break;
        }
        case OP_TYPEID::GreaterEq_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GreaterEq*>(np.get()));
            break;
        }
        case OP_TYPEID::GreaterEqual_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::GreaterEqual*>(np.get()));
            break;
        }
        case OP_TYPEID::GroupConvolution_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GroupConvolution*>(np.get()));
            break;
        }
        case OP_TYPEID::GroupConvolution_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::GroupConvolution*>(np.get()));
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropData_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GroupConvolutionBackpropData*>(np.get()));
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropData_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::GroupConvolutionBackpropData*>(np.get()));
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropFilters_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::GroupConvolutionBackpropFilters*>(np.get()));
            break;
        }
        case OP_TYPEID::HardSigmoid_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::HardSigmoid*>(np.get()));
            break;
        }
        case OP_TYPEID::Interpolate_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Interpolate*>(np.get()));
            break;
        }
        case OP_TYPEID::Interpolate_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::Interpolate*>(np.get()));
            break;
        }
        case OP_TYPEID::LRN_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LRN*>(np.get()));
            break;
        }
        case OP_TYPEID::LSTMCell_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LSTMCell*>(np.get()));
            break;
        }
        case OP_TYPEID::LSTMSequence_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LSTMSequence*>(np.get()));
            break;
        }
        case OP_TYPEID::LayerNorm_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LayerNorm*>(np.get()));
            break;
        }
        case OP_TYPEID::LayerNormBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LayerNormBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::Less_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Less*>(np.get()));
            break;
        }
        case OP_TYPEID::Less_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Less*>(np.get()));
            break;
        }
        case OP_TYPEID::LessEq_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::LessEq*>(np.get()));
            break;
        }
        case OP_TYPEID::LessEqual_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::LessEqual*>(np.get()));
            break;
        }
        case OP_TYPEID::Log_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Log*>(np.get()));
            break;
        }
        case OP_TYPEID::LogicalAnd_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::LogicalAnd*>(np.get()));
            break;
        }
        case OP_TYPEID::LogicalNot_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::LogicalNot*>(np.get()));
            break;
        }
        case OP_TYPEID::LogicalOr_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::LogicalOr*>(np.get()));
            break;
        }
        case OP_TYPEID::LogicalXor_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::LogicalXor*>(np.get()));
            break;
        }
        case OP_TYPEID::MVN_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::MVN*>(np.get()));
            break;
        }
        case OP_TYPEID::MatMul_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::MatMul*>(np.get()));
            break;
        }
        case OP_TYPEID::Max_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Max*>(np.get()));
            break;
        }
        case OP_TYPEID::MaxPool_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::MaxPool*>(np.get()));
            break;
        }
        case OP_TYPEID::MaxPool_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::MaxPool*>(np.get()));
            break;
        }
        case OP_TYPEID::MaxPoolBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::MaxPoolBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::Maximum_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Maximum*>(np.get()));
            break;
        }
        case OP_TYPEID::Maximum_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Maximum*>(np.get()));
            break;
        }
        case OP_TYPEID::Min_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Min*>(np.get()));
            break;
        }
        case OP_TYPEID::Minimum_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Minimum*>(np.get()));
            break;
        }
        case OP_TYPEID::Minimum_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Minimum*>(np.get()));
            break;
        }
        case OP_TYPEID::Mod_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Mod*>(np.get()));
            break;
        }
        case OP_TYPEID::Multiply_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Multiply*>(np.get()));
            break;
        }
        case OP_TYPEID::Multiply_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Multiply*>(np.get()));
            break;
        }
        case OP_TYPEID::Negative_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Negative*>(np.get()));
            break;
        }
        case OP_TYPEID::NonMaxSuppression_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::NonMaxSuppression*>(np.get()));
            break;
        }
        case OP_TYPEID::NonMaxSuppression_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::NonMaxSuppression*>(np.get()));
            break;
        }
        case OP_TYPEID::NonZero_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::NonZero*>(np.get()));
            break;
        }
        case OP_TYPEID::NormalizeL2_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::NormalizeL2*>(np.get()));
            break;
        }
        case OP_TYPEID::Not_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Not*>(np.get()));
            break;
        }
        case OP_TYPEID::NotEqual_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::NotEqual*>(np.get()));
            break;
        }
        case OP_TYPEID::NotEqual_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::NotEqual*>(np.get()));
            break;
        }
        case OP_TYPEID::OneHot_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::OneHot*>(np.get()));
            break;
        }
        case OP_TYPEID::OneHot_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::OneHot*>(np.get()));
            break;
        }
        case OP_TYPEID::Or_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Or*>(np.get()));
            break;
        }
        case OP_TYPEID::PRelu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PRelu*>(np.get()));
            break;
        }
        case OP_TYPEID::PSROIPooling_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PSROIPooling*>(np.get()));
            break;
        }
        case OP_TYPEID::Pad_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Pad*>(np.get()));
            break;
        }
        case OP_TYPEID::Pad_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Pad*>(np.get()));
            break;
        }
        case OP_TYPEID::Parameter_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Parameter*>(np.get()));
            break;
        }
        case OP_TYPEID::PartialSlice_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PartialSlice*>(np.get()));
            break;
        }
        case OP_TYPEID::PartialSliceBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PartialSliceBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::Passthrough_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Passthrough*>(np.get()));
            break;
        }
        case OP_TYPEID::Power_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Power*>(np.get()));
            break;
        }
        case OP_TYPEID::Power_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Power*>(np.get()));
            break;
        }
        case OP_TYPEID::PriorBox_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PriorBox*>(np.get()));
            break;
        }
        case OP_TYPEID::PriorBoxClustered_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::PriorBoxClustered*>(np.get()));
            break;
        }
        case OP_TYPEID::Product_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Product*>(np.get()));
            break;
        }
        case OP_TYPEID::Proposal_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Proposal*>(np.get()));
            break;
        }
        case OP_TYPEID::Quantize_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Quantize*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedConvolution_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedConvolution*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBias_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedConvolutionBias*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasAdd_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedConvolutionBiasAdd*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedConvolutionBiasSignedAdd*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedConvolutionRelu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedConvolutionRelu*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedDot_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedDot*>(np.get()));
            break;
        }
        case OP_TYPEID::QuantizedDotBias_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::QuantizedDotBias*>(np.get()));
            break;
        }
        case OP_TYPEID::RNNCell_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::RNNCell*>(np.get()));
            break;
        }
        case OP_TYPEID::ROIAlign_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::ROIAlign*>(np.get()));
            break;
        }
        case OP_TYPEID::ROIPooling_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ROIPooling*>(np.get()));
            break;
        }
        case OP_TYPEID::RandomUniform_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::RandomUniform*>(np.get()));
            break;
        }
        case OP_TYPEID::Range_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Range*>(np.get()));
            break;
        }
        case OP_TYPEID::Recv_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Recv*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceLogicalAnd_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceLogicalAnd*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceLogicalOr_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceLogicalOr*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceMax_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceMax*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceMean_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceMean*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceMin_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceMin*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceProd_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceProd*>(np.get()));
            break;
        }
        case OP_TYPEID::ReduceSum_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::ReduceSum*>(np.get()));
            break;
        }
        case OP_TYPEID::RegionYolo_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::RegionYolo*>(np.get()));
            break;
        }
        case OP_TYPEID::Relu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Relu*>(np.get()));
            break;
        }
        case OP_TYPEID::ReluBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ReluBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::ReorgYolo_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ReorgYolo*>(np.get()));
            break;
        }
        case OP_TYPEID::ReplaceSlice_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ReplaceSlice*>(np.get()));
            break;
        }
        case OP_TYPEID::Reshape_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Reshape*>(np.get()));
            break;
        }
        case OP_TYPEID::Reshape_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Reshape*>(np.get()));
            break;
        }
        case OP_TYPEID::Result_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Result*>(np.get()));
            break;
        }
        case OP_TYPEID::Reverse_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Reverse*>(np.get()));
            break;
        }
        case OP_TYPEID::Reverse_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Reverse*>(np.get()));
            break;
        }
        case OP_TYPEID::ReverseSequence_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ReverseSequence*>(np.get()));
            break;
        }
        case OP_TYPEID::Round_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Round*>(np.get()));
            break;
        }
        case OP_TYPEID::ScalarConstantLike_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ScalarConstantLike*>(np.get()));
            break;
        }
        case OP_TYPEID::ScaleShift_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ScaleShift*>(np.get()));
            break;
        }
        case OP_TYPEID::ScatterAdd_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ScatterAdd*>(np.get()));
            break;
        }
        case OP_TYPEID::ScatterElementsUpdate_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::ScatterElementsUpdate*>(np.get()));
            break;
        }
        case OP_TYPEID::ScatterND_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ScatterND*>(np.get()));
            break;
        }
        case OP_TYPEID::ScatterNDAdd_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ScatterNDAdd*>(np.get()));
            break;
        }
        case OP_TYPEID::ScatterUpdate_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::ScatterUpdate*>(np.get()));
            break;
        }
        case OP_TYPEID::Select_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Select*>(np.get()));
            break;
        }
        case OP_TYPEID::Select_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Select*>(np.get()));
            break;
        }
        case OP_TYPEID::Selu_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Selu*>(np.get()));
            break;
        }
        case OP_TYPEID::Send_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Send*>(np.get()));
            break;
        }
        case OP_TYPEID::ShapeOf_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ShapeOf*>(np.get()));
            break;
        }
        case OP_TYPEID::ShapeOf_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::ShapeOf*>(np.get()));
            break;
        }
        case OP_TYPEID::ShuffleChannels_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::ShuffleChannels*>(np.get()));
            break;
        }
        case OP_TYPEID::Sigmoid_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sigmoid*>(np.get()));
            break;
        }
        case OP_TYPEID::SigmoidBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::SigmoidBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::Sign_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sign*>(np.get()));
            break;
        }
        case OP_TYPEID::Sin_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sin*>(np.get()));
            break;
        }
        case OP_TYPEID::Sinh_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sinh*>(np.get()));
            break;
        }
        case OP_TYPEID::Slice_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Slice*>(np.get()));
            break;
        }
        case OP_TYPEID::Softmax_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Softmax*>(np.get()));
            break;
        }
        case OP_TYPEID::Softmax_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Softmax*>(np.get()));
            break;
        }
        case OP_TYPEID::SoftmaxCrossEntropy_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::SoftmaxCrossEntropy*>(np.get()));
            break;
        }
        case OP_TYPEID::SoftmaxCrossEntropyBackprop_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::SoftmaxCrossEntropyBackprop*>(np.get()));
            break;
        }
        case OP_TYPEID::SpaceToBatch_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::SpaceToBatch*>(np.get()));
            break;
        }
        case OP_TYPEID::SpaceToDepth_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::SpaceToDepth*>(np.get()));
            break;
        }
        case OP_TYPEID::Split_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Split*>(np.get()));
            break;
        }
        case OP_TYPEID::Split_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Split*>(np.get()));
            break;
        }
        case OP_TYPEID::Sqrt_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sqrt*>(np.get()));
            break;
        }
        case OP_TYPEID::SquaredDifference_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::SquaredDifference*>(np.get()));
            break;
        }
        case OP_TYPEID::Squeeze_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Squeeze*>(np.get()));
            break;
        }
        case OP_TYPEID::Stack_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Stack*>(np.get()));
            break;
        }
        case OP_TYPEID::StopGradient_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::StopGradient*>(np.get()));
            break;
        }
        case OP_TYPEID::StridedSlice_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::StridedSlice*>(np.get()));
            break;
        }
        case OP_TYPEID::Subtract_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Subtract*>(np.get()));
            break;
        }
        case OP_TYPEID::Subtract_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Subtract*>(np.get()));
            break;
        }
        case OP_TYPEID::Sum_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Sum*>(np.get()));
            break;
        }
        case OP_TYPEID::Tan_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Tan*>(np.get()));
            break;
        }
        case OP_TYPEID::Tanh_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Tanh*>(np.get()));
            break;
        }
        case OP_TYPEID::TensorIterator_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::TensorIterator*>(np.get()));
            break;
        }
        case OP_TYPEID::Tile_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Tile*>(np.get()));
            break;
        }
        case OP_TYPEID::TopK_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::TopK*>(np.get()));
            break;
        }
        case OP_TYPEID::TopK_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::TopK*>(np.get()));
            break;
        }
        case OP_TYPEID::TopK_v3:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v3::TopK*>(np.get()));
            break;
        }
        case OP_TYPEID::Transpose_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::Transpose*>(np.get()));
            break;
        }
        case OP_TYPEID::Unsqueeze_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Unsqueeze*>(np.get()));
            break;
        }
        case OP_TYPEID::VariadicSplit_v1:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v1::VariadicSplit*>(np.get()));
            break;
        }
        case OP_TYPEID::Xor_v0:
        {
            NGRAPH_INFO;
            op = createOp(*this, static_cast<const ngraph::op::v0::Xor*>(np.get()));
            break;
        }
        default: NGRAPH_INFO; break;
        }
        NGRAPH_INFO;

        // auto it = opDispatcher.find(np->get_type_info());
        // if (it == opDispatcher.end())
        // {
        //     throw unsupported_op{std::string{"The MLIR backend doesn't currently implement the
        //     '"} +
        //                          np->description() + "' operation"};
        // }
        // mlir::Operation* op = it->second(*this, np.get());

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

// template <typename Op>
// mlir::Operation* pass::NgDialectConversionPass::createGenericOp(const ngraph::Node* ngNode,
//                                                                 int inNum)
// {
//     std::vector<mlir::Value> argValues;
//     std::vector<mlir::Type> resTypes;
//     std::shared_ptr<descriptor::Tensor> argTensor;
//     int i = 0;
//     for (auto& argOutput : ngNode->input_values())
//     {
//         if (inNum != -1 && i == inNum)
//         {
//             break;
//         }
//         argTensor = argOutput.get_tensor_ptr();
//         auto argV = getTensorValue(argTensor.get()).m_value;
//         argValues.push_back(argV);
//         i++;
//     }

//     for (auto& output : ngNode->outputs())
//     {
//         resTypes.push_back(getMlirType(output.get_tensor_ptr().get()));
//     }

//     return (m_builder.create<Op,
//                              ArrayRef<mlir::Type>,
//                              ArrayRef<mlir::Value>,
//                              ArrayRef<mlir::NamedAttribute>>(
//                 mlir::UnknownLoc::get(m_context), resTypes, argValues, {/* no attrs */}))
//         .getOperation();
// }

const pass::NgDialectConversionPass::MLIRCompOpMap& pass::NgDialectConversionPass::getOpDispatcher()
{
    static MLIRCompOpMap opDispatcher{
// #define MLIR_OP(OP)                                                                                \
//     {ngraph::op::OP::type_info, &pass::NgDialectConversionPass::createOp<ngraph::op::OP>},
// #include "contrib/mlir/core/ops_supported.inc"
    };
    return opDispatcher;
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
