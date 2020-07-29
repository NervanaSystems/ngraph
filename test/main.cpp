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

#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/utils.hpp"
#endif
#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
#include <pybind11/embed.h>
#endif

using namespace std;

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
namespace py = pybind11;
#endif

string generate_filename(string name, string x, string version)
{
    stringstream ss;
    for (size_t i = 0; i < name.size(); ++i)
    {
        char c = name[i];
        if (isupper(c))
        {
            c = tolower(c);
            if (i > 0 && i < name.size() - 1 && !isupper(name[i + 1]))
            {
                ss << "_";
            }
        }
        ss << c;
    }
    ss << "_v" << version;
    return ss.str();
}

void generate_file(string name, string x, string version)
{
    string filename = generate_filename(name, x, version);

    string source =
        R"(//*****************************************************************************
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

#include "contrib/mlir/core/pass/ng_dialect_builder.hpp"
#include "ngraph/ops.hpp"

template <>
mlir::Operation*
    ngraph::pass::NgDialectConversionPass::createOp<ngraph::op::v@VER::@OP>
        (NgDialectConversionPass& NgDialectObj, const ngraph::Node* ngNode)
{
    auto node = dynamic_cast<const ngraph::op::v@VER::@OP*>(ngNode);
    NGRAPH_CHECK(ngNode, node!=nullptr, "ngNode ", ngNode->description(), " is not a v@VER::@OP");
    throw unsupported_op("Unsupported op 'v@VER::@OP'");
}
)";

    regex r1("@OP");
    regex r2("@VER");
    source = regex_replace(source, r1, name);
    source = regex_replace(source, r2, version);

    filename =
        "/nfs/pdx/home/rhkimbal/dev/ngraph/src/contrib/mlir/core/pass/convert/" + filename + ".cpp";
    NGRAPH_INFO << filename;

    ofstream f(filename);
    f << source;
}

void xxx()
{
    generate_file("Abs", "Abs", "0");
    generate_file("Acos", "Acos", "0");
    generate_file("Acosh", "Acosh", "3");
    generate_file("Add", "Add", "0");
    generate_file("Add", "Add", "1");
    generate_file("All", "All", "0");
    generate_file("AllReduce", "AllReduce", "0");
    generate_file("And", "And", "0");
    generate_file("Any", "Any", "0");
    generate_file("ArgMax", "ArgMax", "0");
    generate_file("ArgMin", "ArgMin", "0");
    generate_file("Asin", "Asin", "0");
    generate_file("Asinh", "Asinh", "3");
    generate_file("Atan", "Atan", "0");
    generate_file("Atan2", "Atan2", "0");
    generate_file("Atanh", "Atanh", "3");
    generate_file("AvgPool", "AvgPool", "0");
    generate_file("AvgPool", "AvgPool", "1");
    generate_file("AvgPoolBackprop", "AvgPoolBackprop", "0");
    generate_file("BatchMatMul", "BatchMatMul", "0");
    generate_file("BatchMatMulTranspose", "BatchMatMulTranspose", "0");
    generate_file("BatchNormInference", "BatchNormInference", "0");
    generate_file("BatchNormTraining", "BatchNormTraining", "0");
    generate_file("BatchNormTrainingBackprop", "BatchNormTrainingBackprop", "0");
    generate_file("BatchToSpace", "BatchToSpace", "1");
    generate_file("BinaryConvolution", "BinaryConvolution", "1");
    generate_file("Broadcast", "Broadcast", "0");
    generate_file("Broadcast", "Broadcast", "1");
    generate_file("Broadcast", "Broadcast", "3");
    generate_file("BroadcastDistributed", "BroadcastDistributed", "0");
    generate_file("BroadcastLike", "BroadcastLike", "0");
    generate_file("Bucketize", "Bucketize", "3");
    generate_file("CTCGreedyDecoder", "CTCGreedyDecoder", "0");
    generate_file("Ceiling", "Ceiling", "0");
    generate_file("Clamp", "Clamp", "0");
    generate_file("Concat", "Concat", "0");
    generate_file("Constant", "Constant", "0");
    generate_file("Convert", "Convert", "0");
    generate_file("ConvertLike", "ConvertLike", "1");
    generate_file("Convolution", "Convolution", "0");
    generate_file("Convolution", "Convolution", "1");
    generate_file("ConvolutionBackpropData", "ConvolutionBackpropData", "0");
    generate_file("ConvolutionBackpropData", "ConvolutionBackpropData", "1");
    generate_file("ConvolutionBackpropFilters", "ConvolutionBackpropFilters", "0");
    generate_file("ConvolutionBias", "ConvolutionBias", "0");
    generate_file("ConvolutionBiasAdd", "ConvolutionBiasAdd", "0");
    generate_file("ConvolutionBiasBackpropFiltersBias", "ConvolutionBiasBackpropFiltersBias", "0");
    generate_file("Cos", "Cos", "0");
    generate_file("Cosh", "Cosh", "0");
    generate_file("CropAndResize", "CropAndResize", "0");
    generate_file("CrossEntropy", "CrossEntropy", "0");
    generate_file("CrossEntropyBackprop", "CrossEntropyBackprop", "0");
    generate_file("CumSum", "CumSum", "0");
    generate_file("DeformableConvolution", "DeformableConvolution", "1");
    generate_file("DeformablePSROIPooling", "DeformablePSROIPooling", "1");
    generate_file("DepthToSpace", "DepthToSpace", "0");
    generate_file("Dequantize", "Dequantize", "0");
    generate_file("DetectionOutput", "DetectionOutput", "0");
    generate_file("Divide", "Divide", "0");
    generate_file("Divide", "Divide", "1");
    generate_file("Dot", "Dot", "0");
    generate_file("DynBroadcast", "DynBroadcast", "0");
    generate_file("DynPad", "DynPad", "0");
    generate_file("DynReplaceSlice", "DynReplaceSlice", "0");
    generate_file("DynSlice", "DynSlice", "0");
    generate_file("Elu", "Elu", "0");
    generate_file("EmbeddingBagOffsetsSum", "EmbeddingBagOffsetsSum", "3");
    generate_file("EmbeddingBagPackedSum", "EmbeddingBagPackedSum", "3");
    generate_file("EmbeddingLookup", "EmbeddingLookup", "0");
    generate_file("EmbeddingSegmentsSum", "EmbeddingSegmentsSum", "3");
    generate_file("Equal", "Equal", "0");
    generate_file("Equal", "Equal", "1");
    generate_file("Erf", "Erf", "0");
    generate_file("Exp", "Exp", "0");
    generate_file("ExtractImagePatches", "ExtractImagePatches", "3");
    generate_file("FakeQuantize", "FakeQuantize", "0");
    generate_file("Floor", "Floor", "0");
    generate_file("FloorMod", "FloorMod", "1");
    generate_file("GRN", "GRN", "0");
    generate_file("GRUCell", "GRUCell", "3");
    generate_file("Gather", "Gather", "0");
    generate_file("Gather", "Gather", "1");
    generate_file("GatherND", "GatherND", "0");
    generate_file("GatherTree", "GatherTree", "1");
    generate_file("Gelu", "Gelu", "0");
    generate_file("GeluBackpropFactor", "GeluBackpropFactor", "0");
    generate_file("Gemm", "Gemm", "0");
    generate_file("GenerateMask", "GenerateMask", "0");
    generate_file("Greater", "Greater", "0");
    generate_file("Greater", "Greater", "1");
    generate_file("GreaterEq", "GreaterEq", "0");
    generate_file("GreaterEqual", "GreaterEqual", "1");
    generate_file("GroupConvolution", "GroupConvolution", "0");
    generate_file("GroupConvolution", "GroupConvolution", "1");
    generate_file("GroupConvolutionBackpropData", "GroupConvolutionBackpropData", "0");
    generate_file("GroupConvolutionBackpropData", "GroupConvolutionBackpropData", "1");
    generate_file("GroupConvolutionBackpropFilters", "GroupConvolutionBackpropFilters", "0");
    generate_file("HardSigmoid", "HardSigmoid", "0");
    generate_file("Interpolate", "Interpolate", "0");
    generate_file("Interpolate", "Interpolate", "3");
    generate_file("LRN", "LRN", "0");
    generate_file("LSTMCell", "LSTMCell", "0");
    generate_file("LSTMSequence", "LSTMSequence", "0");
    generate_file("LayerNorm", "LayerNorm", "0");
    generate_file("LayerNormBackprop", "LayerNormBackprop", "0");
    generate_file("Less", "Less", "0");
    generate_file("Less", "Less", "1");
    generate_file("LessEq", "LessEq", "0");
    generate_file("LessEqual", "LessEqual", "1");
    generate_file("Log", "Log", "0");
    generate_file("LogicalAnd", "LogicalAnd", "1");
    generate_file("LogicalNot", "LogicalNot", "1");
    generate_file("LogicalOr", "LogicalOr", "1");
    generate_file("LogicalXor", "LogicalXor", "1");
    generate_file("MVN", "MVN", "0");
    generate_file("MatMul", "MatMul", "0");
    generate_file("Max", "Max", "0");
    generate_file("MaxPool", "MaxPool", "0");
    generate_file("MaxPool", "MaxPool", "1");
    generate_file("MaxPoolBackprop", "MaxPoolBackprop", "0");
    generate_file("Maximum", "Maximum", "0");
    generate_file("Maximum", "Maximum", "1");
    generate_file("Min", "Min", "0");
    generate_file("Minimum", "Minimum", "0");
    generate_file("Minimum", "Minimum", "1");
    generate_file("Mod", "Mod", "1");
    generate_file("Multiply", "Multiply", "0");
    generate_file("Multiply", "Multiply", "1");
    generate_file("Negative", "Negative", "0");
    generate_file("NonMaxSuppression", "NonMaxSuppression", "1");
    generate_file("NonMaxSuppression", "NonMaxSuppression", "3");
    generate_file("NonZero", "NonZero", "3");
    generate_file("NormalizeL2", "NormalizeL2", "0");
    generate_file("Not", "Not", "0");
    generate_file("NotEqual", "NotEqual", "0");
    generate_file("NotEqual", "NotEqual", "1");
    generate_file("OneHot", "OneHot", "0");
    generate_file("OneHot", "OneHot", "1");
    generate_file("Or", "Or", "0");
    generate_file("PRelu", "PRelu", "0");
    generate_file("PSROIPooling", "PSROIPooling", "0");
    generate_file("Pad", "Pad", "0");
    generate_file("Pad", "Pad", "1");
    generate_file("Parameter", "Parameter", "0");
    generate_file("PartialSlice", "PartialSlice", "0");
    generate_file("PartialSliceBackprop", "PartialSliceBackprop", "0");
    generate_file("Passthrough", "Passthrough", "0");
    generate_file("Power", "Power", "0");
    generate_file("Power", "Power", "1");
    generate_file("PriorBox", "PriorBox", "0");
    generate_file("PriorBoxClustered", "PriorBoxClustered", "0");
    generate_file("Product", "Product", "0");
    generate_file("Proposal", "Proposal", "0");
    generate_file("Quantize", "Quantize", "0");
    generate_file("QuantizedConvolution", "QuantizedConvolution", "0");
    generate_file("QuantizedConvolutionBias", "QuantizedConvolutionBias", "0");
    generate_file("QuantizedConvolutionBiasAdd", "QuantizedConvolutionBiasAdd", "0");
    generate_file("QuantizedConvolutionBiasSignedAdd", "QuantizedConvolutionBiasSignedAdd", "0");
    generate_file("QuantizedConvolutionRelu", "QuantizedConvolutionRelu", "0");
    generate_file("QuantizedDot", "QuantizedDot", "0");
    generate_file("QuantizedDotBias", "QuantizedDotBias", "0");
    generate_file("RNNCell", "RNNCell", "0");
    generate_file("ROIAlign", "ROIAlign", "3");
    generate_file("ROIPooling", "ROIPooling", "0");
    generate_file("RandomUniform", "RandomUniform", "0");
    generate_file("Range", "Range", "0");
    generate_file("Recv", "Recv", "0");
    generate_file("ReduceLogicalAnd", "ReduceLogicalAnd", "1");
    generate_file("ReduceLogicalOr", "ReduceLogicalOr", "1");
    generate_file("ReduceMax", "ReduceMax", "1");
    generate_file("ReduceMean", "ReduceMean", "1");
    generate_file("ReduceMin", "ReduceMin", "1");
    generate_file("ReduceProd", "ReduceProd", "1");
    generate_file("ReduceSum", "ReduceSum", "1");
    generate_file("RegionYolo", "RegionYolo", "0");
    generate_file("Relu", "Relu", "0");
    generate_file("ReluBackprop", "ReluBackprop", "0");
    generate_file("ReorgYolo", "ReorgYolo", "0");
    generate_file("ReplaceSlice", "ReplaceSlice", "0");
    generate_file("Reshape", "Reshape", "0");
    generate_file("Reshape", "Reshape", "1");
    generate_file("Result", "Result", "0");
    generate_file("Reverse", "Reverse", "0");
    generate_file("Reverse", "Reverse", "1");
    generate_file("ReverseSequence", "ReverseSequence", "0");
    generate_file("Round", "Round", "0");
    generate_file("ScalarConstantLike", "ScalarConstantLike", "0");
    generate_file("ScaleShift", "ScaleShift", "0");
    generate_file("ScatterAdd", "ScatterAdd", "0");
    generate_file("ScatterElementsUpdate", "ScatterElementsUpdate", "3");
    generate_file("ScatterND", "ScatterND", "0");
    generate_file("ScatterNDAdd", "ScatterNDAdd", "0");
    generate_file("ScatterUpdate", "ScatterUpdate", "3");
    generate_file("Select", "Select", "0");
    generate_file("Select", "Select", "1");
    generate_file("Selu", "Selu", "0");
    generate_file("Send", "Send", "0");
    generate_file("ShapeOf", "ShapeOf", "0");
    generate_file("ShapeOf", "ShapeOf", "3");
    generate_file("ShuffleChannels", "ShuffleChannels", "0");
    generate_file("Sigmoid", "Sigmoid", "0");
    generate_file("SigmoidBackprop", "SigmoidBackprop", "0");
    generate_file("Sign", "Sign", "0");
    generate_file("Sin", "Sin", "0");
    generate_file("Sinh", "Sinh", "0");
    generate_file("Slice", "Slice", "0");
    generate_file("Softmax", "Softmax", "0");
    generate_file("Softmax", "Softmax", "1");
    generate_file("SoftmaxCrossEntropy", "SoftmaxCrossEntropy", "0");
    generate_file("SoftmaxCrossEntropyBackprop", "SoftmaxCrossEntropyBackprop", "0");
    generate_file("SpaceToBatch", "SpaceToBatch", "1");
    generate_file("SpaceToDepth", "SpaceToDepth", "0");
    generate_file("Split", "Split", "0");
    generate_file("Split", "Split", "1");
    generate_file("Sqrt", "Sqrt", "0");
    generate_file("SquaredDifference", "SquaredDifference", "0");
    generate_file("Squeeze", "Squeeze", "0");
    generate_file("Stack", "Stack", "0");
    generate_file("StopGradient", "StopGradient", "0");
    generate_file("StridedSlice", "StridedSlice", "1");
    generate_file("Subtract", "Subtract", "0");
    generate_file("Subtract", "Subtract", "1");
    generate_file("Sum", "Sum", "0");
    generate_file("Tan", "Tan", "0");
    generate_file("Tanh", "Tanh", "0");
    generate_file("TensorIterator", "TensorIterator", "0");
    generate_file("Tile", "Tile", "0");
    generate_file("TopK", "TopK", "0");
    generate_file("TopK", "TopK", "1");
    generate_file("TopK", "TopK", "3");
    generate_file("Transpose", "Transpose", "1");
    generate_file("Unsqueeze", "Unsqueeze", "0");
    generate_file("VariadicSplit", "VariadicSplit", "1");
    generate_file("Xor", "Xor", "0");
}

int main(int argc, char** argv)
{
    xxx();
    exit(0);
    const string cpath_flag{"--cpath"};
    string cpath;
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc = argv_vector.size();
    ::testing::InitGoogleTest(&argc, argv_vector.data());
    for (int i = 1; i < argc; i++)
    {
        if (cpath_flag == argv[i] && (++i) < argc)
        {
            cpath = argv[i];
        }
    }
    ngraph::runtime::Backend::set_backend_shared_library_search_directory(cpath);
#ifdef NGRAPH_MLIR_ENABLE
    // Initialize MLIR
    ngraph::runtime::ngmlir::initializeNGraphMLIR();
#endif

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
    // Setup embedded python interpreter and import numpy
    py::scoped_interpreter guard{};
    py::exec(R"(
import numpy as np
)",
             py::globals(),
             py::dict());
#endif

    int rc = RUN_ALL_TESTS();

    return rc;
}
