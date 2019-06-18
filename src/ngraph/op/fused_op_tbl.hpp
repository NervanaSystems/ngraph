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

// This collection contains one entry for each fused op.
//

NGRAPH_OP(Clamp, ngraph::op)
NGRAPH_OP(ConvolutionBias, ngraph::op)
NGRAPH_OP(ConvolutionBiasAdd, ngraph::op)
NGRAPH_OP(ConvolutionBiasBackpropFiltersBias, ngraph::op)
NGRAPH_OP(DepthToSpace, ngraph::op)
NGRAPH_OP(Elu, ngraph::op)
NGRAPH_OP(FakeQuantize, ngraph::op)
NGRAPH_OP(GRN, ngraph::op)
NGRAPH_OP(Gemm, ngraph::op)
NGRAPH_OP(GroupConvolution, ngraph::op)
NGRAPH_OP(GroupConvolutionTranspose, ngraph::op)
NGRAPH_OP(HardSigmoid, ngraph::op)
NGRAPH_OP(LeakyRelu, ngraph::op)
NGRAPH_OP(MVN, ngraph::op)
NGRAPH_OP(Normalize, ngraph::op)
NGRAPH_OP(PRelu, ngraph::op)
NGRAPH_OP(ScaleShift, ngraph::op)
NGRAPH_OP(SpaceToDepth, ngraph::op)
NGRAPH_OP(ShuffleChannels, ngraph::op)
NGRAPH_OP(SquaredDifference, ngraph::op)
NGRAPH_OP(Squeeze, ngraph::op)
NGRAPH_OP(Split, ngraph::op)
NGRAPH_OP(Unsqueeze, ngraph::op)
