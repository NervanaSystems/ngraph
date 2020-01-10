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

// Add new dialect ops with lowering support to this file
#ifndef MLIR_OP
#define MLIR_OP
#endif

#ifndef MLIR_LAST_OP
#define MLIR_LAST_OP(OP, INPLACE) MLIR_OP(OP, INPLACE)
#endif

        /* op name */         /* in-place safe ? */ 
MLIR_OP(NGAddOp             , true                  )
MLIR_OP(NGArgMaxRedOp       , false                 )
MLIR_OP(NGArgMinRedOp       , false                 )
MLIR_OP(NGConcatOp          , true                  )
MLIR_OP(NGConvolutionOp     , false                 )
MLIR_OP(NGDivOp             , true                  )
MLIR_OP(NGDotOp             , false                 )
MLIR_OP(NGGatherOp          , false                 )
MLIR_OP(NGGreaterOp         , true                  )
MLIR_OP(NGLessOp            , true                  )
MLIR_OP(NGGreaterEqOp       , true                  )
MLIR_OP(NGLessEqOp          , true                  )
MLIR_OP(NGEqOp              , true                  )
MLIR_OP(NGNotEqOp           , true                  )
MLIR_OP(NGMulOp             , true                  )
MLIR_OP(NGMaxOp             , true                  )
MLIR_OP(NGMinOp             , true                  )
MLIR_OP(NGNegOp             , true                  )
MLIR_OP(NGReluOp            , true                  )
MLIR_OP(NGSubOp             , true                  )
MLIR_LAST_OP(NGReturnOp     , false                 )

#undef MLIR_OP
#undef MLIR_LAST_OP
