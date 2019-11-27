// RUN: ngraph-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: ngraph-opt %s | ngraph-opt | FileCheck %s

// These tests verify the parser, builder and printer of element-wise binary ops.

// CHECK-LABEL: func @squeeze
func @squeeze(%arg0: !ng.tensor<2x1x2x1xf32>, %arg1: !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{[0-9]+}} = "ng.squeeze"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x1x2x1xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32>
  %0 = "ng.squeeze"(%arg0, %arg1) : (!ng.tensor<2x1x2x1xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @unsqueeze
func @unsqueeze(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2xi64>) -> !ng.tensor<2x1x2x1xf32> {
  // CHECK: %{{[0-9]+}} = "ng.unsqueeze"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x1x2x1xf32>
  %0 = "ng.unsqueeze"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x1x2x1xf32>
  "ng.return"(%0) : (!ng.tensor<2x1x2x1xf32>) -> ()
}

// -----

// CHECK-LABEL: func @sqrddiff
func @sqrddiff(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{[0-9]+}} = "ng.sqrdDiff"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %0 = "ng.sqrdDiff"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @split
func @split(%arg0: !ng.tensor<2x2x16xf32>) -> !ng.tensor<2x2x4xf32> {
  // CHECK: %{{[0-9]+}}:4 = "ng.split"(%{{.*}}) {axis = 2 : i64, numSplits = [4, 4, 4, 4]} : (!ng.tensor<2x2x16xf32>) -> (!ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>)
  %0:4= "ng.split"(%arg0) {axis = 2, numSplits = [4, 4, 4, 4]} 
        : (!ng.tensor<2x2x16xf32>) -> (!ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>, !ng.tensor<2x2x4xf32>)
  "ng.return"(%0#0) : (!ng.tensor<2x2x4xf32>) -> ()
}

// -----

// CHECK-LABEL: func @spaceToDepth
func @spaceToDepth(%arg0: !ng.tensor<1x4x16x16xf32>) -> !ng.tensor<1x64x4x4xf32> {
  // CHECK: %{{[0-9]+}} = "ng.spaceToDepth"(%{{.*}}) {blockSize = 4 : i64, mode = 0 : i32} : (!ng.tensor<1x4x16x16xf32>) -> !ng.tensor<1x64x4x4xf32>
  %0 = "ng.spaceToDepth"(%arg0) {blockSize = 4, mode = 0 : i32} : (!ng.tensor<1x4x16x16xf32>) -> (!ng.tensor<1x64x4x4xf32>)
  "ng.return"(%0) : (!ng.tensor<1x64x4x4xf32>) -> ()
}

// -----

// CHECK-LABEL: func @shuffleChannels
func @shuffleChannels(%arg0: !ng.tensor<1x16x16x16xf32>) -> !ng.tensor<1x16x16x16xf32> {
  // CHECK: %{{[0-9]+}} = "ng.shuffleChannels"(%{{.*}}) {axis = 1 : i64, groups = 4 : i64} : (!ng.tensor<1x16x16x16xf32>) -> !ng.tensor<1x16x16x16xf32>
  %0 = "ng.shuffleChannels"(%arg0) {axis = 1 : i64, groups = 4 : i64} : (!ng.tensor<1x16x16x16xf32>) -> !ng.tensor<1x16x16x16xf32>
  "ng.return"(%0) : (!ng.tensor<1x16x16x16xf32>) -> ()
}

// -----

// CHECK-LABEL: func @scaleShift
func @scaleShift(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>, %arg2: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{[0-9]+}} = "ng.scaleShift"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>

  %0 = "ng.scaleShift"(%arg0, %arg1, %arg2) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @prelu
func @prelu(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{[0-9]+}} = "ng.prelu"(%{{.*}}, %{{.*}})  : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %0 = "ng.prelu"(%arg0, %arg1) {} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> (!ng.tensor<2x2xf32>)
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @normalizeL2
func @normalizeL2(%arg0: !ng.tensor<1x2x3x4xf32>, %arg1: !ng.tensor<3x!ng.i64>) -> !ng.tensor<1x2x3x4xf32> {
  // CHECK: %{{[0-9]+}} = "ng.normalizeL2"(%{{.*}}, %{{.*}}) {eps = {{0.[0-9]+}} : f32, epsMode = 0 : i32} : (!ng.tensor<1x2x3x4xf32>, !ng.tensor<3x!ng.i64>) -> !ng.tensor<1x2x3x4xf32>
  %0 = "ng.normalizeL2"(%arg0, %arg1) {eps = 0.01 : f32, epsMode = 0 : i32} : (!ng.tensor<1x2x3x4xf32> , !ng.tensor<3x!ng.i64>) -> !ng.tensor<1x2x3x4xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x3x4xf32>) -> ()
}

// -----

// CHECK-LABEL: func @mvn
func @mvn(%arg0: !ng.tensor<1x2x5xf32>) -> !ng.tensor<1x2x5xf32> {
  // CHECK: %{{[0-9]+}} = "ng.mvn"(%{{.*}}) {normalizeVariance = false} : (!ng.tensor<1x2x5xf32>) -> !ng.tensor<1x2x5xf32>
  %0 = "ng.mvn"(%arg0) {normalizeVariance = false} : (!ng.tensor<1x2x5xf32>) -> !ng.tensor<1x2x5xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x5xf32>) -> ()
}

// -----

// CHECK-LABEL: func @matmul
func @matmul(%arg0: !ng.tensor<2x5xf32>, %arg1: !ng.tensor<2x5xf32>) -> !ng.tensor<2x5xf32> {
  // CHECK: %{{[0-9]+}} = "ng.matmul"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x5xf32>, !ng.tensor<2x5xf32>) -> !ng.tensor<2x5xf32>
  %0 = "ng.matmul"(%arg0, %arg1) : (!ng.tensor<2x5xf32>, !ng.tensor<2x5xf32>) -> !ng.tensor<2x5xf32>
 
  "ng.return"(%0) : (!ng.tensor<2x5xf32>) -> ()
} 

// ------

// CHECK-LABEL: func @layernorm
func @layernorm(%arg0: !ng.tensor<2x4xf32>, %arg1: !ng.tensor<4xf32>, %arg2: !ng.tensor<4xf32>) -> !ng.tensor<2x4xf32> {
  // CHECK %{{[0-9]+}}:3 = "ng.layernorm"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ng.tensor<2x4xf32>, !ng.tensor<4xf32>, !ng.tensor<4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  %0:3 = "ng.layernorm"(%arg0, %arg1, %arg2) 
         : (!ng.tensor<2x4xf32>, !ng.tensor<4xf32>, !ng.tensor<4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  // CHECK %{{[0-9]+}}:3 = "ng.layernorm"(%{{.*}}) : (!ng.tensor<2x4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  %1:3 = "ng.layernorm"(%arg0) 
           : (!ng.tensor<2x4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  "ng.return"(%0#0) : (!ng.tensor<2x4xf32>) -> ()
} 

// -----

// CHECK-LABEL: func @layernormBackprop
func @layernormBackprop(%arg0: !ng.tensor<2x4xf32>, %arg1: !ng.tensor<2x4xf32>, %arg2: !ng.tensor<2xf32>, %arg3: !ng.tensor<2xf32>, %arg4: !ng.tensor<4xf32>) -> !ng.tensor<2x4xf32> {
  // CHECK: %{{[0-9]+}}:3 = "ng.layernormBackprop"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!ng.tensor<2x4xf32>, !ng.tensor<2x4xf32>, !ng.tensor<4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  %0:3 = "ng.layernormBackprop"(%arg0, %arg1, %arg4, %arg2, %arg3) 
          : (!ng.tensor<2x4xf32>, !ng.tensor<2x4xf32>, !ng.tensor<4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)

  // CHECK: %{{[0-9]+}}:3 = "ng.layernormBackprop"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ng.tensor<2x4xf32>, !ng.tensor<2x4xf32>, !ng.tensor<4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)
  %1:3 = "ng.layernormBackprop"(%arg0, %arg1, %arg4) 
          : (!ng.tensor<2x4xf32>, !ng.tensor<2x4xf32>, !ng.tensor<4xf32>) -> (!ng.tensor<2x4xf32>, !ng.tensor<2xf32>, !ng.tensor<2xf32>)        
  "ng.return"(%0#0) : (!ng.tensor<2x4xf32>) -> ()
} 

// -----

// CHECK-LABEL: func @hardSigmoid
func @hardSigmoid(%arg0: !ng.tensor<2x7xf32>) -> !ng.tensor<2x7xf32>
{
  %0 = "ng.hardSigmoid"(%arg0) {alpha = 0.125 : f32, beta = 0.642 : f32} : (!ng.tensor<2x7xf32>) -> !ng.tensor<2x7xf32>
  "ng.return"(%0) : (!ng.tensor<2x7xf32>) -> ()
}

// -----

// CHECK-LABEL: func @gemm
func @gemm(%arg0: !ng.tensor<3x6xf32>, %arg1: !ng.tensor<6x4xf32>, %arg2: !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32> {
  // CHECK: %{{[0-9]+}} = "ng.gemm"(%{{.*}}, %{{.*}}, %{{.*}}) : (!ng.tensor<3x6xf32>, !ng.tensor<6x4xf32>, !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32>
  %0 = "ng.gemm"(%arg0, %arg1, %arg2) : (!ng.tensor<3x6xf32>, !ng.tensor<6x4xf32>, !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32>
  
  "ng.return"(%0) : (!ng.tensor<3x4xf32>) -> ()
} 

// -----

// CHECK-LABEL: func @groupConv
func @groupConv(%arg0: !ng.tensor<1x4x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.groupConv"(%{{.*}}, %{{.*}}) {groups = 2 : i64, padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  %0 = "ng.groupConv"(%arg0, %arg1) {groups=2 : i64, padAbove=[0,0], padBelow=[0,0], strides=[1, 1]} 
        : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @groupConvTranspose
func @groupConvTranspose(%arg0: !ng.tensor<1x4x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.groupConvTranspose"(%{{.*}}, %{{.*}}) {groups = 2 : i64, outputPad = [1, 1], outputShape = [], padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  %0 = "ng.groupConvTranspose"(%arg0, %arg1) {groups=2 : i64, padAbove=[0,0], padBelow=[0,0], outputPad=[1,1], outputShape=[], strides=[1, 1]}
       : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}

// -----

// CHECK-LABEL: func @grn
func @grn(%arg0: !ng.tensor<1x2x3x4xf32>) -> !ng.tensor<1x2x3x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.grn"(%{{.*}}) {bias = {{.*}} : f32} : (!ng.tensor<1x2x3x4xf32>) -> !ng.tensor<1x2x3x4xf32>
  %0 = "ng.grn"(%arg0) {bias = 0.1 : f32 } : (!ng.tensor<1x2x3x4xf32>) -> !ng.tensor<1x2x3x4xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x3x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @clamp
func @clamp(%arg0: !ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.clamp"(%{{.*}}) {max = {{.*}} : f64, min = {{.*}} : f64} : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  %0 = "ng.clamp"(%arg0) {max = 20.0 : f64, min = 10.0 : f64} : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  "ng.return"(%0) : (!ng.tensor<4x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @gelu
func @gelu(%arg0: !ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.gelu"({{.*}}) : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  %0 = "ng.gelu"(%arg0) : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  "ng.return"(%0) : (!ng.tensor<4x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @geluBackpropFactor
func @geluBackpropFactor(%arg0: !ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.geluBackpropFactor"({{.*}}) : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  %0 = "ng.geluBackpropFactor"(%arg0) : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  "ng.return"(%0) : (!ng.tensor<4x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @elu
func @elu(%arg0: !ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.elu"({{.*}}) {alpha = {{.*}} : f64} : (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  %0 = "ng.elu"(%arg0) {alpha = 0.001 : f64}: (!ng.tensor<4x4xf32>) -> !ng.tensor<4x4xf32>
  "ng.return"(%0) : (!ng.tensor<4x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @fakeQuant
func @fakeQuant(%arg0: !ng.tensor<1x2x3x4xf32>, %arg1: !ng.tensor<1xf32>,
                %arg2: !ng.tensor<1xf32>, %arg3: !ng.tensor<1xf32>, %arg4: !ng.tensor<1xf32>) -> !ng.tensor<1x2x3x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.fakeQuant"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {levels = 4 : i64} : (!ng.tensor<1x2x3x4xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>) -> !ng.tensor<1x2x3x4xf32>
  %0 = "ng.fakeQuant"(%arg0, %arg1, %arg2, %arg3, %arg4) {levels = 4 : i64} 
       : (!ng.tensor<1x2x3x4xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>, !ng.tensor<1xf32>) -> !ng.tensor<1x2x3x4xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x3x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @depthToSpace
func @depthToSpace(%arg0: !ng.tensor<1x8x2x2xf32>) -> !ng.tensor<1x2x4x4xf32>
{
  //CHECK: %{{[0-9]+}} = "ng.depthToSpace"(%{{.*}}) {blockSize = 2 : i64, mode = 0 : i32} : (!ng.tensor<1x8x2x2xf32>) -> !ng.tensor<1x2x4x4xf32>
  %0 = "ng.depthToSpace"(%arg0) {blockSize = 2 : i64, mode = 0 : i32} : (!ng.tensor<1x8x2x2xf32>) -> !ng.tensor<1x2x4x4xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x4x4xf32>) -> ()
}

// -----

//CHECK-LABEL: func @convBias
func @convBias(%arg0: !ng.tensor<1x3x2xf32>, %arg1: !ng.tensor<2x3x1xf32>, %arg2: !ng.tensor<2xf32>) -> (!ng.tensor<1x2x2xf32>)
{
  //CHECK: %{{[0-9]+}} = "ng.convBias"(%{{.*}}, %{{.*}}, %{{.*}}) {padAbove = [0], padBelow = [0], strides = [1]} : (!ng.tensor<1x3x2xf32>, !ng.tensor<2x3x1xf32>, !ng.tensor<2xf32>) -> !ng.tensor<1x2x2xf32>
  %0 = "ng.convBias"(%arg0, %arg1, %arg2) {padAbove=[0], padBelow=[0], strides=[1]} 
       : (!ng.tensor<1x3x2xf32>, !ng.tensor<2x3x1xf32>, !ng.tensor<2xf32>) -> !ng.tensor<1x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2xf32>) -> ()
}

// -----

//CHECK-LABEL: func @convBiasBackprop
func @convBiasBackprop(%arg0: !ng.tensor<1x3x2x2xf32>, %arg1: !ng.tensor<1x2x2x2xf32>) -> (!ng.tensor<12x3x1x1xf32>)
{
  //CHECK: %{{[0-9]+}}:2 = "ng.convBiasBackpropFiltersBias"(%{{.*}}, %{{.*}}) {biasShape = [2], filtersShape = [2, 3, 1, 1], padAbove = [0], padBelow = [0], strides = [1]} : (!ng.tensor<1x3x2x2xf32>, !ng.tensor<1x2x2x2xf32>) -> (!ng.tensor<2x3x1x1xf32>, !ng.tensor<2xf32>)
  %0:2 = "ng.convBiasBackpropFiltersBias"(%arg0, %arg1) {biasShape=[2], filtersShape=[2, 3, 1, 1], padAbove=[0], padBelow=[0], strides=[1]} 
          : (!ng.tensor<1x3x2x2xf32>, !ng.tensor<1x2x2x2xf32>) -> (!ng.tensor<2x3x1x1xf32>, !ng.tensor<2xf32>)
  "ng.return"(%0#0) : (!ng.tensor<2x3x1x1xf32>) -> ()
}

// -----

//CHECK-LABEL: func @convBiasAdd
func @convBiasAdd(%arg0: !ng.tensor<1x3x2x2xf32>, %arg1: !ng.tensor<2x3x1x1xf32>, %arg2: !ng.tensor<2xf32>, %arg3: !ng.tensor<1x2x2x2xf32>) -> !ng.tensor<1x2x2x2xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.convBiasAdd"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x3x2x2xf32>, !ng.tensor<2x3x1x1xf32>, !ng.tensor<2xf32>, !ng.tensor<1x2x2x2xf32>) -> !ng.tensor<1x2x2x2xf32>
  %0 = "ng.convBiasAdd" (%arg0, %arg1, %arg2, %arg3) {padAbove=[0, 0], padBelow=[0, 0], strides=[1, 1]} 
       : (!ng.tensor<1x3x2x2xf32>, !ng.tensor<2x3x1x1xf32>, !ng.tensor<2xf32>, !ng.tensor<1x2x2x2xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}

// -----

//CHECK-LABEL: func @rnnCell
func @rnnCell(%arg0: !ng.tensor<2x3xf32>, %arg1: !ng.tensor<2x3xf32>, %arg2: !ng.tensor<2x3xf32>, %arg3: !ng.tensor<3x3xf32>, %arg4: !ng.tensor<3xf32>) -> !ng.tensor<2x3xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.rnnCell"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hiddenSize = 3 : i64} : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<3x3xf32>) -> !ng.tensor<2x3xf32>
  %0 = "ng.rnnCell" (%arg0, %arg1, %arg2, %arg3) {hiddenSize = 3 : i64} 
       : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<3x3xf32>) -> !ng.tensor<2x3xf32>

  // CHECK: %{{[0-9]+}} = "ng.rnnCell"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hiddenSize = 3 : i64} : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<3x3xf32>, !ng.tensor<3xf32>) -> !ng.tensor<2x3xf32>
  %1 = "ng.rnnCell" (%arg0, %arg1, %arg2, %arg3, %arg4) {hiddenSize = 3 : i64} 
       : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<3x3xf32>, !ng.tensor<3xf32>) -> !ng.tensor<2x3xf32>

  "ng.return" (%0) : (!ng.tensor<2x3xf32>)->()

}

// -----

//CHECK-LABEL: func @lstmCell
func @lstmCell(%arg0: !ng.tensor<2x3xf32>, %arg1: !ng.tensor<2x3xf32>, %arg2: !ng.tensor<2x3xf32>, %arg3: !ng.tensor<12x3xf32>, %arg4: !ng.tensor<12x3xf32>) -> !ng.tensor<2x3xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.lstmCell"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hiddenSize = 3 : i64} : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<12x3xf32>, !ng.tensor<12x3xf32>) -> !ng.tensor<2x3xf32>
  %0 = "ng.lstmCell" (%arg0, %arg1, %arg2, %arg3, %arg4) {hiddenSize = 3 : i64} 
       : (!ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<12x3xf32>, !ng.tensor<12x3xf32>) -> !ng.tensor<2x3xf32>
  "ng.return" (%0) : (!ng.tensor<2x3xf32>)->()
}

// -----

//CHECK-LABEL: func @gruCell
func @gruCell(%arg0: !ng.tensor<2x3xf32>, %arg1: !ng.tensor<9x3xf32>, %arg2: !ng.tensor<9x3xf32>, %arg3: !ng.tensor<2x3xf32>, %arg4: !ng.tensor<18xf32>) -> !ng.tensor<2x3xf32>
{
  // CHECK: %{{[0-9]+}} = "ng.gruCell"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {activations = ["sigmoid", "tanh"], activationsAlpha = [], clip = {{.*}} : f32, hiddenSize = 3 : i64} : (!ng.tensor<2x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<18xf32>) -> !ng.tensor<2x3xf32>
  %0 = "ng.gruCell" (%arg0, %arg1, %arg2, %arg3, %arg4) {activations=["sigmoid", "tanh"], activationsAlpha = [], clip = 2.88 : f32, hiddenSize = 3 : i64} 
       : (!ng.tensor<2x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<2x3xf32>, !ng.tensor<18xf32>) -> !ng.tensor<2x3xf32>

  // CHECK: %{{[0-9]+}} = "ng.gruCell"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {activations = ["sigmoid", "tanh"], activationsAlpha = [], clip = {{.*}} : f32, hiddenSize = 3 : i64} : (!ng.tensor<2x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<2x3xf32>) -> !ng.tensor<2x3xf32>
  %1 = "ng.gruCell" (%arg0, %arg1, %arg2, %arg3) {activations=["sigmoid", "tanh"], activationsAlpha = [], clip = 2.88 : f32, hiddenSize = 3 : i64} 
       : (!ng.tensor<2x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<9x3xf32>, !ng.tensor<2x3xf32>) -> !ng.tensor<2x3xf32>
  "ng.return" (%0) : (!ng.tensor<2x3xf32>) -> ()
}
