#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // end namespace mlir
