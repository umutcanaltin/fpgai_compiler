#include "fpgai/Dialect/FPGAI/FPGAIDialect.h"

#include "fpgai/Dialect/FPGAI/FPGAIDialect.cpp.inc"
#define GET_OP_CLASSES
#include "fpgai/Dialect/FPGAI/FPGAIOps.cpp.inc"

using namespace mlir;
using namespace mlir::fpgai;

void FPGAIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "fpgai/Dialect/FPGAI/FPGAIOps.cpp.inc"
      >();
}
