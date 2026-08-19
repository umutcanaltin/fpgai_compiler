#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "fpgai/Dialect/FPGAI/FPGAIDialect.h.inc"
#define GET_OP_CLASSES
#include "fpgai/Dialect/FPGAI/FPGAIOps.h.inc"
