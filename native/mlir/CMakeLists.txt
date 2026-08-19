set(LLVM_TARGET_DEFINITIONS include/fpgai/Dialect/FPGAI/FPGAIDialect.td)
mlir_tablegen(FPGAIDialect.h.inc -gen-dialect-decls -dialect=fpgai)
mlir_tablegen(FPGAIDialect.cpp.inc -gen-dialect-defs -dialect=fpgai)
add_public_tablegen_target(FPGAIDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/fpgai/Dialect/FPGAI/FPGAIOps.td)
mlir_tablegen(FPGAIOps.h.inc -gen-op-decls)
mlir_tablegen(FPGAIOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FPGAIOpsIncGen)

add_mlir_dialect_library(FPGAIMLIRDialect
  lib/Dialect/FPGAI/FPGAIDialect.cpp
  DEPENDS FPGAIDialectIncGen FPGAIOpsIncGen
  LINK_LIBS PUBLIC MLIRIR
)
target_include_directories(FPGAIMLIRDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)
