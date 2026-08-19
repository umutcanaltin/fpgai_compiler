include "mlir/IR/OpBase.td"

def FPGAI_Dialect : Dialect {
  let name = "fpgai";
  let cppNamespace = "::mlir::fpgai";
  let summary = "FPGAI hardware, training, transport and implementation semantics";
  let description = [{
    The FPGAI dialect is the optional native MLIR representation of FPGAI's
    domain contracts. It complements, rather than replaces, FPGAI IR v2.
  }];
}

class FPGAI_Op<string mnemonic, list<Trait> traits = []> :
    Op<FPGAI_Dialect, mnemonic, traits>;
