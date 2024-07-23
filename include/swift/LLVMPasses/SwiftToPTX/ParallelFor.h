//===- ParallelFor.h - Lift parallel_for loops to NVPTX ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef SWIFT_LLVMPASSES_SWIFT2PTX_PARALLELFOR_H
#define SWIFT_LLVMPASSES_SWIFT2PTX_PARALLELFOR_H

#include "llvm/IR/PassManager.h"

namespace swift {

struct ParallelForPass : public llvm::PassInfoMixin<ParallelForPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &AM);
};

} // namespace swift

#endif // SWIFT_LLVMPASSES_SWIFT2PTX_PARALLELFOR_H

