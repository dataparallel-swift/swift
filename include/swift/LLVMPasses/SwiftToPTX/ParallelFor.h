//===- ParallelFor.h - Lift parallel_for loops to NVPTX ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_PARALLELFOR_H
#define LLVM_TRANSFORMS_PARALLELFOR_H

#include <llvm/IR/PassManager.h>

namespace llvm {

struct ParallelForPass : public PassInfoMixin<ParallelForPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_PARALLELFOR_H

