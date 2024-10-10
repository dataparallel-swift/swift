//===- ParallelFor.cpp - Lift parallel_for loops to PTX -------------------===//
//
//===----------------------------------------------------------------------===//

#include "swift/LLVMPasses/SwiftToPTX/ParallelFor.h"
#include "swift/Demangling/Demangler.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

using namespace llvm;

#define DEBUG_TYPE "swift-to-ptx"

namespace {

static cl::opt<bool> KeepIntermediateFiles (
  "swift-to-ptx-keep-intermediate-files", cl::Hidden, cl::init(false),
  cl::desc("Keep intermediate files of swift-to-ptx pass"));

static cl::opt<StringRef> PTXASPath (
  "swift-to-ptx-ptxas-path", cl::Hidden, cl::init("/usr/local/cuda/bin/ptxas"),
  cl::desc("Path to the ptxas executable"));

static cl::opt<StringRef> TargetGPU (
  "swift-to-ptx-target-gpu", cl::Hidden, cl::init("sm_87"),   // default: Orin
  cl::desc("Target a specific GPU architecture in swift-to-ptx pass"));

static cl::opt<StringRef> TargetFeatures (
  "swift-to-ptx-target-attr", cl::Hidden, cl::init(""),  // default: +ptx75 (llvm-17 produces assembly using features from this version no matter what)
  cl::desc("Target specific attributes in swift-to-ptx pass"));

static const MemoryBufferRef parallel_for_kernel = MemoryBufferRef(R"KERNEL(
; ModuleID = '<parallel_for_kernel>'
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: argmemonly nofree nosync nounwind
define void @parallel_for(i64 %iterations, ptr nonnull %env) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #2
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
  %2 = mul i32 %1, %0
  %3 = sext i32 %2 to i64
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
  %6 = mul i32 %4, %1
  %7 = add i32 %5, %6
  %8 = sext i32 %7 to i64
  %9 = icmp slt i64 %8, %iterations
  br i1 %9, label %while1.top, label %while1.exit

while1.top:                                       ; preds = %entry, %while1.top
  %10 = phi i64 [ %11, %while1.top ], [ %8, %entry ]
  call void @body(i64 %10, ptr nonnull %env)
  %11 = add i64 %10, %3
  %12 = icmp slt i64 %11, %iterations
  br i1 %12, label %while1.top, label %while1.exit

while1.exit:                                      ; preds = %while1.top, %entry
  ret void
}

define internal void @body(i64 %0, ptr nonnull %1) {
  ret void
}

; Function Attrs: nofree nosync nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nofree nosync nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nofree nosync nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

; Function Attrs: nofree nosync nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: convergent nounwind
declare void @__assertfail(ptr noundef, ptr noundef, i32 noundef, ptr noundef, i64 noundef) #3
;                           │            │            │            │            ╰────── character size in bytes (must be 1)
;                           │            │            │            ╰─────────────────── function name string
;                           │            │            ╰──────────────────────────────── line number
;                           │            ╰───────────────────────────────────────────── file name
;                           ╰────────────────────────────────────────────────────────── message

attributes #0 = { argmemonly nofree nosync nounwind }
attributes #1 = { nofree nosync nounwind readnone }
attributes #2 = { nounwind readnone }
attributes #3 = { convergent nounwind }

!nvvm.annotations = !{!0}

!0 = !{ptr @parallel_for, !"kernel", i32 1}
)KERNEL", "<parallel_for_kernel>");


static const MemoryBufferRef host_support = MemoryBufferRef(R"HOST(
; ModuleID = '<host_support>'
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%TSP = type <{ ptr }>
%Ts5Int32V = type <{ i32 }>
%Ts13OpaquePointerV = type <{ ptr }>
%Ts13OpaquePointerVSg = type <{ [8 x i8] }>
%T10SwiftToPTX17ParallelForKernelV = type <{ %TSP, %Ts13OpaquePointerVSg, %Ts13OpaquePointerV, %Ts5Int32V, %Ts5Int32V }>

; SwiftToPTX.CachingHostAllocator.alloc(Swift.Int) -> Swift.UnsafeMutableRawPointer
declare swiftcc ptr @"$s10SwiftToPTX20CachingHostAllocatorV5allocySvSiF"(i64, ptr, ptr, ptr) local_unnamed_addr #0
;                                                                         │    ╰────┬────╯
;                                                                         │         ╰───────── SwiftToPTX.CachingHostAllocator
;                                                                         ╰─────────────────── size in bytes

; SwiftToPTX.CachingHostAllocator.free(Swift.UnsafeMutableRawPointer, SwiftToPTX.Event) -> ()
declare swiftcc void @"$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF"(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0
;                                                                                   │    │    ╰────┬────╯
;                                                                                   │    │         ╰───────── SwiftToPTX.CachingHostAllocator
;                                                                                   │    ╰─────────────────── ready event
;                                                                                   ╰──────────────────────── pointer to free

; SwiftToPTX.parallel_for(iterations: Swift.Int, context: SwiftToPTX.Context, allocator: SwiftToPTX.CachingHostAllocator, stream: SwiftToPTX.Stream, _: (Swift.Int) -> ()) -> SwiftToPTX.Event
declare swiftcc ptr @"$s10SwiftToPTX12parallel_for10iterations7context9allocator6stream_AA5EventCSi_AA7ContextVAA20CachingHostAllocatorVAA6StreamVySiXEtF"(i64, ptr, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0
;                                                                                                                                                           │     ╰────┬────╯   ╰────┬────╯    │    │    ╰──── closure environment
;                                                                                                                                                           │          │             │         │    ╰───────── body of the parallel_for loop
;                                                                                                                                                           │          │             │         ╰────────────── execution stream
;                                                                                                                                                           │          │             ╰──────────────────────── SwiftToPTX.CachingHostAllocator
;                                                                                                                                                           │          ╰────────────────────────────────────── SwiftToPTX.Context
;                                                                                                                                                           ╰───────────────────────────────────────────────── iterations

; SwiftToPTX.launch_parallel_for(iterations: Swift.Int, kernel: inout SwiftToPTX.ParallelForKernel, env: Swift.UnsafeMutableRawPointer, context: SwiftToPTX.Context, stream: SwiftToPTX.Stream) -> SwiftToPTX.Event
declare swiftcc ptr @"$s10SwiftToPTX19launch_parallel_for10iterations6kernel3env7context6streamAA5EventCSi_AA17ParallelForKernelVzSvAA7ContextVAA6StreamVtF"(i64, ptr nocapture dereferenceable(32), ptr, ptr, i64, i64, ptr) local_unnamed_addr #0
;                                                                                                                                                             │    │                                  │    ╰────┬────╯    ╰──── execution stream
;                                                                                                                                                             │    │                                  │         ╰────────────── SwiftToPTX.Context
;                                                                                                                                                             │    │                                  ╰──────────────────────── closure environment (updated to be GPU accessible)
;                                                                                                                                                             │    ╰─────────────────────────────────────────────────────────── SwiftToPTX.ParallelForKernel struct
;                                                                                                                                                             ╰──────────────────────────────────────────────────────────────── iterations

attributes #0 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }
attributes #1 = { sspreq "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 1}
!6 = !{i32 4, !"Objective-C Garbage Collection", i32 84477696}
!7 = !{i32 1, !"Swift Version", i32 7}
)HOST", "<host_support>");


// Mapping from llvm intrinsic name to corresponding libdevice function name (if
// it exists).
//
// https://llvm.org/docs/LangRef.html#standard-c-c-library-intrinsics
// https://docs.nvidia.com/cuda/libdevice-users-guide/index.html
//
// TODO: This is only the list of intrinsics supported by libdevice; some
// remapping of arguments might be necessary in order to match calling
// conventions, but this is not yet checked.
//
// TODO: Missing operations might be supportable by mapping to other
// intrinsics/sequences.
//
const StringMap<StringRef> libdeviceFunctions =
  {{"llvm.abs.i64",         "__nv_llabs"}
  ,{"llvm.abs.i32",         "__nv_abs"}
  ,{"llvm.smax.i64",        "__nv_llmax"}
  ,{"llvm.smax.i32",        "__nv_max"}
  ,{"llvm.smin.i64",        "__nv_llmin"}
  ,{"llvm.smin.i32",        "__nv_min"}
  ,{"llvm.umax.i64",        "__nv_ullmax"}
  ,{"llvm.umax.i32",        "__nv_umax"}
  ,{"llvm.umin.i64",        "__nv_ullmin"}
  ,{"llvm.umin.i32",        "__nv_umin"}
  /* ,{"llvm.memcpy",          "" */
  /* ,{"llvm.memcpy.inline",   "" */
  /* ,{"llvm.memmove",         "" */
  /* ,{"llvm.memmove.inline",  "" */
  /* ,{"llvm.memset",          "" */
  /* ,{"llvm.memset.inline",   "" */
  ,{"llvm.sqrt.f64",        "__nv_sqrt"},   {"sqrt",  "__nv_sqrt"}
  ,{"llvm.sqrt.f32",        "__nv_sqrtf"},  {"sqrtf", "__nv_sqrtf"}
  ,{"llvm.powi.f64.i32",    "__nv_powi"}
  ,{"llvm.powi.f32.i32",    "__nv_powif"}
  ,{"llvm.sin.f64",         "__nv_sin"},    {"sin",   "__nv_sin"}
  ,{"llvm.sin.f32",         "__nv_sinf"},   {"sinf",  "__nv_sinf"}
  ,{"llvm.cos.f64",         "__nv_cos"},    {"cos",   "__nv_cos"}
  ,{"llvm.cos.f32",         "__nv_cosf"},   {"cosf",  "__nv_cosf"}
  ,{"llvm.pow.f64",         "__nv_pow"}
  ,{"llvm.pow.f32",         "__nv_powf"}
  ,{"llvm.exp.f64",         "__nv_exp"}
  ,{"llvm.exp.f32",         "__nv_expf"}
  ,{"llvm.exp2.f64",        "__nv_exp2"}
  ,{"llvm.exp2.f32",        "__nv_exp2f"}
  ,{"llvm.exp10.f64",       "__nv_exp10"}
  ,{"llvm.exp10.f32",       "__nv_exp10f"}
  ,{"llvm.ldexp.f64.i32",   "__nv_ldexp"}
  ,{"llvm.ldexp.f32.i32",   "__nv_ldexp"}
  ,{"llvm.frexp.f64.i32",   "__nv_frexp"}
  ,{"llvm.frexp.f32.i32",   "__nv_frexpf"}
  ,{"llvm.log.f64",         "__nv_log"}
  ,{"llvm.log.f32",         "__nv_logf"}
  ,{"llvm.log10.f64",       "__nv_log10"}
  ,{"llvm.log10.f32",       "__nv_log10f"}
  ,{"llvm.log2.f64",        "__nv_log2"}
  ,{"llvm.log2.f32",        "__nv_log2f"}
  ,{"llvm.fma.f64",         "__nv_fma"}
  ,{"llvm.fma.f32",         "__nv_fmaf"}
  ,{"llvm.fabs.f64",        "__nv_fabs"}
  ,{"llvm.fabs.f32",        "__nv_fabsf"}
  ,{"llvm.minnum.f64",      "__nv_fmin"}
  ,{"llvm.minnum.f32",      "__nv_fminf"}
  ,{"llvm.maxnum.f64",      "__nv_fmax"}
  ,{"llvm.maxnum.f32",      "__nv_fmaxf"}
  /* ,{"llvm.minimum.*",       ""} */
  /* ,{"llvm.maximum.*",       ""} */
  ,{"llvm.copysign.f64",    "__nv_copysign"}
  ,{"llvm.copysign.f32",    "__nv_copysignf"}
  ,{"llvm.floor.f64",       "__nv_floor"}
  ,{"llvm.floor.f32",       "__nv_floorf"}
  ,{"llvm.ceiling.f64",     "__nv_ceiling"}
  ,{"llvm.ceiling.f32",     "__nv_ceilingf"}
  ,{"llvm.trunc.f64",       "__nv_trunc"}
  ,{"llvm.trunc.f32",       "__nv_truncf"}
  ,{"llvm.rint.f64",        "__nv_rint"}
  ,{"llvm.rint.f32",        "__nv_rintf"}
  ,{"llvm.nearbyint.f64",   "__nv_nearbyint"}
  ,{"llvm.nearbyint.f32",   "__nv_nearbyintf"}
  ,{"llvm.round.f64",       "__nv_round"}
  ,{"llvm.round.f32",       "__nv_roundf"}
  /* ,{"llvm.roundeven.*",     ""} */
  /* ,{"llvm.lround.*",        ""} */
  ,{"llvm.llround.i64.f64", "__nv_llround"}
  ,{"llvm.llround.i64.f32", "__nv_llroundf"}
  /* ,{"llvm.lrint.*",         ""} */
  ,{"llvm.llrint.i64.f64",  "__nv_llrint"}
  ,{"llvm.llrint.i64.f32",  "__nv_llrintf"}
  /* ,{"llvm.bitreverse.*",    ""} */
  /* ,{"llvm.bswap.*",         ""} */
  ,{"llvm.ctpop.i64",       "__nv_popcll"}
  ,{"llvm.ctpop.i32",       "__nv_popc"}
  ,{"llvm.ctlz.i64",        "__nv_clzll"}
  ,{"llvm.ctlz.i32",        "__nv_clz"}
  /* ,{"llvm.cttz.*",          ""} */
  /* ,{"llvm.fshl.*",          ""} */
  /* ,{"llvm.fshr.*",          ""} */
  };


// Return the location of the libdevice bitcode file.
//
// XXX: Update the path to the actual location of the libdevice module, which is
// typically located at '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc' for a
// usual CUDA installation, but might be located elsewhere for clang.
//
const Twine LocateLibdeviceFile()
{
  return "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc";
}


// Load the libdevice module from file. This implements many common mathematical
// functions, and we will lower llvm intrinsics to these operations, e.g.:
//
//   llvm.sin.f32  -->  __nv_sinf
//
std::unique_ptr<Module> LoadLibdeviceModule(SMDiagnostic &Err, LLVMContext &Context)
{
  auto FilePath = LocateLibdeviceFile();
  auto FileOrError = MemoryBuffer::getFile(FilePath);
  if (std::error_code EC = FileOrError.getError()) {
    report_fatal_error("Could not open libdevice module", false);
  }
  MemoryBufferRef Buffer = MemoryBufferRef(*std::move(FileOrError.get()));
  auto ModuleOrError = parseBitcodeFile(Buffer, Context);
  if (Error E = ModuleOrError.takeError()) {
    handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
        Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error, EIB.message());
        });
    return nullptr;
  }

  return std::move(ModuleOrError.get());
}


// Load the libdevice module and link that into the given module
//
void LinkInLibdeviceModule(Module &M, LLVMContext &Context)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> Lib = LoadLibdeviceModule(Err, Context);
  if (!Lib) {
    Err.print("swift-to-ptx<parallel_for>", errs());
    exit(1);
  }

  // During linking we get a set of functions which should be internalised
  if (Linker::linkModules(M, std::move(Lib), Linker::Flags::None,
        [](Module &M, const StringSet<> &Internalise) {
          for (auto &I : Internalise) {
            if (auto F = M.getFunction(I.getKey())) {
              F->setLinkage(GlobalValue::InternalLinkage);
            }
          }
        })) {
    report_fatal_error("failed to link in libdevice module", false);
  }
}

// Load the host support code and link that into the given module
//
void LinkInHostSupportCode(Module &M, LLVMContext &Context)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> M2 = parseAssembly(host_support, Err, Context);
  if (!M2) {
    Err.print("swift-to-ptx<parallel_for>", errs());
    exit(1);
  }

  // Copy over the contents of the module directly. Linker::linkModules() will
  // only copy what is needed---which at this point is nothing. Thanks A LOT
  // linkModules(), THANKS. A. LOT.
  //
  // Build the VMap table as we go, which we'll need for CloneFunctionInto().
  ValueToValueMapTy VMap;
  for (auto &GV : M2->globals()) {
    auto *Init = GV.hasInitializer() ? GV.getInitializer() : nullptr;
    auto *NewGV = new GlobalVariable(M, GV.getValueType(), GV.isConstant(), GV.getLinkage(), Init, GV.getName());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }
  for (auto &F : M2->functions()) {
    auto NewF = M.getFunction(F.getName());
    if (!NewF) {
      NewF = Function::Create(F.getFunctionType(), F.getLinkage(), F.getName(), M);
      NewF->copyAttributesFrom(&F);
    }
    VMap[&F] = NewF;
  }

  // Finally copy over the function bodies.
  //
  // XXX: We are continually polluting the VMap here, adding the arguments to
  // each new function. Should we reset it at the start of each loop?
  for (auto &Src : M2->functions()) {
    if (Src.isDeclaration())
      continue;

    Function *Dst = M.getFunction(Src.getName());
    Function::arg_iterator DstI = Dst->arg_begin();
    for (const Argument &I : Src.args()) {
      DstI->setName(I.getName());
      VMap[&I] = &*DstI++;
    }
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(Dst, &Src, VMap, CloneFunctionChangeType::DifferentModule, Returns);
  }
}

#if false
// TODO: We should update the address space of the environment pointer (and
// anything that reads from it) to avoid some additional address space casts in
// the generated code. It seems to help the optimiser a bit as well.
//   ---TLM-2024-03-12
class PointerAddressSpaceUpdater : public ValueMapTypeRemapper {
public:
  Type* remapType(Type *T)
  {
    if (PointerType* P = dyn_cast<PointerType>(T)) {
      if (0 == P->getAddressSpace()) {
        return PointerType::get(P->getContext(), 1);
      }
    }

    return T;
  }
};

class TypeContextRemapper : public ValueMapTypeRemapper {
private:
  LLVMContext& Context;
public:
  TypeContextRemapper(LLVMContext& Context) : Context(Context) {}

  Type* remapType(Type *T)
  {
    // First, handle the derived types
    if (ArrayType *S = dyn_cast<ArrayType>(T)) {
      return ArrayType::get(remapType(S->getElementType()), S->getNumElements());
    }
    if (FunctionType *S = dyn_cast<FunctionType>(T)) {
      std::vector<Type*> Params;
      Params.reserve(S->getNumParams());
      for (Type *P : S->params()) {
        Params.push_back(remapType(P));
      }
      return FunctionType::get(remapType(S->getReturnType()), ArrayRef(Params), S->isVarArg());
    }
    if (IntegerType *S = dyn_cast<IntegerType>(T)) {
      return IntegerType::get(Context, S->getBitWidth());
    }
    if (PointerType *S = dyn_cast<PointerType>(T)) {
      // Convert typed pointers to opaque pointers. This happens with internally
      // as well, so we aren't losing anything here.
      return PointerType::get(Context, S->getAddressSpace());
    }
    if (StructType *S = dyn_cast<StructType>(T)) {
      std::vector<Type*> Elements;
      Elements.reserve(S->getNumElements());
      for (Type *E : S->elements()) {
        Elements.push_back(remapType(E));
      }
      if (S->hasName()) {
        return StructType::create(Context, ArrayRef(Elements), S->getName(), S->isPacked());
      }
      else {
        if (S->isLiteral()) {
          return StructType::get(Context, ArrayRef(Elements), S->isPacked());
        }
        else {
          return StructType::create(Context, ArrayRef(Elements));
        }
      }
    }
    if (TargetExtType *S = dyn_cast<TargetExtType>(T)) {
      std::vector<Type*> Params;
      Params.reserve(S->getNumTypeParameters());
      for (Type* P : S->type_params()) {
        Params.push_back(remapType(P));
      }
      return TargetExtType::get(Context, S->getName(), ArrayRef(Params), S->int_params());
    }
    if (TypedPointerType *S = dyn_cast<TypedPointerType>(T)) {
      return TypedPointerType::get(remapType(S->getElementType()), S->getAddressSpace());
    }
    if (VectorType *S = dyn_cast<VectorType>(T)) {
      if (FixedVectorType *U = dyn_cast<FixedVectorType>(S)) {
        return FixedVectorType::get(remapType(S->getElementType()), U->getNumElements());
      }
      if (ScalableVectorType *U = dyn_cast<ScalableVectorType>(S)) {
        return ScalableVectorType::get(remapType(S->getElementType()), U->getMinNumElements());
      }
      report_fatal_error("unhandled VectorType", false);
    }
    // Everything else should be a primitive type
    return Type::getPrimitiveType(Context, T->getTypeID());
  }
};
#endif

// Compile the given PTX assembly code into SASS object code. This will call out
// to 'ptxas' in order to do the work, piping the data in and out via pipes and
// thus avoiding the creation of temporary files on disk.
//
// The returned memory is owned by the caller, who is expected to eventually
// call free() on the underlying data once it is no longer required.
//
// TODO: The CUDA toolkit now provides an API for calling into 'ptxas'. That may
// be simpler than this method, but we should test if it is actually better
// (e.g. avoids temporary files) as well.
//
ArrayRef<uint8_t> CompileKernel(SmallVector<char> Asm)
{
  int fd0[2]; // stdin
  int fd1[2]; // stdout
  int fd2[2]; // stderr

  if ( pipe(fd0) < 0 || pipe(fd1) < 0 || pipe(fd2) < 0) {
    report_fatal_error("pipe error", false);
  }

  pid_t pid = fork();
  if (pid < 0) {
    report_fatal_error("fork() error", false);
  }

  if (pid == 0) {
    // CHILD PROCESS
    close(fd0[1]);
    close(fd1[0]);
    close(fd2[0]);

    if (fd0[0] != STDIN_FILENO) {
      if (dup2(fd0[0], STDIN_FILENO) != STDIN_FILENO) {
        LLVM_DEBUG(dbgs() << "dup2() error on stdin\n");
      }
      close(fd0[0]);
    }

    if (fd1[1] != STDOUT_FILENO) {
      if (dup2(fd1[1], STDOUT_FILENO) != STDOUT_FILENO) {
        LLVM_DEBUG(dbgs() << "dup2() error on stdout\n");
      }
      close(fd1[1]);
    }

    if (fd2[1] != STDERR_FILENO) {
      if (dup2(fd2[1], STDERR_FILENO) != STDERR_FILENO) {
        LLVM_DEBUG(dbgs() << "dup2() error on stderr\n");
      }
      close(fd2[1]);
    }

    // Replace the current process image. If this returns then an error has occurred.
    const char* const argv[] =
      { PTXASPath.data()
      , "--verbose"
      , "-arch", TargetGPU.data()
      , "-o", "/dev/stdout"       // send the output to stdout pipe
      , "-"                       // read input from stdin pipe
      , nullptr
      };
    execv(PTXASPath.data(), const_cast<char* const*>(argv));
    report_fatal_error("execv() failed", false);
  }
  else {
    // PARENT PROCESS
    close(fd0[0]);
    close(fd1[1]);
    close(fd2[1]);

    if (write(fd0[1], Asm.data(), Asm.size()) != static_cast<ssize_t>(Asm.size())) {
      report_fatal_error("failed to write data to pipe", false);
    }
    close(fd0[1]); // send EOF

    // Read data from the connected pipes
    size_t   capacity   = 65536;
    uint8_t* obj_buffer = (uint8_t*) malloc(capacity); // 64KB
    char*    msg_buffer = (char*)    malloc(capacity); // overkill

    // Read in the compiled object code
    size_t offset = 0;
    while (true) {
      ssize_t rv = read(fd1[0], obj_buffer + offset, capacity - offset);
      if (rv == 0)
        break;  // child closed pipe

      if (rv < 0)
        report_fatal_error("pipe error", false);

      offset += rv;
      assert(offset < capacity);
    }
    ArrayRef<uint8_t> obj = ArrayRef(obj_buffer, offset);

    // Read in any error/warning messages
    offset = 0;
    while (true) {
      ssize_t rv = read(fd2[0], msg_buffer + offset, capacity - offset);
      if (rv == 0)
        break;  // child closed pipe

      if (rv < 0)
        report_fatal_error("pipe error", false);

      offset += rv;
      assert(offset < capacity);
    }
    StringRef msg = StringRef(msg_buffer, offset);

    // Get the exit status of the process
    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      report_fatal_error("ptxas exited with code " + Twine(WEXITSTATUS(status)) + " :\n" + msg, false);
    }

    if (WIFSIGNALED(status)) {
      report_fatal_error("ptxas received signal " + Twine(WTERMSIG(status)) + (WCOREDUMP(status) ? " (core dumped)" : ""), false);
    }

    // normal termination
    assert(status == 0);

    LLVM_DEBUG(dbgs() << msg);
    free(msg_buffer);

    return obj;
  }
}

#if false
bool isAcceptableChar(char c) {
  return isAlnum(c) || c == '_' || c == '$' || c == '.' || c == '@';
}

bool isValidUnquotedName(StringRef Name) {
  assert (!Name.empty());

  for (auto c : Name) {
    if (!isAcceptableChar(c))
      return false;
  }

  return true;
}

std::string makeValidUnquotedName(StringRef Name)
{
  // Avoid the extra allocation if possible
  if (isValidUnquotedName(Name))
    return Name.str();

  std::string Encoded;
  Encoded.reserve(Name.size());

  for (auto c : Name) {
    if (isAcceptableChar(c)) {
      Encoded.push_back(c);
    } else {
      if (c == ' ') {
        Encoded.push_back('_');
      } else {
        Encoded.append(utohexstr(c));
      }
    }
  }

  return Encoded;
}
#endif

std::optional<StringRef> getGlobalInitializerString(Value* Value)
{
  std::optional<StringRef> R = {};

  if (auto C = dyn_cast<ConstantExpr>(Value)) {
    if (auto I = dyn_cast<PtrToIntInst>(C->getAsInstruction())) {
      if (auto G = dyn_cast<GlobalVariable>(I->getPointerOperand())) {
        if (auto D = dyn_cast<ConstantDataSequential>(G->getInitializer())) {
          R = D->getAsCString();
        }
      }
      delete I; // getAsInstruction() creates a parent-less instruction
    }
  }

  return R;
}

GlobalValue* newStaticString(LLVMContext& Context, Module& Module, std::string String)
{
  auto C = ConstantDataArray::getString(Context, String);
  auto G = new GlobalVariable(Module, C->getType(), true, GlobalValue::PrivateLinkage, C);
  G->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  return G;
}

// Create a new parallel_for GPU kernel consisting of the given set of
// functions. The generated kernel code is returned.
//
// XXX: In principle we should create this in a separate LLVMContext (it is an
// entirely separate module with a different target architecture) but copying
// the functions over will require more work because their types, attributes,
// and other metadata etc. still refer to the context of the source module. If
// we cleanly separate this we should be able to compile multiple units
// concurrently, which could be useful.
//
ArrayRef<uint8_t> CreateKernel
(
    LLVMContext& Context,
    Module& Src,
    StringRef Main,
    SetVector<GlobalValue*> GVs
)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssembly(parallel_for_kernel, Err, Context);
  if (!M) {
    Err.print("swift-to-ptx<parallel_for>", errs());
    exit(1);
  }

  // Loop over all of the functions that we need to copy from the input module.
  // Just make the declarations, the bodies will come later. Take care of
  // functions that need to be handled specially on the device.
  ValueToValueMapTy VMap;
  bool HaveLibdevice = false;
  for (auto &GV : GVs) {
    Function *Src = cast<Function>(GV);

    // If this is a declaration for a function provided by libdevice (e.g.
    // llvm.sin.f32) then link in the libdevice module and record in the VMap the
    // mapping to the corresponding libdevice implementation (e.g. __nvsinf).
    //
    // Delay linking in libdevice to this point where we are certain we need it,
    // which saves a few hundred ms in case it would not have been used.
    StringRef Name = Src->getName();
    StringRef Lib  = libdeviceFunctions.lookup(Name);
    if (!Lib.empty()) {
        if (!HaveLibdevice) {
          LinkInLibdeviceModule(*M, Context);
          HaveLibdevice = true;
        }
        VMap[Src] = M->getFunction(Lib);;
        continue;
    }

    // Otherwise, this is just a regular function (declaration). Just make the
    // declaration, we'll copy over the function body later.
    Function* Dst = Function::Create(Src->getFunctionType(), Src->getLinkage(), Name, *M);
    Dst->copyAttributesFrom(Src);
    VMap[Src] = Dst;
  }

  // Copy over the function bodies. Also enable floating point contraction for
  // compatible instructions.
  for (auto &GV : GVs) {
    if (GV->isDeclaration())
      continue;

    Function *Src = cast<Function>(GV);
    Function *Dst = cast<Function>(VMap[Src]);

    Function::arg_iterator DstI = Dst->arg_begin();
    for (const Argument &I : Src->args()) {
      DstI->setName(I.getName());
      VMap[&I] = &*DstI++;
    }
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(Dst, Src, VMap, CloneFunctionChangeType::DifferentModule, Returns);
    /* TypeContextRemapper TypeMapper(Context); */
    /* PointerAddressSpaceUpdater TypeMapper; */
    /* CloneFunctionInto(Dst, Src, VMap, CloneFunctionChangeType::DifferentModule, Returns, "", nullptr, &TypeMapper); */
    Dst->setCallingConv(CallingConv::C);
    Dst->setLinkage(GlobalValue::InternalLinkage);

    if (Src->hasPersonalityFn())
      Dst->setPersonalityFn(MapValue(Src->getPersonalityFn(), VMap));

    // Allow floating-point contraction (i.e. FMA)
    for (auto &BB : *Dst) {
      for (auto &I : BB) {
        if (isa<FPMathOperator>(&I)) {
          I.setHasAllowContract(true);
        }
      }
    }
  }

  // Replace swift error handling functions with equivalents that we can call
  // from the device
  if (Function* _fatalErrorMessage = M->getFunction("$ss18_fatalErrorMessage__4file4line5flagss5NeverOs12StaticStringV_A2HSus6UInt32VtF")) {
    for (auto U = _fatalErrorMessage->user_begin(), UE = _fatalErrorMessage->user_end(); U != UE; ) {
      Value* V = *U++;

      if (CallInst* CI = dyn_cast<CallInst>(V)) {
        // The arguments to Swift._fatalErrorMessage are all StaticString, but
        // we need to first unpack the arguments to get a pointer to the static
        // string data, and then format the prefix and message parts together.
        auto Prefix  = getGlobalInitializerString(CI->getArgOperand(0));
        auto Message = getGlobalInitializerString(CI->getArgOperand(3));
        auto File    = getGlobalInitializerString(CI->getArgOperand(6));
        auto Line    = cast<ConstantInt>(CI->getArgOperand(9))->getZExtValue();

        auto __assertfail = M->getFunction("__assertfail");
        CallInst* CINew = CallInst::Create(__assertfail->getFunctionType(), __assertfail,
            { newStaticString(Context, *M, Prefix->str() + (Message ? ": " + Message->str() : ""))
            , newStaticString(Context, *M, File->str())
            , ConstantInt::get(IntegerType::getInt32Ty(Context), Line)
            , newStaticString(Context, *M, demangleSymbolAsString(CI->getCaller()->getName(), swift::Demangle::DemangleOptions()))
            , ConstantInt::get(IntegerType::getInt64Ty(Context), 1)
            });

        ReplaceInstWithInst(CI, CINew);
      }
    }
  }

  // XXX: CUDA-11.4 doesn't understand the PTX-7.5 syntax that LLVM-17 is
  // (incorrectly) generating for debug information
  StripDebugInfo(*M);

  // Update the kernel function to call the main (entry) function from the set
  // that we extracted in the previous step.
  //
  // XXX: We really should do a lot of simplification and beta reduction here,
  // to e.g. remove generic type parameters.
  Function *Body = M->getFunction("body");
  assert(Body->hasOneUser() && "expected only one call to the kernel body");
  assert(isa<CallInst>(Body->getUniqueUndroppableUser()));
  CallInst *CI = cast<CallInst>(Body->getUniqueUndroppableUser());
  CI->setCalledOperand(M->getFunction(Main));

  // Create a target machine
  std::string Error;
  auto TargetTriple = "nvptx64-nvidia-cuda";
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    report_fatal_error(StringRef(Error), false);
  }
  TargetOptions opt;
  TargetMachine* TargetMachine = Target->createTargetMachine(TargetTriple, TargetGPU, TargetFeatures, opt, Reloc::PIC_);
  M->setDataLayout(TargetMachine->createDataLayout());

  // Run a full optimisation pass on this module
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB(TargetMachine);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // This corresponds to the typical -O3 optimization pipeline
  ModulePassManager PM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  PM.addPass(VerifierPass());
  PM.run(*M, MAM);

  // Generate target assembly. Being a backend/code generation pass, this
  // uses the legacy pass manager so does not integrate with the above.
  legacy::PassManager legacy;
  SmallVector<char> Asm;  // XXX: reserve space to avoid growing too frequently?
  raw_svector_ostream ostream(Asm);
  if (TargetMachine->addPassesToEmitFile(legacy, ostream, nullptr, CGFT_AssemblyFile)) {
    report_fatal_error("could not create output stream", false);
  }
  legacy.run(*M);

  // Compile the target assembly to object code. This calls out to ptxas to do
  // the work, piping the data in and out via pipes and thus avoiding the
  // creation of temporary files on disk.
  ArrayRef<uint8_t> Obj = CompileKernel(Asm);

  // ===== DEBUGGING =====

  // Since we are processing completely in-memory, in order to "keep"
  // intermediate files, we need to write them to disk for the first time.
  if (KeepIntermediateFiles) {
    int src_fd = 0;
    SmallVector<char> src_path;
    if (sys::fs::createTemporaryFile("kernel", "ll", src_fd, src_path)) {
      report_fatal_error("Failed to create output file", false);
    }
    // XXX: Note that we need to dump the generated IR from a _different_
    // pass manager, otherwise generating target assembly with the old pass
    // manager will cause Bad Things™ to happen.
    ModulePassManager PM;
    auto src_out = raw_fd_ostream(src_fd, true);
    PM.addPass(PrintModulePass(src_out));
    PM.run(*M, MAM);
    src_out.close();
    errs() << src_path << "\n";

    int ptx_fd = 0;
    SmallVector<char> ptx_path;
    if (sys::fs::createTemporaryFile("kernel", "ptx", ptx_fd, ptx_path)) {
      report_fatal_error("Failed to create output file", false);
    }
    auto ptx_out = raw_fd_ostream(ptx_fd, true);
    ptx_out << Asm;
    ptx_out.close();
    errs() << ptx_path << "\n";

    int obj_fd = 0;
    SmallVector<char> obj_path;
    if (sys::fs::createTemporaryFile("kernel", "o", obj_fd, obj_path)) {
      report_fatal_error("Failed to create output file", false);
    }
    auto obj_out = raw_fd_ostream(obj_fd, true);
    obj_out.write((const char*) Obj.data(), Obj.size());
    obj_out.close();
    errs() << obj_path << "\n";
  }

  // TODO FIXME: For unknown reasons I am currently having a problem using the
  // embedded object code, getting a 'CUDA_ERROR_INVALID_IMAGE (200): device
  // kernel image is invalid' when trying to load the module. Switching to the
  // PTX compiler API may or may not fix this. For now, just return the PTX
  // assembly.
  //   --- TLM 2024-10-09
  /* return Obj; */

  size_t   sz  = Asm.size() + 1;
  uint8_t* buf = (uint8_t*) malloc(sz);
  if (!buf) {
    report_fatal_error("failed to allocate %ld bytes", sz);
  }

  memcpy(buf, Asm.data(), Asm.size());
  buf[Asm.size()] = '\0';

  return ArrayRef(buf, sz);
}


// Swift will apply scalar replacement of aggregates in order to pass struct
// (components) in registers for function calls.
//
// Somewhat ironically, we repackage those components again to make it a bit
// more convenient to work with, and hope that the (C++) compiler again does the
// same thing to make our function calls more efficient. We should really check
// whether it does...
typedef std::tuple<Value*, Value*, Value*> CUDAContext;
typedef std::tuple<Value*, Value*, Value*> CachingHostAllocator;

// Update the environment so that its contents are accessible from the device.
// This is the recursive overload of UpdateClosureEnvironment() that does all
// the work; the other overload is the one is the entry point that does the
// setup and finalisation work.
//
// This is perhaps the most tedious part of the entire operation. There are a
// few different cases to handle:
//
//   1. alloca instructions (temporary allocations on the stack) are converted
//      into a allocation using cuMemAllocHost(), which must then be deallocated
//      once the kernel completes.
//
//      This is handled by CachingHostAllocator in the swift-to-ptx support
//      library, which provides a cached block allocator for pinned host memory.
//
//   2. Regular pointers must be made accessible to the device via
//      cuMemHostGetDevicePointer(), and that address substituted into the
//      environment instead. The main difficulty here is that we need to know
//      the size of the allocation.
//
//      XXX: It might be better that at the Swift layer we provide our own Array
//      type/extension that always allocated into pinned memory, so that the
//      host and device pointers (on the Orin) are the same, and thus no extra
//      work needs to happen.
//
// We need to do this recursively because the closure may consist of many layers
// of indirection.
//
template <unsigned N>
void UpdateClosureEnvironment (
    LLVMContext& Context,
    Module& M,
    Value* K,
    Value* P,
    CUDAContext CUDA,
    CachingHostAllocator Allocator,
    Value* Event,
    SmallVector<Instruction*>& ToErase,
    SmallPtrSet<Value*, N>& ToFree
)
{
  /* std::string dots(depth, '.'); */
  /* errs() << dots << *P << "\n"; */

  // Recur through address calculations. The user of this GEP will most likely
  // need to refer back to this instruction in order to analyse the address
  // calculation or pointer operand; perhaps we should do that already and guide
  // the recursion?
  if (GetElementPtrInst *I = dyn_cast<GetElementPtrInst>(P)) {
    for (auto *U : I->users()) {
      UpdateClosureEnvironment(Context, M, K, U, CUDA, Allocator, Event, ToErase, ToFree);
    }
  }

  // Convert stack allocations into a pinned memory allocations, so that the
  // memory is accessible from the device.
  //
  // Later, we also need to deallocate the memory. There are three possible
  // cases for handling this:
  //
  //   1. The memory is bracketed by llvm.stacksave/llvm.stackrestore
  //   2. The memory is marked as explicitly alive/dead by llvm.lifetime.start/llvm.lifetime.end respectively
  //   3. The memory is implicitly released when the function returns
  //
  // We use the CachingHostAllocator for this, which handles our asynchronous
  // deallocation requirements (lifetime extends beyond the lexical scope of the
  // function call) and amortizes the cost of calling into the CUDA runtime to
  // allocate pinned memory.
  //
  else if (AllocaInst* I = dyn_cast<AllocaInst>(P)) {
    auto TypeSize = I->getAllocationSize(M.getDataLayout());
    assert(TypeSize && "could not determine size of alloca");

    // Convert from stack to pinned heap allocation
    IntegerType *i64_t = IntegerType::getInt64Ty(Context);
    ConstantInt *size = ConstantInt::get(i64_t, TypeSize->getFixedValue());
    Function *F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV5allocySvSiF");
    CallInst *NewI = CallInst::Create(F->getFunctionType(), F, {size, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
    NewI->setCallingConv(CallingConv::Swift);
    ReplaceInstWithInst(I, NewI);
    ToFree.insert(NewI);

    // Determine how to (asynchronously) free the memmory
    // [Free 1]: Check for an llvm.stacksave() instruction
    if (auto *Prev = dyn_cast<Instruction>(NewI)->getPrevNonDebugInstruction()) {
      if (CallInst *Save = dyn_cast<CallInst>(Prev)) {
        if (Save->getCalledFunction()->getName() == "llvm.stacksave") {
          assert(Save->hasOneUser() && "expected llvm.stacksave() to have a single unique undroppable user");

          auto *Restore = cast<Instruction>(Save->getUniqueUndroppableUser());
          assert(cast<CallInst>(Restore)->getCalledFunction()->getName() == "llvm.stackrestore");

          Function *F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
          CallInst *Free = CallInst::Create(F->getFunctionType(), F, {NewI, Event, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
          Free->setCallingConv(CallingConv::Swift);
          Free->insertAfter(Restore);
          ToFree.erase(NewI);

          // Safe to erase these directly since users() will not be iterating
          // over them. XXX: The order is important here
          Restore->eraseFromParent();
          Save->eraseFromParent();
        }
      }
    }

    // Recursively update all users of this memory. This will also handle case [Free 2].
    for (auto *U : NewI->users()) {
      UpdateClosureEnvironment(Context, M, K, U, CUDA, Allocator, Event, ToErase, ToFree);
    }

    // [Free 3] If the memory is still not freed, this will be handled once the
    // recursive update is complete.
  }

  // There is a lot to be done here, depending on the type of the value
  // operand.
  //
  // 1. If this is an instruction, recur. There is a potential to get into a
  //    loop here because we might be storing a pointer to the memory that was
  //    alloca'd as part of the closure environment. This cycle is broken
  //    because that alloca instruction will have already been replaced with a
  //    call to the caching allocator.
  //
  // 2. Otherwise, this must be an input to the function. There are two
  //    possibilities:
  //
  //    a. This a data pointer (e.g. array). We need to determine the pointer to
  //       the underlying data, as well as the number of bytes in the payload.
  //       The data can then be registered as accessible from the device, and
  //       the new device-accessible address is stored in place of the existing
  //       address.
  //
  //       TODO: We should also unregister this memory once it is no longer
  //       needed on the device, or at least when the array is deallocated, so
  //       that we do not leak *pinned* memory. Note that this array might be
  //       used from multiple places, so effectively we need to do hook into the
  //       reference counting system.
  //
  //    b. This is a function pointer (i.e. closure). There are further
  //       possibilities:
  //
  //       a. If we can determine (from the callsite of the parent function)
  //          what function is being called (i.e. rely on the compiler
  //          specialising this call), we should also lift that code out and
  //          compile it (i.e. beta-reduce) directly into the kernel.
  //
  //       b. If we can't determine what function this is (i.e. it is just a raw
  //          function pointer), then we'll need to update the closure pointer
  //          at runtime after the CUDA module has been loaded with the address
  //          of the device function.
  //
  //       TODO: Note that function pointers are currently not handled at all.
  //
  else if (StoreInst *I = dyn_cast<StoreInst>(P)) {
    Value* A = I->getValueOperand();
    if (isa<Instruction>(A)) {
      UpdateClosureEnvironment(Context, M, K, A, CUDA, Allocator, Event, ToErase, ToFree);
    }
  }

  // Calls to llvm.lifetime.{start,end}---which manage stack allocation
  // lifetimes---will be replaced with calls to our caching pinned (heap) memory
  // allocator.
  else if (CallInst *I = dyn_cast<CallInst>(P)) {
    if (Function* F = I->getCalledFunction()) {
      StringRef Name = F->getName();
      if (Name.starts_with("llvm.lifetime.start")) {
        // Assume that we will encounter the corresponding .end()
        ToErase.push_back(I);
      }
      else if (Name.starts_with("llvm.lifetime.end")) {
        // Assume that we will encounter the corresponding .start()
        Value *Alloca = I->getArgOperand(1);
        Function *F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
        CallInst *Free = CallInst::Create(F->getFunctionType(), F, {Alloca, Event, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
        Free->setCallingConv(CallingConv::Swift);
        Free->insertAfter(I);
        ToFree.erase(Alloca);
        ToErase.push_back(I);
      }
    }
  }

  else if (LoadInst *I = dyn_cast<LoadInst>(P)) {
    // Nothing
  }

  else {
    LLVM_DEBUG(dbgs() << "UpdateClosureEnvironment: unhandled instruction: " << *P << "\n");
  }
}

// This is the entry point to UpdateClosureEnvironment(). This handles some
// setup and finalisation. The other overload of this function is the recursive
// one that does all the hard work.
//
void UpdateClosureEnvironment (
    LLVMContext& Context,
    Module& M,
    Value* Body,
    Value* Env,
    CUDAContext CUDA,
    CachingHostAllocator Allocator,
    Value* Event
)
{
  SmallVector<Instruction*> ToErase;
  SmallPtrSet<Value*, 8> ToFree;
  Function *Parent = cast<Instruction>(Env)->getFunction();

  // Recursively marshal the closure environment to device-accessible memory
  UpdateClosureEnvironment(Context, M, Body, Env, CUDA, Allocator, Event, ToErase, ToFree);

  // Erase any instructions that we couldn't remove along the way (instructions
  // that might be part of the users() chain).
  for (auto *I : ToErase) {
    I->eraseFromParent();
  }

  // Free any remaining allocations at function exit.
  //
  // The way to do this is to find (or make) a single return point from the
  // function, and free the memory just before this.
  //
  //   1. Iterate through the basic blocks of the function and see whether the
  //      terminator of each is a ReturnInst. Add the ones that are to a list.
  //   2. If this list contains more than one entry then we must make a new
  //      basic block to return from. If the return type isn't void (it should
  //      be for parallel_for, but in future this might change) then create phi
  //      nodes to populate the new block with the incoming values.
  //   3. Replace all the old returns from (1) with branches to the new block.
  //   4. Free the remaining allocations before the (new) single return
  if (ToFree.size()) {
    SmallVector<Instruction*> Returns;
    for (auto &BB : *Parent) {
      if (ReturnInst *I = dyn_cast<ReturnInst>(BB.getTerminator()))
        Returns.push_back(I);
    }

    assert(Returns.size() == 1 && "Expected function with single 'ret' instruction. Run pass 'mergereturn'?");
    Function* F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
    Instruction *Ret = Returns.front();
    for (auto *Alloca : ToFree) {
      CallInst *Free = CallInst::Create(F->getFunctionType(), F, {Alloca, Event, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
      Free->setCallingConv(CallingConv::Swift);
      Free->insertBefore(Ret);
    }
  }

  // Make sure the definition of the allocator dominates all uses. This is
  // necessary when e.g. the default parameter allocator is used. This gets
  // initialised only right before the call to parallel_for, but we may need it
  // available before that point in order to marshal the closure environment to
  // the GPU. That case will look something like:
  //
  // > %17 = call swiftcc { ptr, ptr, ptr } @"$s10SwiftToPTX12parallel_for10iterations7context9allocator6stream_AA5EventCSi_AA7ContextVAA20CachingHostAllocatorVAA6StreamVySiXEtFfA1_"()
  // > %18 = extractvalue { ptr, ptr, ptr } %17, 0
  // > %19 = extractvalue { ptr, ptr, ptr } %17, 1
  // > %20 = extractvalue { ptr, ptr, ptr } %17, 2
  //
  // We'll also need to move the corresponding swift_release() calls to after
  // the last use of them. That chuck looks like:
  //
  // > call void @swift_release(ptr %20) #6
  // > call void @swift_release(ptr %19) #6
  // > call void @swift_release(ptr %18) #6
  //
  // Similarly we need to make sure that the context is initialised before using
  // the allocator
  //
  DominatorTree DT(*Parent);
  if (isa<Instruction>(get<0>(CUDA)) || isa<Instruction>(get<0>(Allocator))) {
    Instruction* Lowest = nullptr;
    Instruction* Highest = nullptr;
    Instruction* Release0 = nullptr;
    Instruction* Release1 = nullptr;
    Instruction* Release2 = nullptr;

    // Locate the pre- and post-dominating instructions
    for (auto *U : get<0>(Allocator)->users()) {
      if (Instruction* I = dyn_cast<Instruction>(U)) {
        if (CallInst* CI = dyn_cast<CallInst>(I)) {
          if (CI->getCalledFunction()->getName() == "swift_release") {
            assert(!Release0 && "expected a single call to swift_release");
            Release0 = I;
            Release1 = Release0->getPrevNonDebugInstruction();
            Release2 = Release1->getPrevNonDebugInstruction();
            assert(cast<CallInst>(Release1)->getCalledFunction()->getName() == "swift_release");
            assert(cast<CallInst>(Release2)->getCalledFunction()->getName() == "swift_release");
            continue;
          }
        }

        if (!Highest) {
          Lowest = Highest = I;
          continue;
        }

        if (!DT.dominates(Highest, I)) {
          Highest = I;
          continue;
        }

        if (DT.dominates(Lowest, I)) {
          Lowest = I;
          continue;
        }
      }
    }

    // Move context and allocator (de)initialisers
    if (Instruction *I = dyn_cast<Instruction>(get<0>(CUDA))) {
      if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(I)) {
        if (Instruction *I0 = dyn_cast<Instruction>(EV->getAggregateOperand())) {
          I0->moveBefore(Highest);
        }
      }
      I->moveBefore(Highest);
      cast<Instruction>(get<1>(CUDA))->moveBefore(Highest);
      cast<Instruction>(get<2>(CUDA))->moveBefore(Highest);
    }

    if (Instruction *I = dyn_cast<Instruction>(get<0>(Allocator))) {
      if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(I)) {
        if (Instruction *I0 = dyn_cast<Instruction>(EV->getAggregateOperand())) {
          I0->moveBefore(Highest);
        }
      }
      I->moveBefore(Highest);
      cast<Instruction>(get<1>(Allocator))->moveBefore(Highest);
      cast<Instruction>(get<2>(Allocator))->moveBefore(Highest);
    }

    if (Release0) {
      Release0->moveAfter(Lowest);
      Release1->moveAfter(Lowest);
      Release2->moveAfter(Lowest);
    }
  }

  // Also ensure we don't swift_release the ready event too early
  Instruction* Lowest = nullptr;
  Instruction* Release = nullptr;
  for (auto *U : Event->users()) {
    if (Instruction* I = dyn_cast<Instruction>(U)) {
      if (CallInst* CI = dyn_cast<CallInst>(I)) {
        if (CI->getCalledFunction()->getName() == "swift_release") {
          Release = I;
          continue;
        }
      }

      if (!Lowest) {
        Lowest = I;
        continue;
      }

      if (DT.dominates(Lowest, I)) {
        Lowest = I;
      }
    }
  }

  if (Release && Lowest) {
    Release->moveAfter(Lowest);
  }
}

} // end of anonymous namespace


PreservedAnalyses swift::ParallelForPass::run(Module &M, ModuleAnalysisManager &MAM)
{
  LLVMContext &Context = M.getContext();

  // Find all use sites of the `parallel_for` function.
  Function* Fseq = M.getFunction("$s10SwiftToPTX12parallel_for10iterations7context9allocator6stream_AA5EventCSi_AA7ContextVAA20CachingHostAllocatorVAA6StreamVySiXEtF");
  if (!Fseq) {
    LLVM_DEBUG(dbgs() << "No uses of function `SwiftToPTX.parallel_for()` found in this module\n");
    return PreservedAnalyses::all();
  }

  // Required to parse host support textual IR
  bool DiscardValueNames = Context.shouldDiscardValueNames();
  Context.setDiscardValueNames(false);

  // Initialise the target machines we require
  InitializeNativeTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  // We have found at least one call to parallel_for() that we will convert into
  // a parallel GPU kernel. First, add all the necessary host-side support code.
  LinkInHostSupportCode(M, Context);

  Function* Fpar = M.getFunction("$s10SwiftToPTX19launch_parallel_for10iterations6kernel3env7context6streamAA5EventCSi_AA17ParallelForKernelVzSvAA7ContextVAA6StreamVtF");
  StructType* kernel_t = StructType::getTypeByName(Context, "T10SwiftToPTX17ParallelForKernelV");

  // Iterate over all uses of the `parallel_for(iterations: body:)` function
  for (auto U = Fseq->user_begin(), UE = Fseq->user_end(); U != UE; /* See: [1] */) {
    // NOTE [1]: Update the iterator to point to the next User already, because
    // we might modify this instruction and thus break the sequence.
    Value* V = *U++;

    if (CallInst *CI = dyn_cast<CallInst>(V)) {
      Value* Iterations = CI->getArgOperand(0);
      Value* Context0 = CI->getArgOperand(1);   // CUcontext
      Value* Context1 = CI->getArgOperand(2);   // { cuDevice, multiProcessorCount }
      Value* Context2 = CI->getArgOperand(3);   // { maxThreadsPerMultiprocessor, warpSize }
      Value* Allocator0 = CI->getArgOperand(4); // bin_size_bytes
      Value* Allocator1 = CI->getArgOperand(5); // cached_blocks
      Value* Allocator2 = CI->getArgOperand(6); // live_blocks
      Value* Stream = CI->getArgOperand(7);
      Value* Body = CI->getArgOperand(8);
      Value* Env = CI->getArgOperand(9);

      // The function we are interested in lifting as the body of a parallel loop
      SetVector<GlobalValue*> GVs;
      auto F = cast<Function>(Body);
      GVs.insert(F);

      // The continuation launched might in turn call other functions.
      // Recursively record those functions for extraction as well.
      std::vector<Function *> WorkQueue;
      WorkQueue.push_back(F);

      while (!WorkQueue.empty()) {
        F = &*WorkQueue.back();
        WorkQueue.pop_back();

        for (auto &BB : *F) {
          for (auto &I : BB) {
            if (auto *CB = dyn_cast<CallBase>(&I)) {
              if (auto *CF = CB->getCalledFunction()) {
                if (!GVs.contains(CF)) {
                  GVs.insert(CF);
                  WorkQueue.push_back(CF);
                }
              }
            }
          }
        }
      }

      // Generate PTX assembly for the (set of) functions called by the
      // `parallel_for` launcher, and embed the generated code into the module
      ArrayRef<uint8_t> Obj = CreateKernel(Context, M, Body->getName(), GVs);
      size_t buffer_size = Obj.size();
      IntegerType* i8_t = IntegerType::getInt8Ty(Context);
      ArrayType* image_t = ArrayType::get(i8_t, buffer_size);

      std::vector<Constant*> KernelData(buffer_size);
      std::transform(Obj.begin(), Obj.end(), KernelData.begin(), [&](uint8_t c) { return ConstantInt::get(i8_t, c); });
      free((void*) Obj.data()); // ArrayRef does not own the underlying buffer

      GlobalVariable* Image = new GlobalVariable(M, image_t, true, GlobalValue::InternalLinkage, ConstantArray::get(image_t, ArrayRef(KernelData)));
      Image->setAlignment(Align(1));
      Image->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

      // Swift is kind of bonkers and wraps all data types as struct types,
      GlobalVariable* Kernel = new GlobalVariable(M, kernel_t, false,
          GlobalValue::InternalLinkage,
          ConstantStruct::get(kernel_t,
              { ConstantStruct::get(cast<StructType>(kernel_t->getElementType(0)), {Image})
              , ConstantAggregateZero::get(kernel_t->getElementType(1))
              , ConstantAggregateZero::get(kernel_t->getElementType(2))
              , ConstantAggregateZero::get(kernel_t->getElementType(3))
              , ConstantAggregateZero::get(kernel_t->getElementType(4))
              }));
      Kernel->setAlignment(Align(8));
      Kernel->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

      // Update the calling instruction to our placeholder `parallel_for` to our
      // kernel launcher. This assumes that the environment is set up correctly,
      // which we will do in the next step.
      std::vector<Value*> params = {Iterations, Kernel, Env, Context0, Context1, Context2, Stream};
      CallInst* CIpar = CallInst::Create(Fpar->getFunctionType(), Fpar, params);
      CIpar->setCallingConv(CallingConv::Swift);
      CIpar->addParamAttr(1, Attribute::NonNull);
      CIpar->addParamAttr(2, Attribute::NonNull);
      ReplaceInstWithInst(CI, CIpar);

      // Update the closure environment so that its contents are accessible from the device
      UpdateClosureEnvironment(Context, M, Body, Env, { Context0, Context1, Context2 }, { Allocator0, Allocator1, Allocator2 }, CIpar);
    }
  }

  Context.setDiscardValueNames(DiscardValueNames);
  return PreservedAnalyses::none();
}


#if false
/* ----------------------------------------------------------------------------
 * Registering passes as plugins
 *
 * https://llvm.org/docs/WritingAnLLVMNewPMPass.html#registering-passes-as-plugins
 */

llvm::PassPluginLibraryInfo getParallelForPluginInfo()
{
  return
    { LLVM_PLUGIN_API_VERSION
    , "swift-to-ptx<parallel_for>"
    , LLVM_VERSION_STRING
    , [](PassBuilder &PB) {
        PB.registerOptimizerLastEPCallback(
            [](ModulePassManager &PM, OptimizationLevel O) {
              PM.addPass(ParallelForPass());
              PM.addPass(createModuleToFunctionPassAdaptor(GVNPass()));
              PM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
            });
      }
    };
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize it when added to the pass pipeline on the
// command line, i.e. via '-passes=swift-to-ptx<parallel_for>'
//
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
  return getParallelForPluginInfo();
}
#endif

