//===- ParallelFor.cpp - Lift parallel_for loops to PTX -------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/ParallelFor.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRPrinter/IRPrintingPasses.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <stdlib.h>

using namespace llvm;

#define DEBUG_TYPE "lift-to-ptx"

namespace {

static cl::opt<bool> KeepIntermediateFiles (
  "lift-to-ptx-keep-intermediate-files", cl::Hidden, cl::init(false),
  cl::desc("Keep intermediate files of lift-to-ptx pass"));

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

attributes #0 = { argmemonly nofree nosync nounwind }
attributes #1 = { nofree nosync nounwind readnone }
attributes #2 = { nounwind readnone }

!nvvm.annotations = !{!0}

!0 = !{ptr @parallel_for, !"kernel", i32 1}
)KERNEL", "<parallel_for_kernel>");


static const MemoryBufferRef host_support = MemoryBufferRef(R"HOST(
; ModuleID = '<host_support>'
source_filename = "<host_support>"

%struct.parallel_for_kernel_t = type { ptr, ptr, ptr, i32, i32 }

@stderr = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [41 x i8] c"CUDA call failed with error %s (%d): %s\0A\00", align 1

; Function Attrs: nounwind uwtable
define internal fastcc void @cuda_safe_call(i32 noundef %result) unnamed_addr #0 {
entry:
  %name = alloca ptr, align 8
  %desc = alloca ptr, align 8
  %cmp.not = icmp eq i32 %result, 0
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %name) #5
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %desc) #5
  %call = call i32 @cuGetErrorName(i32 noundef %result, ptr noundef nonnull %name) #5
  %call1 = call i32 @cuGetErrorString(i32 noundef %result, ptr noundef nonnull %desc) #5
  %0 = load ptr, ptr @stderr, align 8, !tbaa !5
  %1 = load ptr, ptr %name, align 8, !tbaa !5
  %2 = load ptr, ptr %desc, align 8, !tbaa !5
  %call2 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str, ptr noundef %1, i32 noundef %result, ptr noundef %2) #6
  call void @exit(i32 noundef 1) #7
  unreachable

if.end:                                           ; preds = %entry
  ret void
}

; Function Attrs: nounwind uwtable
define internal fastcc ptr @cuda_caching_alloc_host(i64 noundef %bytes) local_unnamed_addr #0 {
entry:
  %ptr = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ptr) #5
  %call = call i32 @cuMemAllocHost_v2(ptr noundef nonnull %ptr, i64 noundef %bytes) #5
  call fastcc void @cuda_safe_call(i32 noundef %call)
  %0 = load ptr, ptr %ptr, align 8, !tbaa !5
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ptr) #5
  ret ptr %0
}

; Function Attrs: nounwind uwtable
define internal fastcc void @cuda_caching_free_host(ptr noundef %ptr) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @cuMemFreeHost(ptr noundef %ptr) #5
  tail call fastcc void @cuda_safe_call(i32 noundef %call)
  ret void
}

declare ptr @launch_parallel_for(i64, ptr nonnull, ptr nonnull, ptr)

declare i32 @cuMemHostRegister_v2(ptr noundef, i64 noundef, i32 noundef) local_unnamed_addr #1

declare i32 @cuMemHostGetDevicePointer_v2(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #1

declare i32 @cuMemAllocHost_v2(ptr noundef, i64 noundef) local_unnamed_addr #1

declare i32 @cuMemHostUnregister(ptr noundef) local_unnamed_addr #1

declare i32 @cuMemFreeHost(ptr noundef) local_unnamed_addr #1

declare i32 @cuGetErrorName(i32 noundef, ptr noundef) local_unnamed_addr #1

declare i32 @cuGetErrorString(i32 noundef, ptr noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }
attributes #4 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+outline-atomics,+v8a" }
attributes #5 = { nounwind }
attributes #6 = { cold }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
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
  ,{"llvm.sqrt.f64",        "__nv_sqrt"}
  ,{"llvm.sqrt.f32",        "__nv_sqrtf"}
  ,{"llvm.powi.f64.i32",    "__nv_powi"}
  ,{"llvm.powi.f32.i32",    "__nv_powif"}
  ,{"llvm.sin.f64",         "__nv_sin"}
  ,{"llvm.sin.f32",         "__nv_sinf"}
  ,{"llvm.cos.f64",         "__nv_cos"}
  ,{"llvm.cos.f32",         "__nv_cosf"}
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
  return "libdevice.10.bc";
  /* return "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"; */
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
    Err.print("lift-to-ptx<parallel_for>", errs());
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
    Err.print("lift-to-ptx<parallel_for>", errs());
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
SmallVector<char> CreateKernel(StringRef Main, SetVector<GlobalValue*> GVs, LLVMContext& Context)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssembly(parallel_for_kernel, Err, Context);
  if (!M) {
    Err.print("lift-to-ptx<parallel_for>", errs());
    exit(1);
  }

  // First just declare all of the functions from the input module which will
  // make up (or be called by) the loop body. In the next step we will actually
  // copy over the function bodies.
  ValueToValueMapTy VMap;
  for (auto &GV : GVs) {
    Function *Src = dyn_cast<Function>(GV);
    Function *Dst = Function::Create(Src->getFunctionType(), Src->getLinkage(), Src->getName(), *M);
    Dst->copyAttributesFrom(Src);
    VMap[Src] = Dst;
  }

  // Now, copy over the function bodies.
  //
  // At this point we also lower intrinsic functions (e.g. llvm.sin.f32) to the
  // corresponding libdevice routine (c.f. __nvsinf). We could add all of these
  // functions to the VMap so that it happens as part of CloneFunctionInto(),
  // but that requires linking in the lib device module every time, even if it
  // is not necessary, and this way we can do it lazily, which saves some time
  // (a few tens of milliseconds, but for small functions this approaches half
  // of the overall runtime time).
  //
  // Also enable floating point contraction for compatible instructions.
  bool HaveLibdevice = false;
  for (auto &GV: GVs) {
    if (GV->isDeclaration())
      continue;

    Function *Src = dyn_cast<Function>(GV);
    Function *Dst = M->getFunction(Src->getName());
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

    for (auto &BB : *Dst) {
      for (auto &I : BB) {
        // Lower intrinsic functions to their libdevice module implementations
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *F = CI->getCalledFunction()) {
            StringRef target = libdeviceFunctions.lookup(F->getName());
            if (!target.empty()) {
              if (!HaveLibdevice) {
                LinkInLibdeviceModule(*M, Context);
                HaveLibdevice = true;
              }
              CI->setCalledOperand(M->getFunction(target));
            }
          }
        }
        // Allow floating-point contraction (i.e. FMA)
        else if (isa<FPMathOperator>(&I)) {
          I.setHasAllowContract(true);
        }
      }
    }
  }

  // Set the debug info version of this module, otherwise debug info will be
  // dropped during code generation
  /* M->addModuleFlag(Module::Error, "Debug Info Version", ...); */

  // Update the kernel function to call the main (entry) function from the set
  // that we extracted in the previous step.
  //
  // XXX: We really should do a lot of simplification and beta reduction here,
  // to e.g. remove generic type parameters.
  Function *Body = M->getFunction("body");
  assert(Body->hasOneUser() && "expected only one call to the kernel body");
  assert(isa<CallInst>(Body->getUniqueUndroppableUser()));
  CallInst *CI = dyn_cast<CallInst>(Body->getUniqueUndroppableUser());
  CI->setCalledOperand(M->getFunction(Main));

  // Create a target machine for the Orin
  std::string Error;
  auto TargetTriple = "nvptx64-nvidia-cuda";
  auto CPU = "sm_87";
  auto Features = "";
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    report_fatal_error(StringRef(Error), false);
  }
  TargetOptions opt;
  TargetMachine* TargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, Reloc::PIC_);
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
  /* PM.addPass(VerifierPass()); */
  PM.run(*M, MAM);

  // Generate target assembly. Being a backend/code generation pass, this
  // uses the legacy pass manager so does not integrate with the above.
  legacy::PassManager legacy;
  SmallVector<char> Asm;  // XXX: reserve space to avoid growing too frequently?
  raw_svector_ostream ostream(Asm);
  if (TargetMachine->addPassesToEmitFile(legacy, ostream, nullptr, CGFT_AssemblyFile)) {
    report_fatal_error("could not emit output file", false);
  }
  legacy.run(*M);

  // ===== DEBUGGING =====

  // Since we are processing as much as possible in-memory, in order to
  // "keep" intermediate files, we actually need to write them to disk for
  // the first time. The only exception to this will be once we are piping
  // the generated PTX through ptxas, since that will always store its
  // result to a file on disk.
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

    int obj_fd = 0;
    SmallVector<char> obj_path;
    if (sys::fs::createTemporaryFile("kernel", "ptx", obj_fd, obj_path)) {
      report_fatal_error("Failed to create output file", false);
    }
    auto obj_out = raw_fd_ostream(obj_fd, true);
    obj_out << Asm;
    obj_out.close();
    errs() << obj_path << "\n";
  }

  return Asm;
}

// Update the environment so that its contents is accessible from in
// device memory. This is the recursive overload of UpdateClosureEnvironment()
// that does all the work; the other overload is the one is the entry point that
// does the setup and finalisation work.
//
// This is perhaps the most tedious part of the entire operation. There
// are a few different cases to handle:
//
//   1. alloca instructions (temporary allocations on the stack) are
//      converted into a allocation using cuMemAllocHost(), which must
//      then be deallocated once the kernel completes.
//
//      XXX: Currently this memory is never freed. cuLaunchHostFunc() is
//      the obvious choice for freeing this memory, but can
//      not call CUDA API functions (i.e. cuMemFreeHost()). The stream
//      ordered memory allocator can only allocate device memory (i.e. can
//      not be used for unified memory allocations).
//
//      Additionally, since cuMemAllocHost() is relatively expensive (it
//      must allocate page-locked memory, thus calls into the kernel)
//      we'll implement our own small block caching allocator. Allocations
//      will take a block from the free list if available, otherwise
//      actually allocate a block, and deallocations return the block the
//      the free list, thus avoiding an API call.
//
//      e.g. https://nvidia.github.io/cccl/cub/api/program_listing_file_cub_util_allocator.cuh.html
//
//   2. TODO: Regular pointers must be made accessible to the device via
//      cuMemHostGetDevicePointer(), and that address substituted into the
//      environment instead. The main difficulty here is that we need to
//      know the size of the allocation.
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
void UpdateClosureEnvironment(LLVMContext& Context, Module &M, Value* P, SmallVector<Instruction*> &ToErase, SmallPtrSet<Value*, N> &ToFree)
{
  if (AllocaInst* I = dyn_cast<AllocaInst>(P)) {
    auto TypeSize = I->getAllocationSize(M.getDataLayout());
    assert(TypeSize && "could not determine size of alloca");

    // Convert the stack allocation into a pinned memory allocation, so that the
    // memory is accessible from the device.
    //
    // Later, we also need to deallocate the memory. There are three possible
    // cases for handling this:
    //
    //   1. The memory is bracketed by llvm.stacksave/llvm.stackrestore
    //   2. The memory is marked as explicitly alive/dead by llvm.lifetime.start/llvm.lifetime.end respectively
    //   3. The memory is implicitly released when the function returns
    //
    IntegerType *i64_t = IntegerType::getInt64Ty(Context);
    ConstantInt *size = ConstantInt::get(i64_t, TypeSize->getFixedValue());
    Function *F = M.getFunction("cuda_caching_alloc_host");
    CallInst *NewI = CallInst::Create(F->getFunctionType(), F, {size});
    NewI->setCallingConv(CallingConv::Fast);
    ReplaceInstWithInst(I, NewI);
    ToFree.insert(NewI);

    // [Free 1]: Check for an llvm.stacksave() instruction
    if (auto *Prev = dyn_cast<Instruction>(NewI)->getPrevNonDebugInstruction()) {
      if (CallInst *Save = dyn_cast<CallInst>(Prev)) {
        if (Save->getCalledFunction()->getName().starts_with("llvm.stacksave")) {
          assert(Save->hasOneUser() && "expected llvm.stacksave() to have a single unique undroppable user");

          auto *Restore = dyn_cast<Instruction>(Save->getUniqueUndroppableUser());
          assert(Restore && dyn_cast<CallInst>(Restore)->getCalledFunction()->getName().starts_with("llvm.stackrestore"));

          Function *F = M.getFunction("cuda_caching_free_host");
          CallInst *Free = CallInst::Create(F->getFunctionType(), F, {NewI});
          Free->setCallingConv(CallingConv::Fast);
          Free->insertAfter(Restore);
          ToFree.erase(NewI);

          // Safe to erase these directly since users() will not be iterating
          // over them. XXX: The order is important here
          Restore->eraseFromParent();
          Save->eraseFromParent();
        }
      }
    }

    // Recursively update all users. This will also handle case [Free 2].
    for (auto *U : NewI->users()) {
      UpdateClosureEnvironment(Context, M, U, ToErase, ToFree);
    }

    // [Free 3] If the memory is still not freed, this will be handled once the
    // recursive update is complete.
  }

  else if (GetElementPtrInst *I = dyn_cast<GetElementPtrInst>(P)) {
    for (auto *U : I->users()) {
      UpdateClosureEnvironment(Context, M, U, ToErase, ToFree);
    }
  }

  else if (StoreInst *I = dyn_cast<StoreInst>(P)) {
    // XXX: There is a lot more that could (should) be done here, depending on
    // the type of the value operand. Pointer operands in particular are where
    // the difficulty lies.
    //
    //   1. If this an array pointer, should we register it as usable from the
    //      device, or require that the array memory is already (directly)
    //      accessible from the device? We should be able to recover the size of
    //      the array, but determining the size of the elements?
    //
    //   2. If this is a function pointer, then
    //
    //      (a) If we can determine (from the callsite of the parent function)
    //          what function is being called (i.e. rely on the compiler
    //          specialising this call), we should also lift that code out and
    //          compile it (i.e. beta-reduce) directly into the kernel.
    //
    //      (b) If we can't determine what function this is (i.e. it is just a
    //          raw function pointer, then..?
    //
    // Deciding what *kind* of pointer it is in the first place may already
    // challenging...
    UpdateClosureEnvironment(Context, M, I->getValueOperand(), ToErase, ToFree);
  }

  else if (CallInst *I = dyn_cast<CallInst>(P)) {
    if (Function* F = I->getCalledFunction()) {
      StringRef Name = F->getName();
      if (Name.starts_with("llvm.lifetime.start")) {
        // Assume that we will encounter the corresponding .end()
        ToErase.push_back(I);
      }
      else if (Name.starts_with("llvm.lifetime.end")) {
        // Assume that we will encounter the corresponding .start()
        Function *F = M.getFunction("cuda_caching_free_host");
        Value *Alloca = I->getArgOperand(1);
        CallInst *Free = CallInst::Create(F->getFunctionType(), F, {Alloca});
        Free->setCallingConv(CallingConv::Fast);
        Free->insertAfter(I);
        ToFree.erase(Alloca);
        ToErase.push_back(I);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "UpdateClosureEnvironment: unhandled instruction: " << *P << "\n");
}

// This is the entry point to UpdateClosureEnvironment(). This handles some
// setup and finalisation. The other overload of this function is the recursive
// one that does all the hard work.
//
void UpdateClosureEnvironment(LLVMContext& Context, Module &M, Value* P)
{
  SmallVector<Instruction*> ToErase;
  SmallPtrSet<Value*, 8> ToFree;
  Function *Parent = dyn_cast<Instruction>(P)->getFunction();

  // Recursively marshal the closure environment to device-accessible memory
  UpdateClosureEnvironment(Context, M, P, ToErase, ToFree);

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
    Function* F = M.getFunction("cuda_caching_free_host");
    Instruction *Ret = Returns.front();
    for (auto *Alloca : ToFree) {
      CallInst *Free = CallInst::Create(F->getFunctionType(), F, {Alloca});
      Free->setCallingConv(CallingConv::Fast);
      Free->insertBefore(Ret);
    }
  }
}

} // end of anonymous namespace


PreservedAnalyses ParallelForPass::run(Module &M, ModuleAnalysisManager &MAM)
{
  LLVMContext &Context = M.getContext();

  // Find all use sites of the parallel_for function.
  //
  // XXX: This is using the swift mangled name that happens to pop out from the
  // examples, but we'll obviously need to change this in the future to be more
  // robust. Most likely we'll put this in a separate swift package (containing
  // other scaffolding as well) in order to fix the name of this symbol.
  //
  //   > swift demangle 's4main12parallel_for10iterations_ySi_ySiXEtF'
  //   $s4main12parallel_for10iterations_ySi_ySiXEtF ---> main.parallel_for(iterations: Swift.Int, _: (Swift.Int) -> ()) -> ()
  //
  Function* Fseq = M.getFunction("$s4main12parallel_for10iterations_ySi_ySiXEtF");
  if (!Fseq) {
    LLVM_DEBUG(dbgs() << "No uses of function `parallel_for` found in this module\n");
    return PreservedAnalyses::all();
  }

  // We have found at least one call to parallel_for() that we will convert into
  // a parallel GPU kernel. First, add all the necessary host-side support code.
  LinkInHostSupportCode(M, Context);

  Function* Fpar = M.getFunction("launch_parallel_for");
  PointerType* ptr_t = PointerType::getUnqual(Context);
  IntegerType* i32_t = IntegerType::getInt32Ty(Context);
  StructType* kernel_t = StructType::create(Context, {ptr_t, ptr_t, ptr_t, i32_t, i32_t}, "parallel_for_kernel_t");

  // Iterate over all uses of the `parallel_for(iterations: body:)` function
  for (auto U = Fseq->user_begin(), UE = Fseq->user_end(); U != UE; /* See: [1] */) {
    if (CallInst *CI = dyn_cast<CallInst>(*U)) {
      Value* Iterations = CI->getArgOperand(0);
      Value* Body = CI->getArgOperand(1);
      Value* Env = CI->getArgOperand(2);

      GlobalValue* G = dyn_cast<GlobalValue>(Body);
      if (!G) {
        // TODO: We should be able to handle other forms; occasionally the
        // continuation will be passed as part of the closure environment and we
        // need to extract that by unwinding the environment.
        report_fatal_error("expected continuation function in call to `parallel_for`", false);
      }

      // The continuation launched might in turn call other functions.
      // Recursively record those functions for extraction as well.
      SetVector<GlobalValue *> GVs;
      GVs.insert(G);

      if (Function* F = dyn_cast<Function>(G)) {
        std::vector<Function *> WorkQueue;
        WorkQueue.push_back(F);

        while (!WorkQueue.empty()) {
          Function* F = &*WorkQueue.back();
          WorkQueue.pop_back();

          for (auto &BB : *F) {
            for (auto &I : BB) {
              if (CallBase* CB = dyn_cast<CallBase>(&I)) {
                if (Function* CF = CB->getCalledFunction()) {
                  if (!GVs.contains(CF)) {
                    GVs.insert(CF);
                    WorkQueue.push_back(CF);
                  }
                }
              }
            }
          }
        }
      }

      // Generate PTX assembly for the (set of) functions called by the
      // `parallel_for` launcher, and embed the generated code into the module
      SmallVector<char> Asm = CreateKernel(Body->getName(), GVs, Context);
      size_t buffer_size = Asm.size() + 1;  // include space for null terminator
      IntegerType* i8_t = IntegerType::getInt8Ty(Context);
      ArrayType* image_t = ArrayType::get(i8_t, buffer_size);

      std::vector<Constant*> KernelData(buffer_size);
      std::transform(Asm.begin(), Asm.end(), KernelData.begin(), [&](char c) { return ConstantInt::get(i8_t, c); });
      KernelData.back() = ConstantInt::get(i8_t, 0);

      GlobalVariable* Image = new GlobalVariable(M, image_t, true,
          GlobalValue::InternalLinkage,
          ConstantArray::get(image_t, ArrayRef(KernelData)),
          Body->getName() + "$image");  // XXX this will need to get more complicated as we do more specialisation
      Image->setAlignment(Align(1));
      Image->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

      ConstantInt* zero_c = ConstantInt::get(i32_t, 0);
      ConstantPointerNull* nullptr_c = ConstantPointerNull::get(ptr_t);
      GlobalVariable* Kernel = new GlobalVariable(M, kernel_t, false,
          GlobalValue::InternalLinkage,
          ConstantStruct::get(kernel_t, {Image, nullptr_c, nullptr_c , zero_c, zero_c}),
          Body->getName() + "$kernel");
      Kernel->setAlignment(Align(8));
      Kernel->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

      // NOTE [1]: Update the iterator for the next use of the `parallel_for`
      // function *before* updating the operand of the call site, otherwise the
      // iterator sequence will be lost.
      U++;

      // Update the calling instruction to our placeholder `parallel_for` to our
      // kernel launcher. This assumes that the environment is set up correctly,
      // which we will do in the next step.
      std::vector<Value*> params = {Iterations, Kernel, Env, nullptr_c};
      CallInst* CIpar = CallInst::Create(Fpar->getFunctionType(), Fpar, params);
      CIpar->addParamAttr(1, Attribute::NonNull);
      CIpar->addParamAttr(2, Attribute::NonNull);
      CIpar->insertBefore(CI);
      CIpar->setDebugLoc(CI->getDebugLoc());  // XXX: copy all metadata?
      CI->eraseFromParent();

      // Update the closure environment so that its contents are accessible from
      // the device.
      UpdateClosureEnvironment(Context, M, Env);
    }
  }

  return PreservedAnalyses::none();
}


/* ----------------------------------------------------------------------------
 * Registering passes as plugins
 *
 * https://llvm.org/docs/WritingAnLLVMNewPMPass.html#registering-passes-as-plugins
 */

llvm::PassPluginLibraryInfo getParallelForPluginInfo()
{
  return
    { LLVM_PLUGIN_API_VERSION
    , "lift-to-ptx<parallel_for>"
    , LLVM_VERSION_STRING
    , [](PassBuilder &PB) {
        PB.registerOptimizerLastEPCallback(
            [](ModulePassManager &PM, OptimizationLevel O) {
              PM.addPass(ParallelForPass());
            });
      }
    };
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize it when added to the pass pipeline on the
// command line, i.e. via '-passes=lift-to-ptx<parallel_for>'
//
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
  return getParallelForPluginInfo();
}

