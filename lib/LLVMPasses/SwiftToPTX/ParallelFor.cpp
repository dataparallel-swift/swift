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
#include "llvm/IR/Module.h"
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
#define DEBUG_CLONE_ALL_GLOBALS false

namespace {

static cl::opt<bool> Verbose (
  "swift-to-ptx-verbose", cl::Hidden, cl::init(false),
  cl::desc("Use verbose output"));

static cl::opt<bool> KeepIntermediateFiles (
  "swift-to-ptx-keep-intermediate-files", cl::Hidden, cl::init(false),
  cl::desc("Keep intermediate files of swift-to-ptx pass"));

static cl::opt<StringRef> PTXASPath (
  "swift-to-ptx-ptxas-path", cl::Hidden, cl::init("/usr/local/cuda/bin/ptxas"),
  cl::desc("Path to the ptxas executable"));

static cl::opt<StringRef> TargetGPU (
  "swift-to-ptx-target-gpu", cl::Hidden, cl::init("sm_87"),   // default: sm_87 (NVIDIA Jetson Orin)
  cl::desc("Target a specific GPU architecture in swift-to-ptx pass"));

static cl::opt<StringRef> TargetFeatures (
  "swift-to-ptx-target-attr", cl::Hidden, cl::init("+ptx81"), // default: +ptx81 (highest version supported by LLVM-17 and CUDA-12.2|L4T R36.3)
  cl::desc("Target specific attributes in swift-to-ptx pass"));

static cl::opt<bool> AllowFPArcp (
  "swift-to-ptx-allow-fp-arcp", cl::Hidden, cl::init(true),
  cl::desc("Allow floating-point division to be treated as multiplication by a reciprocal"));

static cl::opt<bool> AllowFPContract (
  "swift-to-ptx-allow-fp-contract", cl::Hidden, cl::init(true),
  cl::desc("Allow floating-point contraction, e.g. fusing a multiply followed by an addition into a fused multiply-add"));

static cl::opt<bool> AllowFPAfn (
  "swift-to-ptx-allow-fp-afn", cl::Hidden, cl::init(true),
  cl::desc("Allow substitution of approximate calculation for functions, e.g. sin, log, sqrt, etc."));

static cl::opt<bool> AllowFPReassoc (
  "swift-to-ptx-allow-fp-reassoc", cl::Hidden, cl::init(true),
  cl::desc("Allow re-association transformations for floating-point operations"));

static cl::opt<bool> StripDebugInfo (
  "swift-to-ptx-strip-debug-info", cl::Hidden, cl::init(false),
  cl::desc("Strip debug information from device code"));

static const MemoryBufferRef parallel_for_kernel = MemoryBufferRef(R"KERNEL(
; ModuleID = '<parallel_for_kernel>'
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: argmemonly nofree nosync nounwind
define void @parallel_for(i64 %iterations, ptr nonnull %env, ptr noalias nocapture swifterror dereferenceable(8) %swifterror, ptr nocapture readnone %thrownerror) local_unnamed_addr #0 {
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
  call swiftcc void @body(i64 %10, ptr nonnull %env, ptr noalias nocapture swifterror dereferenceable(8) %swifterror, ptr nocapture readnone %thrownerror)
  %11 = add i64 %10, %3
  %12 = icmp slt i64 %11, %iterations
  br i1 %12, label %while1.top, label %while1.exit

while1.exit:                                      ; preds = %while1.top, %entry
  ret void
}

define internal swiftcc void @body(i64 %0, ptr nonnull %1, ptr noalias nocapture swifterror dereferenceable(8) %2, ptr nocapture readnone %3) {
  ret void
}

define internal zeroext i1 @swift_isUniquelyReferenced_nonNull_native(ptr nonnull %0) {
  ret i1 true
}

define internal void @nanosleep(i32 %0) {
  tail call void asm sideeffect "nanosleep.u32 $0;", "r"(i32 %0) #3
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

; SwiftToPTX.parallel_for<A where A: Swift.Error>(iterations: Swift.Int, context: SwiftToPTX.Context, allocator: SwiftToPTX.CachingHostAllocator, stream: SwiftToPTX.Stream, _: (Swift.Int) throws(A) -> ()) throws(A) -> SwiftToPTX.Event
declare swiftcc ptr @"$s10SwiftToPTX12parallel_for10iterations7context9allocator6stream_AA5EventCSi_AA7ContextVAA20CachingHostAllocatorVAA6StreamVySixYKXEtxYKs5ErrorRzlF"(
    i64,                                                  ; iterations
    ptr, i64, i64,                                        ; SwiftToPTX.Context
    ptr, ptr, ptr,                                        ; SwiftToPTX.CachingHostAllocator
    ptr,                                                  ; execution stream
    ptr,                                                  ; body of the parallel_for loop
    ptr,                                                  ; closure environment
    ptr,                                                  ; type metadata for A
    ptr,                                                  ; protocol witness table for A
    ptr swiftself,                                        ; swift self
    ptr noalias nocapture swifterror dereferenceable(8),  ; swift error
    ptr nocapture readnone                                ; thrown error
  ) local_unnamed_addr #0

; SwiftToPTX.launch_parallel_for(iterations: Swift.Int, context: SwiftToPTX.Context, stream: SwiftToPTX.Stream, kernel: inout SwiftToPTX.ParallelForKernel, env: Swift.UnsafeMutableRawPointer, swifterror: Swift.UnsafeMutableRawPointer, thrownerror: Swift.UnsafeMutableRawPointer) -> SwiftToPTX.Event
declare swiftcc ptr @"$s10SwiftToPTX19launch_parallel_for10iterations7context6stream6kernel3env10swifterror11thrownerrorAA5EventCSi_AA7ContextVAA6StreamVAA17ParallelForKernelVzS3vtF"(
    i64,                                                  ; iterations
    ptr, i64, i64,                                        ; SwiftToPTX.Context
    ptr,                                                  ; execution stream
    ptr nocapture dereferenceable(32),                    ; SwiftToPTX.ParallelForKernel struct
    ptr,                                                  ; closure environment (updated to be accessible from the GPU)
    ptr noalias nocapture dereferenceable(8),             ; swift error (passed to kernel)
    ptr nocapture readnone                                ; thrown error (passed to kernel)
  ) local_unnamed_addr #0

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
// NOTE: Several of these mappings are disabled because they will be undone by
// later stages of the optimiser anyway.
//
const StringMap<StringRef> libdeviceFunctions =
  /* {{"llvm.abs.i64",         "__nv_llabs"} */
  /* ,{"llvm.abs.i32",         "__nv_abs"} */
  /* ,{"llvm.smax.i64",        "__nv_llmax"} */
  /* ,{"llvm.smax.i32",        "__nv_max"} */
  /* ,{"llvm.smin.i64",        "__nv_llmin"} */
  /* ,{"llvm.smin.i32",        "__nv_min"} */
  /* ,{"llvm.umax.i64",        "__nv_ullmax"} */
  /* ,{"llvm.umax.i32",        "__nv_umax"} */
  /* ,{"llvm.umin.i64",        "__nv_ullmin"} */
  /* ,{"llvm.umin.i32",        "__nv_umin"} */
  /* ,{"llvm.memcpy",          "" */
  /* ,{"llvm.memcpy.inline",   "" */
  /* ,{"llvm.memmove",         "" */
  /* ,{"llvm.memmove.inline",  "" */
  /* ,{"llvm.memset",          "" */
  /* ,{"llvm.memset.inline",   "" */
  {{"sqrt",                 "__nv_sqrt"}      /* ,{"llvm.sqrt.f64",        "__nv_sqrt"} */
  ,{"sqrtf",                "__nv_sqrtf"}     /* ,{"llvm.sqrt.f32",        "__nv_sqrtf"} */
  ,{"llvm.powi.f64.i32",    "__nv_powi"}
  ,{"llvm.powi.f32.i32",    "__nv_powif"}
  ,{"sin",                  "__nv_sin"}       /* ,{"llvm.sin.f64",         "__nv_sin"} */
  ,{"sinf",                 "__nv_sinf"}      /* ,{"llvm.sin.f32",         "__nv_sinf"} */
  ,{"cos",                  "__nv_cos"}       /* ,{"llvm.cos.f64",         "__nv_cos"} */
  ,{"cosf",                 "__nv_cosf"}      /* ,{"llvm.cos.f32",         "__nv_cosf"} */
  ,{"tan",                  "__nv_tan"}       /* ,{"llvm.tan.f64",         "__nv_tan"} */
  ,{"tanf",                 "__nv_tanf"}      /* ,{"llvm.tan.f32",         "__nv_tanf"} */
  ,{"asin",                 "__nv_asin"}      /* ,{"llvm.asin.f64",        "__nv_asin"} */
  ,{"asinf",                "__nv_asinf"}     /* ,{"llvm.asin.f32",        "__nv_asinf"} */
  ,{"acos",                 "__nv_acos"}      /* ,{"llvm.acos.f64",        "__nv_acos"} */
  ,{"acosf",                "__nv_acosf"}     /* ,{"llvm.acos.f32",        "__nv_acosf"} */
  ,{"atan",                 "__nv_atan"}      /* ,{"llvm.atan.f64",        "__nv_atan"} */
  ,{"atanf",                "__nv_atanf"}     /* ,{"llvm.atan.f32",        "__nv_atanf"} */
  ,{"atan2",                "__nv_atan2"}     /* ,{"llvm.atan2.f64",       "__nv_atan"} */
  ,{"atan2f",               "__nv_atan2f"}    /* ,{"llvm.atan2.f32",       "__nv_atan2f"} */
  ,{"sinh",                 "__nv_sinh"}      /* ,{"llvm.sinh.f64",        "__nv_sinh"} */
  ,{"sinhf",                "__nv_sinhf"}     /* ,{"llvm.sinh.f32",        "__nv_sinhf"} */
  ,{"cosh",                 "__nv_cosh"}      /* ,{"llvm.cosh.f64",        "__nv_cosh"} */
  ,{"coshf",                "__nv_coshf"}     /* ,{"llvm.cosh.f32",        "__nv_coshf"} */
  ,{"tanh",                 "__nv_tanh"}      /* ,{"llvm.tanh.f64",        "__nv_tanh"} */
  ,{"tanhf",                "__nv_tanhf"}     /* ,{"llvm.tanh.f32",        "__nv_tanhf"} */
  ,{"asinh",                "__nv_asinh"}
  ,{"asinhf",               "__nv_asinhf"}
  ,{"acosh",                "__nv_acosh"}
  ,{"acoshf",               "__nv_acoshf"}
  ,{"atanh",                "__nv_atanh"}
  ,{"atanhf",               "__nv_atanhf"}
  ,{"pow",                  "__nv_pow"}       /* ,{"llvm.pow.f64",         "__nv_pow"} */
  ,{"powf",                 "__nv_powf"}      /* ,{"llvm.pow.f32",         "__nv_powf"} */
  ,{"exp",                  "__nv_exp"}       /* ,{"llvm.exp.f64",         "__nv_exp"} */
  ,{"expf",                 "__nv_expf"}      /* ,{"llvm.exp.f32",         "__nv_expf"} */
  ,{"expm1",                "__nv_expm1"}
  ,{"expm1f",               "__nv_expm1f"}
  ,{"exp2",                 "__nv_exp2"}      /* ,{"llvm.exp2.f64",        "__nv_exp2"} */
  ,{"exp2f",                "__nv_exp2f"}     /* ,{"llvm.exp2.f32",        "__nv_exp2f"} */
  ,{"exp10",                "__nv_exp10"}     /* ,{"llvm.exp10.f64",       "__nv_exp10"} */
  ,{"exp10f",               "__nv_exp10f"}    /* ,{"llvm.exp10.f32",       "__nv_exp10f"} */
  ,{"llvm.ldexp.f64.i32",   "__nv_ldexp"}
  ,{"llvm.ldexp.f32.i32",   "__nv_ldexp"}
  ,{"llvm.frexp.f64.i32",   "__nv_frexp"}
  ,{"llvm.frexp.f32.i32",   "__nv_frexpf"}
  ,{"log",                  "__nv_log"}       /* ,{"llvm.log.f64",         "__nv_log"} */
  ,{"logf",                 "__nv_logf"}      /* ,{"llvm.log.f32",         "__nv_logf"} */
  ,{"log1p",                "__nv_log1p"}
  ,{"log1pf",               "__nv_log1pf"}
  ,{"log10",                "__nv_log10"}     /* ,{"llvm.log10.f64",       "__nv_log10"} */
  ,{"log10f",               "__nv_log10f"}    /* ,{"llvm.log10.f32",       "__nv_log10f"} */
  ,{"log2",                 "__nv_log2"}      /* ,{"llvm.log2.f64",        "__nv_log2"} */
  ,{"log2f",                "__nv_log2f"}     /* ,{"llvm.log2.f32",        "__nv_log2f"} */
  ,{"llvm.fma.f64",         "__nv_fma"}
  ,{"llvm.fma.f32",         "__nv_fmaf"}
  /* ,{"llvm.fabs.f64",        "__nv_fabs"} */
  /* ,{"llvm.fabs.f32",        "__nv_fabsf"} */
  ,{"llvm.minnum.f64",      "__nv_fmin"}
  ,{"llvm.minnum.f32",      "__nv_fminf"}
  ,{"llvm.maxnum.f64",      "__nv_fmax"}
  ,{"llvm.maxnum.f32",      "__nv_fmaxf"}
  /* ,{"llvm.minimum.*",       ""} */
  /* ,{"llvm.maximum.*",       ""} */
  ,{"llvm.copysign.f64",    "__nv_copysign"}
  ,{"llvm.copysign.f32",    "__nv_copysignf"}
  /* ,{"llvm.floor.f64",       "__nv_floor"} */
  /* ,{"llvm.floor.f32",       "__nv_floorf"} */
  /* ,{"llvm.ceiling.f64",     "__nv_ceiling"} */
  /* ,{"llvm.ceiling.f32",     "__nv_ceilingf"} */
  /* ,{"llvm.trunc.f64",       "__nv_trunc"} */
  /* ,{"llvm.trunc.f32",       "__nv_truncf"} */
  ,{"llvm.rint.f64",        "__nv_rint"}
  ,{"llvm.rint.f32",        "__nv_rintf"}
  ,{"llvm.nearbyint.f64",   "__nv_nearbyint"}
  ,{"llvm.nearbyint.f32",   "__nv_nearbyintf"}
  /* ,{"llvm.round.f64",       "__nv_round"} */
  /* ,{"llvm.round.f32",       "__nv_roundf"} */
  /* ,{"llvm.roundeven.*",     ""} */
  /* ,{"llvm.lround.*",        ""} */
  ,{"llvm.llround.i64.f64", "__nv_llround"}
  ,{"llvm.llround.i64.f32", "__nv_llroundf"}
  /* ,{"llvm.lrint.*",         ""} */
  ,{"llvm.llrint.i64.f64",  "__nv_llrint"}
  ,{"llvm.llrint.i64.f32",  "__nv_llrintf"}
  /* ,{"llvm.bitreverse.*",    ""} */
  /* ,{"llvm.bswap.*",         ""} */
  /* ,{"llvm.ctpop.i64",       "__nv_popcll"} */
  /* ,{"llvm.ctpop.i32",       "__nv_popc"} */
  /* ,{"llvm.ctlz.i64",        "__nv_clzll"} */
  /* ,{"llvm.ctlz.i32",        "__nv_clz"} */
  /* ,{"llvm.cttz.*",          ""} */
  /* ,{"llvm.fshl.*",          ""} */
  /* ,{"llvm.fshr.*",          ""} */
  ,{"erf",                  "__nv_erf"}
  ,{"erff",                 "__nv_erff"}
  ,{"erfc",                 "__nv_erfc"}
  ,{"erfcf",                "__nv_erfcf"}
  ,{"tgamma",               "__nv_tgamma"}
  ,{"tgammaf",              "__nv_tgammaf"}
  ,{"lgamma_r",             "__nv_lgamma"}
  ,{"lgammaf_r",            "__nv_lgammaf"}
  ,{"hypot",                "__nv_hypot"}
  ,{"hypotf",               "__nv_hypotf"}
  };


const StringMap<StringRef> stubFunctions =
  {{"swift_isUniquelyReferenced_nonNull_native",  "swift_isUniquelyReferenced_nonNull_native"}
  ,{"$s10SwiftToPTX9nanosleepyys6UInt32VF",       "nanosleep"}
  };


// Swift will apply scalar replacement of aggregates in order to pass struct
// (components) in registers for function calls.
//
// Somewhat ironically, we repackage those components again to make it a bit
// more convenient to work with, and hope that the (C++) compiler again does the
// same thing to make our function calls more efficient. We should really check
// whether it does...
typedef std::tuple<Value*, Value*, Value*> CUDAContext;
typedef std::tuple<Value*, Value*, Value*> CachingHostAllocator;


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
    report_fatal_error("could not open libdevice module", false);
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
    SmallVector<ReturnInst *> Returns;
    CloneFunctionInto(Dst, &Src, VMap, CloneFunctionChangeType::DifferentModule, Returns);
  }
}

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
    report_fatal_error("pipe() error", false);
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
    report_fatal_error("execv() failed (" + Twine(errno) + ")", false);
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
    size_t   offset;
    size_t   capacity;
    uint8_t* obj_buffer;
    char*    msg_buffer;

    // Read in the compiled object code
    offset = 0;
    capacity = 65536; // 64KB, the smallest functions are around ~22KB (with debug info)
    obj_buffer = (uint8_t*) malloc(capacity);
    while (true) {
      if (!obj_buffer)
        report_fatal_error("malloc() error", false);

      ssize_t rv = read(fd1[0], obj_buffer + offset, capacity - offset);
      if (rv == 0)
        break;  // child closed pipe

      if (rv < 0)
        report_fatal_error("pipe() error", false);

      offset += rv;
      if (offset == capacity) {
        // Not a great multiplicative factor to choose as it causes
        // heap fragmentation, but it won't live for much longer anyway
        capacity  *= 2;
        obj_buffer = (uint8_t*) realloc(obj_buffer, capacity);
      }
    }
    ArrayRef<uint8_t> obj = ArrayRef(obj_buffer, offset);

    // Read in any error/warning messages
    offset = 0;
    capacity = 1024;  // 1KB, should be more than large enough
    msg_buffer = (char*) malloc(capacity);
    while (true) {
      if (!msg_buffer)
        report_fatal_error("malloc() error", false);

      ssize_t rv = read(fd2[0], msg_buffer + offset, capacity - offset);
      if (rv == 0)
        break;  // child closed pipe

      if (rv < 0)
        report_fatal_error("pipe() error", false);

      offset += rv;
      if (offset == capacity) {
        capacity  *= 2; // see comment above
        msg_buffer = (char*) realloc(msg_buffer, capacity);
      }
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

    // Information about register usage etc. of the compiled kernel
    if (Verbose)
      errs() << msg;

    LLVM_DEBUG(dbgs() << msg);
    free(msg_buffer);

    return obj;
  }
}

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
    Module& M,
    StringRef Main,
    SmallPtrSetImpl<GlobalValue*>& GVs,
    SmallPtrSetImpl<GlobalValue*>& DeclOnlyGVs,
    ValueToValueMapTy& IndirectMap
)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> K = parseAssembly(parallel_for_kernel, Err, Context);
  if (!K) {
    Err.print("swift-to-ptx<parallel_for>", errs());
    exit(1);
  }

  // Initially populate the VMap with operations that will be handled by the
  // kernel skeleton, rather than copied over from the source module.
  ValueToValueMapTy VMap;
  for (auto &[Src,Dst] : stubFunctions) {
    if (auto F = M.getFunction(Src))
      VMap[F] = K->getFunction(Dst);
  }

#if DEBUG_CLONE_ALL_GLOBALS
  // Loop over all of the global variables, making corresponding globals in the
  // new module. Here we add them to the VMap and to the new Module. We don't
  // worry about attributes or initialisers yet, those will come later.
  for (auto &I : M.globals()) {
    if (I.getName() == "llvm.used")
      continue;

    GlobalVariable *GV = new GlobalVariable(*K, I.getValueType(), I.isConstant(), I.getLinkage(), nullptr, I.getName(), nullptr, I.getThreadLocalMode(), I.getType()->getAddressSpace());
    GV->copyAttributesFrom(&I);
    VMap[&I] = GV;
  }
#endif

  // Loop over all of the globals that we need to copy from the input module.
  // Just make the declarations, the bodies will come later. Take care of
  // functions that need to be handled specially on the device.
  bool HaveLibdevice = false;
  DeclOnlyGVs.insert(GVs.begin(), GVs.end());
  for (auto I : DeclOnlyGVs) {
    if (auto Src = dyn_cast<Function>(I)) {
      // If this is a declaration for a function provided by libdevice (e.g.
      // llvm.sin.f32) then link in the libdevice module and record in the VMap
      // the mapping to the corresponding libdevice implementation (e.g. __nvsinf).
      //
      // Delay linking in libdevice to the point where we are certain we need
      // it, which saves a few hundred ms in case it would not have been used.
      StringRef Name = Src->getName();
      StringRef Lib  = libdeviceFunctions.lookup(Name);
      if (!Lib.empty()) {
        if (!HaveLibdevice) {
          LinkInLibdeviceModule(*K, Context);
          HaveLibdevice = true;
        }
        VMap[Src] = K->getFunction(Lib);
        continue;
      }

      // Otherwise, this is just a regular function (declaration). Just make the
      // declaration, we'll copy over the function body later.
      Function* Dst = Function::Create(Src->getFunctionType(), Src->getLinkage(), Name, *K);
      Dst->copyAttributesFrom(Src);
      VMap[Src] = Dst;
    }

    if (auto Src = dyn_cast<GlobalVariable>(I)) {
      GlobalVariable *Dst = new GlobalVariable(*K, Src->getValueType(), Src->isConstant(), Src->getLinkage(), nullptr, Src->getName(), nullptr, Src->getThreadLocalMode(), Src->getType()->getAddressSpace());
      Dst->copyAttributesFrom(Src);
      VMap[Src] = Dst;
    }
  }

#if DEBUG_CLONE_ALL_GLOBALS
  // Loop over the aliases...
  for (auto &I : M.aliases()) {
    GlobalAlias *GA = GlobalAlias::create(I.getValueType(), I.getType()->getPointerAddressSpace(), I.getLinkage(), I.getName(), K.get());
    GA->copyAttributesFrom(&I);
    VMap[&I] = GA;
  }

  // ...and indirect functions in the module
  for (auto &I : M.ifuncs()) {
    // Defer setting the resolver function until after functions are cloned.
    GlobalIFunc *GI = GlobalIFunc::create(I.getValueType(), I.getAddressSpace(), I.getLinkage(), I.getName(), nullptr, K.get());
    GI->copyAttributesFrom(&I);
    VMap[&I] = GI;
  }

  // Now that all the things that a global variable initialiser can refer to
  // have been created, loop through and copy the global variable referees over.
  // Also set the attributes on the global now.
  for (auto &I : M.globals()) {
    GlobalVariable* GV = cast_if_present<GlobalVariable>(VMap[&I]);
    if (!GV)
      continue;

    SmallVector<std::pair<unsigned, MDNode*>> MDs;
    I.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    if (I.isDeclaration())
      continue;

    if (I.hasInitializer())
      GV->setInitializer(MapValue(I.getInitializer(), VMap));
  }
#endif

  // Now that all things that a global variable initializer can refer to have
  // been created, copy any initialisers over. For any functions we copy over
  // the body of the function now, as well as enable floating point contraction
  // for compatible instructions and specialise any indirect function calls.
  for (auto I : GVs) {
    // If this is only a declaration, we are done
    if (I->isDeclaration())
      continue;

    // Copy global variable initialisers
    if (auto Src = dyn_cast<GlobalVariable>(I)) {
      GlobalVariable* Dst = cast_if_present<GlobalVariable>(VMap[Src]);
      if (!Dst)
        continue;

      if (Src->hasInitializer())
        Dst->setInitializer(MapValue(Src->getInitializer(), VMap));
    }

    // Copy function bodies and add any specialisations
    if (auto Src = dyn_cast<Function>(I)) {
      auto Dst = cast_if_present<Function>(VMap[Src]);
      if (!Dst)
        continue;

      Function::arg_iterator DstI = Dst->arg_begin();
      for (const Argument &I : Src->args()) {
        DstI->setName(I.getName());
        VMap[&I] = &*DstI++;
      }
      SmallVector<ReturnInst *> Returns;
      CloneFunctionInto(Dst, Src, VMap, CloneFunctionChangeType::DifferentModule, Returns);
      /* TypeContextRemapper TypeMapper(Context); */
      /* PointerAddressSpaceUpdater TypeMapper; */
      /* CloneFunctionInto(Dst, Src, VMap, CloneFunctionChangeType::DifferentModule, Returns, "", nullptr, &TypeMapper); */
      Dst->setCallingConv(Src->getCallingConv());
      Dst->setLinkage(GlobalValue::InternalLinkage);

      if (Src->hasPersonalityFn())
        Dst->setPersonalityFn(MapValue(Src->getPersonalityFn(), VMap));

      // Update any per-instruction attributes
      for (auto &BB : *Dst) {
        for (auto &I : BB) {
          // Set any floating-point rewrite rules
          if (isa<FPMathOperator>(&I)) {
            I.setHasAllowReciprocal(AllowFPArcp);
            I.setHasAllowContract(AllowFPContract);
            I.setHasApproxFunc(AllowFPAfn);
            I.setHasAllowReassoc(AllowFPReassoc);
          }

          // Allow function calls to be inlined. This attribute tends to creep in
          // because we currently need to sprinkle @inline(never) in places to
          // ensure that all Swift code for the GPU code is present in a single
          // LLVM module.
          if (auto CB = dyn_cast<CallBase>(&I)) {
            CB->removeFnAttr(Attribute::NoInline);
          }
        }
      }
    }
  }

  // Now that the function bodies have been copied over, update any indirect
  // calls to point directly to their called functions that were extracted from
  // the closure environment.
  for (auto [Src,Dst] : IndirectMap) {
    if (auto I = dyn_cast<CallBase>(VMap[Src])) {
      if (auto J = VMap[Dst]) {
        I->setCalledOperand(J);
      }
    }
  }

#if DEBUG_CLONE_ALL_GLOBALS
  // Copy over any remaining definitions...
  for (auto &I : M.aliases()) {
    GlobalAlias* GA = cast<GlobalAlias>(VMap[&I]);
    if (const Constant *C = I.getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  // ...indirect functions...
  for (auto &I : M.ifuncs()) {
    GlobalIFunc *GI = cast<GlobalIFunc>(VMap[&I]);
    if (const Constant *Resolver = I.getResolver())
      GI->setResolver(MapValue(Resolver, VMap));
  }

  // ...and named metadata
  for (auto &I : M.named_metadata()) {
    NamedMDNode *NMD = K->getOrInsertNamedMetadata(I.getName());
    for (const MDNode *MD : I.operands())
      NMD->addOperand(MapMetadata(MD, VMap));
  }
#endif

  // Replace swift error handling functions with equivalents that we can call
  // from the device
  if (Function* _fatalErrorMessage = K->getFunction("$ss18_fatalErrorMessage__4file4line5flagss5NeverOs12StaticStringV_A2HSus6UInt32VtF")) {
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

        auto __assertfail = K->getFunction("__assertfail");
        CallInst* CINew = CallInst::Create(__assertfail->getFunctionType(), __assertfail,
            { newStaticString(Context, *K, Prefix->str() + (Message ? ": " + Message->str() : ""))
            , newStaticString(Context, *K, File->str())
            , ConstantInt::get(IntegerType::getInt32Ty(Context), Line)
            , newStaticString(Context, *K, demangleSymbolAsString(CI->getCaller()->getName(), swift::Demangle::DemangleOptions()))
            , ConstantInt::get(IntegerType::getInt64Ty(Context), 1)
            });

        ReplaceInstWithInst(CI, CINew);
      }
    }
  }

  // Strip debug information from device code. This may be necessary on some
  // combinations of Swift/CUDA due to bugs in LLVM. Debug information will only
  // be present if already enabled as part of the swift compilation pipeline
  // (default), it will not be generated as part of this plugin.
  if (StripDebugInfo)
    llvm::StripDebugInfo(*K);

  // Update the kernel function to call the main (entry) function from the set
  // that we extracted in the previous step.
  Function *Body = K->getFunction("body");
  assert(Body->hasOneUser() && "expected only one call to the kernel body");
  assert(isa<CallInst>(Body->getUniqueUndroppableUser()));
  CallInst *CI = cast<CallInst>(Body->getUniqueUndroppableUser());
  CI->setCalledOperand(K->getFunction(Main));

  // Create a target machine
  std::string Error;
  auto TargetTriple = "nvptx64-nvidia-cuda";
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    report_fatal_error(StringRef(Error), false);
  }
  TargetOptions opt;
  TargetMachine* TargetMachine = Target->createTargetMachine(TargetTriple, TargetGPU, TargetFeatures, opt, Reloc::PIC_);
  K->setDataLayout(TargetMachine->createDataLayout());

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

  PM.run(*K, MAM);

  // Generate target assembly. Being a backend/code generation pass, this
  // uses the legacy pass manager so does not integrate with the above.
  legacy::PassManager legacy;
  SmallVector<char> Asm;  // XXX: reserve space to avoid growing too frequently?
  raw_svector_ostream ostream(Asm);
  if (TargetMachine->addPassesToEmitFile(legacy, ostream, nullptr, CodeGenFileType::AssemblyFile)) {
    report_fatal_error("could not create output stream", false);
  }
  legacy.run(*K);

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
      report_fatal_error("failed to create output file", false);
    }
    // XXX: Note that we need to dump the generated IR from a _different_
    // pass manager, otherwise generating target assembly with the old pass
    // manager will cause Bad Things™ to happen.
    ModulePassManager PM;
    auto src_out = raw_fd_ostream(src_fd, true);
    PM.addPass(PrintModulePass(src_out));
    PM.run(*K, MAM);
    src_out.close();
    errs() << src_path << "\n";

    int ptx_fd = 0;
    SmallVector<char> ptx_path;
    if (sys::fs::createTemporaryFile("kernel", "ptx", ptx_fd, ptx_path)) {
      report_fatal_error("failed to create output file", false);
    }
    auto ptx_out = raw_fd_ostream(ptx_fd, true);
    ptx_out << Asm;
    ptx_out.close();
    errs() << ptx_path << "\n";

    int obj_fd = 0;
    SmallVector<char> obj_path;
    if (sys::fs::createTemporaryFile("kernel", "o", obj_fd, obj_path)) {
      report_fatal_error("failed to create output file", false);
    }
    auto obj_out = raw_fd_ostream(obj_fd, true);
    obj_out.write((const char*) Obj.data(), Obj.size());
    obj_out.close();
    errs() << obj_path << "\n";
  }

  return Obj;
}

// XXX: This is a bit lazy; we could encode this on the actual call stack as an
// intrusive linked list by transforming ExtractKernel into a recursive
// function, and avoid the overhead of allocating/copying this vector.
// However, SmallVector should avoid the allocation for small N by storing the
// values in the object itself, so in practice it may not be necessary. We need
// to benchmark this, and test how deep the call stack is in practice.
//    --- TLM 2025-03-24
typedef SmallVector<CallBase*> CallStack;

bool isEquivalentGEP(const GetElementPtrInst* A, const GetElementPtrInst* B)
{
  if (A->getNumIndices() != B->getNumIndices())
    return false;

  if (A->getSourceElementType() != B->getSourceElementType())
    return false;

  for (unsigned i = 1; i < A->getNumOperands(); ++i) {
    auto a = A->getOperand(i);
    auto b = B->getOperand(i);

    if (!isa<ConstantInt>(a))
      return false;

    if (!isa<ConstantInt>(b))
      return false;

    if (a != b)
      return false;
  }

  return true;
}

// Upsweep phase, where we traverse the environment passed to the parallel_for,
// following each GEP we encountered in the downsweep phase
Value* getValueFromClosure(SmallVector<GetElementPtrInst*>& Indices, Value* V)
{
  if (auto I = dyn_cast<AllocaInst>(V)) {
    for (auto U : I->users()) {
      if (!isa<GetElementPtrInst>(U))
        continue;

      if (auto R = getValueFromClosure(Indices, U))
        return R;
    }
  }

  if (auto I = dyn_cast<GetElementPtrInst>(V)) {
    auto E = Indices.back();

    if (!isEquivalentGEP(E, I))
      return nullptr;

    assert(I->hasOneUser());

    Indices.pop_back();
    return getValueFromClosure(Indices, I->getUniqueUndroppableUser());
  }

  if (auto I = dyn_cast<StoreInst>(V)) {
    if (Indices.empty()) {
      // We can probably delete the store at this point, since we will probably
      // be using this value directly now rather than reading it from the
      // captured closure environment.
      /* I->eraseFromParent(); */
      return I->getValueOperand();
    } else {
      return getValueFromClosure(Indices, I->getValueOperand());
    }
  }

  return nullptr;
}

// Downsweep phase, where we build a stack of indices into the closure
// environment until we reach the parallel_for invocation
Value* getValueFromClosure(CallStack& CS, SmallVector<GetElementPtrInst*>& Indices, Value* Env, Value* V)
{
  if (auto I = dyn_cast<Argument>(V)) {
    // This was an argument to a function call. Trace up the call stack to
    // examine the calling function. Once we reach the top of the stack---the
    // call to parallel_for---then we can work our way back up the indices stack
    // poking through the environment.
    auto CB = CS.back();
    CS.pop_back();

    if (CS.empty())
      return getValueFromClosure(Indices, Env);
    else
      return getValueFromClosure(CS, Indices, Env, CB->getOperand(I->getArgNo()));
  }

  if (auto I = dyn_cast<GlobalValue>(V)) {
    return I;
  }

  if (auto I = dyn_cast<LoadInst>(V)) {
    return getValueFromClosure(CS, Indices, Env, I->getPointerOperand());
  }

  if (auto I = dyn_cast<GetElementPtrInst>(V)) {
    Indices.push_back(I);
    return getValueFromClosure(CS, Indices, Env, I->getPointerOperand());
  }

  return nullptr;
}


// Entry point to peek through the closure environment looking for values that
// were stored in it that we can specialise on (namely, indirect function
// calls). Note that this version takes a copy of the callstack, so that we are
// free to mutate it in the recursive invocations that do the actual work.
//
// TLM: I still don't think this is right. We have two pointers that we want to
// walk down, one from the closure (callee) side, where we look for LOADs at some
// GEP; and one from the parallel_for (caller) side, where we look for STOREs at
// the corresponding GEP. Right now we walk up the entire call stack collecting
// all the GEPs we find along the way, but I think this only works in the simple
// case (no nested closures), and because we have an early exit if we encounter
// a global value already. I think we should instead take a single "step" from
// each end at a time, and keep going until we find a global value? But I'm not
// sure if that is the correct stopping condition or not. ---TLM 2025-04-02
//
Value* getValueFromClosure(CallStack CS, Value* Env, Value* V)
{
  SmallVector<GetElementPtrInst*> Indices;
  return getValueFromClosure(CS, Indices, Env, V);
}

void ExtractKernel (
    LLVMContext& Context,
    CallBase* Root,
    Value* Main,
    Value* Env,
    SmallPtrSetImpl<GlobalValue*>& GVs,
    SmallPtrSetImpl<GlobalValue*>& DeclOnlyGVs,
    ValueToValueMapTy& IndirectMap,
    ValueToValueMapTy& CopyOnWriteMap
)
{
  auto* F = cast<Function>(Main);
  GVs.insert(F);

  // The continuation launched might in turn call other functions. Recursively
  // record those functions for extraction as well.
  std::vector<std::pair<CallStack, Function*>> WorkQueue;
  WorkQueue.push_back({{Root}, F});

  while (!WorkQueue.empty()) {
    auto [CS, F] = WorkQueue.back();
    WorkQueue.pop_back();

    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function* CF = nullptr;

          if (CB->isIndirectCall()) {
            CF = dyn_cast_if_present<Function>(getValueFromClosure(CS, Env, CB->getCalledOperand()));

            if (!CF)
              report_fatal_error("swift-to-ptx: could not specialise indirect function call", false);

            IndirectMap[CB] = CF;
          } else {
            CF = CB->getCalledFunction();

            // Inline assembly
            if (!CF)
              continue;

            // Find the corresponding copy-on-write handler for this object. We
            // will lift this functionality out of the kernel and execute it as
            // part of closure environment preparation. Also don't bother to
            // copy the handler into the kernel, as it will ultimately be
            // removed as dead code anyway.
            //
            // TODO: At the end of the scope there will (may?) be a now
            // redundant store instruction in the kernel, which saves the data
            // pointer again because it may have been updated by the CoW
            // mechanism. We should remove this.
            //    ---TLM 2025-02-17
            //
            // TODO: Should probably split this out, it's a bit of a mess...
            //    ---TLM 2025-03-25
            if (CF->getName() == "swift_isUniquelyReferenced_nonNull_native") {
              auto Op = CB->getOperand(0);
              for (auto U : Op->users()) {
                if (U == CB)
                  continue;

                if (auto I = dyn_cast<CallBase>(U)) {
                  DeclOnlyGVs.insert(I->getCalledFunction());

                  if (auto V = getValueFromClosure(CS, Env, Op)) {
                    // Clone the copy-on-write handler and update the arguments
                    // so that it is callable from elsewhere.
                    auto Dst = cast<CallBase>(I->clone());

                    for (unsigned i = 0; i < Dst->arg_size(); ++i) {
                      auto A = Dst->getArgOperand(i);

                      // Indicate the argument we'll need to update later. We
                      // can't replace this with null for some reason, so use
                      // the key, which we'll also have access to later.
                      if (Op == A) {
                        Dst->setArgOperand(i, V);
                      }

                      // All arguments to the function must either be a
                      // constant, or something we can access/specialise from
                      // the closure environment.
                      if (!isa<Constant>(A)) {
                        if (auto R = getValueFromClosure(CS, Env, A)) {
                          Dst->setArgOperand(i, R);
                        } else {
                          LLVM_DEBUG(dbgs() << "copy-on-write handler possibly given invalid argument");
                        }
                      }
                    }
                    CopyOnWriteMap[V] = Dst;
                  }
                  break;
                }
              }
              continue;
            }

            // Calls to certain functions from the swift-to-ptx prelude are just
            // stub implementations, with the actual functionality provided as
            // part of the kernel skeleton. These are typically functions that
            // exist only on the GPU, such as warp synchronisation primitives.
            StringRef stub = stubFunctions.lookup(CF->getName());
            if (!stub.empty())
              continue;
          }

          if (CF) {
            if (!GVs.contains(CF) && !DeclOnlyGVs.contains(CF)) {
              GVs.insert(CF);
              if (!CF->isDeclaration()) {
                CallStack NewCS = CallStack(CS);
                NewCS.push_back(CB);
                WorkQueue.push_back({NewCS,CF});
              }
            }
          } else {
            LLVM_DEBUG(dbgs() << "Unhandled function call: " << *CB << "\n");
          }
        }

        if (auto *L = dyn_cast<LoadInst>(&I)) {
          if (auto G = dyn_cast<GlobalVariable>(L->getPointerOperand())) {
            if (!GVs.contains(G)) {
              GVs.insert(G);
            }
          }
        }
      }
    }
  }
}

Instruction* ApplyCopyOnWriteHandler (
    LLVMContext& Context,
    Module& M,
    Value* V,
    Instruction* Handler,
    Instruction* InsertBefore,
    Value* StoreAt=nullptr
)
{
    // Add a uniqueness check to determine whether we need to run the
    // copy-on-write handler or not.
    auto F = M.getFunction("swift_isUniquelyReferenced_nonNull_native");
    auto IsUnique = CallInst::Create(F->getFunctionType(), F, { V }, "", InsertBefore);
    IsUnique->setTailCall(true);

    // Split the original basic block, inserting an else-branch to run the
    // handler. Optionally also store this updated value at the given location
    // (only in the else branch).
    auto NewBB = SplitBlockAndInsertIfElse(IsUnique, InsertBefore, /* unreachable */ false);
    Handler->insertBefore(NewBB);
    if (StoreAt)
      new StoreInst(Handler, StoreAt, NewBB);

    // Tie the branches together...
    PHINode* PHI = PHINode::Create(V->getType(), 2, "", InsertBefore);
    PHI->addIncoming(V, IsUnique->getParent());
    PHI->addIncoming(Handler, Handler->getParent());

    // ...and replace any uses of the original term with our now safely handled
    // version (only uses which are dominated by it)
    DominatorTree DT(*PHI->getFunction());
    V->replaceUsesWithIf(PHI, [&](Use &U){
        return DT.dominates(PHI, U);
        });

    return PHI;
}


void UpdateClosureEnvironment (
    LLVMContext& Context,
    Module& M,
    Value* Env,
    ValueToValueMapTy& CopyOnWriteMap
)
{
  for (auto [Src,Dst] : CopyOnWriteMap) {
    auto SrcI = const_cast<Value*>(Src);
    auto I = cast<CallBase>(Dst);

    // This was created in the local scope
    if (auto A = dyn_cast<AllocaInst>(SrcI)) {
      for (auto U : A->users()) {
        if (auto S = dyn_cast<StoreInst>(U)) {
          if (S->getPointerOperand() != A)
            continue;

          auto T = S->getValueOperand();
          for (unsigned i = 0; i < I->arg_size(); ++i) {
            if (I->getArgOperand(i) == Src) {
              I->setArgOperand(i, T);
            }
          }

          auto Q = ApplyCopyOnWriteHandler(Context, M, T, I, S);
          S->setOperand(0, Q);  // value operand
          I->dropLocation();    // must be called on instruction with a parent

        }
      }
      continue;
    }

    // This was given as a function parameter
    if (auto A = dyn_cast<Argument>(SrcI)) {
      for (auto U : A->users()) {
        if (auto L = dyn_cast<LoadInst>(U)) {
          for (unsigned i = 0; i < I->arg_size(); ++i) {
            if (I->getArgOperand(i) == Src)
              I->setOperand(i, L);
          }

          auto Q = ApplyCopyOnWriteHandler(Context, M, L, I, L->getNextNonDebugInstruction(), A);
          I->dropLocation();

          // The load that this function argument is stored in may have been
          // allocated in non-device-accessible memory by the calling
          // function (i.e. an alloca). Replace the input argument with a
          // locally defined alloca and replace subsequent uses of the input
          // argument with this pointer. The subsequent steps of closure
          // conversion will translate this into a host memory
          // (de)allocation that is device accessible.
          auto InsertBefore = Q->getNextNonDebugInstruction();
          auto NewA = new AllocaInst(L->getPointerOperandType(), L->getPointerAddressSpace(), "", InsertBefore);
          new StoreInst(Q, NewA, InsertBefore);

          // Also replace uses with this new alloca
          DominatorTree DT(*Q->getFunction());
          A->replaceUsesWithIf(NewA, [&](Use &U){
              return DT.dominates(NewA, U);
              });
          break;
        }
      }
      continue;
    }
  }
}

// Update the environment so that its contents are accessible from the device.
// There are a few different cases to handle:
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
//      UPDATE: Because I ran into issues with making host heap allocations
//      ex-post-facto accessible to the device, we have instead changed the
//      swift runtime such that the underlying allocator can be replaced with
//      the CUDA allocator by the swift-to-ptx library. This means that any
//      memory allocated by the swift runtime always has the same pointer on the
//      host and the device, so address mangling no longer needs to happen.
//
// We need to do this recursively because the closure may consist of many layers
// of indirection. XXX: Check this, it may have more structure than this,
// especially now that we don't need to do address rewriting.
//
void UpdateClosureEnvironment (
    LLVMContext& Context,
    Module& M,
    Value* K,
    Value* P,
    CUDAContext CUDA,
    CachingHostAllocator Allocator,
    Value* Event,
    SmallPtrSetImpl<Instruction*>& ToErase,
    SmallPtrSetImpl<Value*>& ToFree
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
    if (auto Prev = NewI->getPrevNonDebugInstruction()) {
      if (auto Save = dyn_cast<CallInst>(Prev)) {
        if (Save->getCalledFunction()->getName() == "llvm.stacksave") {
          // We can have multiple terminating blocks of the function when throwing
          // functions are involved
          for (auto U : Save->users()) {
            if (auto Restore = dyn_cast<CallInst>(U)) {
              assert(Restore->getCalledFunction()->getName() == "llvm.stackrestore");

              Function *F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
              CallInst *Free = CallInst::Create(F->getFunctionType(), F, {NewI, Event, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
              Free->setCallingConv(CallingConv::Swift);
              Free->insertAfter(Restore);
              ToErase.insert(Restore);
            }
          }
          ToErase.insert(Save);
          ToFree.erase(NewI);
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
        ToErase.insert(I);
      }
      else if (Name.starts_with("llvm.lifetime.end")) {
        // Assume that we will encounter the corresponding .start()
        Value *Alloca = I->getArgOperand(1);
        Function *F = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
        CallInst *Free = CallInst::Create(F->getFunctionType(), F, {Alloca, Event, get<0>(Allocator), get<1>(Allocator), get<2>(Allocator)});
        Free->setCallingConv(CallingConv::Swift);
        Free->insertAfter(I);
        ToFree.erase(Alloca);
        ToErase.insert(I);
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
    Value* Event,
    ValueToValueMapTy& CopyOnWriteMap
)
{
  SmallPtrSet<Instruction*, 8> ToErase;
  SmallPtrSet<Value*, 8> ToFree;
  Function *Parent = cast<Instruction>(Env)->getFunction();

  // Lift any copy-on-write handlers out of the kernel body
  UpdateClosureEnvironment(Context, M, Env, CopyOnWriteMap);

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
  auto [CUDA0, CUDA1, CUDA2] = CUDA;
  auto [Allocator0, Allocator1, Allocator2] = Allocator;

  if (isa<Instruction>(CUDA0) || isa<Instruction>(Allocator0)) {
    Instruction* Highest = nullptr;

    // Locate the pre-dominating instruction
    for (auto U : Allocator0->users()) {
      if (auto I = dyn_cast<Instruction>(U)) {
        if (!Highest || !DT.dominates(Highest, I)) {
          Highest = I;
        }
      }
    }

    // Now move the context...
    if (auto I = dyn_cast<Instruction>(CUDA0)) {
      if (auto E = dyn_cast<ExtractValueInst>(I)) {
        if (auto I = dyn_cast<Instruction>(E->getAggregateOperand())) {
          I->moveBefore(Highest);
        }
      }
      I->moveBefore(Highest);
      cast<Instruction>(CUDA1)->moveBefore(Highest);
      cast<Instruction>(CUDA2)->moveBefore(Highest);
    }

    // ...and/or allocator initialiser ahead of the dominating instruction
    if (auto I = dyn_cast<Instruction>(Allocator0)) {
      if (auto E = dyn_cast<ExtractValueInst>(I)) {
        if (auto I = dyn_cast<Instruction>(E->getAggregateOperand())) {
          I->moveBefore(Highest);
        }
      }
      I->moveBefore(Highest);
      cast<Instruction>(Allocator1)->moveBefore(Highest);
      cast<Instruction>(Allocator2)->moveBefore(Highest);
    }

    // Now, we need to determine when the allocator can be released. For
    // throwing functions there may not be a single dominating terminator block,
    // so we need to do it along every pathway
    for (auto U : Allocator0->users()) {
      if (auto I = dyn_cast<CallInst>(U)) {
        if (I->getCalledFunction()->getName() == "swift_release") {
          // Now locate the lowest user where this instruction dominates it
          Instruction* Lowest = nullptr;
          Instruction* Release0 = I;
          Instruction* Release1 = Release0->getPrevNonDebugInstruction();
          Instruction* Release2 = Release1->getPrevNonDebugInstruction();
          assert(cast<CallInst>(Release1)->getCalledFunction()->getName() == "swift_release");
          assert(cast<CallInst>(Release2)->getCalledFunction()->getName() == "swift_release");

          for (auto U2 : Allocator0->users()) {
            if (auto I2 = dyn_cast<Instruction>(U2)) {
              if (I == I2)
                continue;

              if (DT.dominates(I, I2)) {
                if (!Lowest || DT.dominates(Lowest, I2)) {
                  Lowest = I2;
                }
              }
            }
          }

          if (Lowest) {
            Release0->moveAfter(Lowest);
            Release1->moveAfter(Lowest);
            Release2->moveAfter(Lowest);
          }
        }
      }
    }
  }

  // Also ensure we don't swift_release the ready event too early
  Instruction* Lowest = nullptr;
  Instruction* Release = nullptr;
  for (auto U : Event->users()) {
    if (auto I = dyn_cast<Instruction>(U)) {
      if (auto CI = dyn_cast<CallInst>(I)) {
        if (CI->getCalledFunction()->getName() == "swift_release") {
          Release = I;
          continue;
        }
      }

      if (!Lowest || DT.dominates(Lowest, I)) {
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
  Function* Fseq = M.getFunction("$s10SwiftToPTX12parallel_for10iterations7context9allocator6stream_AA5EventCSi_AA7ContextVAA20CachingHostAllocatorVAA6StreamVySixYKXEtxYKs5ErrorRzlF");
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

  Function* Fpar = M.getFunction("$s10SwiftToPTX19launch_parallel_for10iterations7context6stream6kernel3env10swifterror11thrownerrorAA5EventCSi_AA7ContextVAA6StreamVAA17ParallelForKernelVzS3vtF");
  StructType* kernel_t = StructType::getTypeByName(Context, "T10SwiftToPTX17ParallelForKernelV");

  // We may encounter functions that need to be inlined into their callsite so
  // that the kernels can be specialised. In future we may want to revisit this,
  // and instead have the calling functions pass down a reference to the
  // compiled kernel module that should be used (in place of pointer to the
  // function to put into the environment that will be indirectly called, for
  // example).
  SmallVector<Value*, 8> WorkQueue(Fseq->users());

  while (!WorkQueue.empty()) {
    auto CI = dyn_cast<CallBase>(WorkQueue.back());
    WorkQueue.pop_back();
    if (!CI)
      continue;

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
    /* Value* TypeMetadata = CI->getArgOperand(10); */
    /* Value* ProtocolWitnessTable = CI->getArgOperand(11); */
    /* Value* SwiftSelf = CI->getArgOperand(12); */
    /* Value* SwiftError = CI->getArgOperand(13); */
    Value* ThrownError = CI->getArgOperand(14);

    // Do we need to specialise this instance into its call site?
    if (auto I = dyn_cast<Argument>(Body)) {
      for (auto U : CI->getFunction()->users()) {
        auto CB = dyn_cast<CallBase>(U);
        if (!CB)
          continue;

        InlineFunctionInfo IFI;
        auto R = InlineFunction(*CB, IFI);
        if (!R.isSuccess())
          report_fatal_error(R.getFailureReason());

        for (auto ICS : IFI.InlinedCallSites) {
          if (Fseq == ICS->getCalledFunction()) {
            WorkQueue.push_back(ICS);
          }
        }
      }
      continue;
    }

    // Extract the set of global functions that this kernel consists of. Also
    // keep track of indirect functions that need to be specialised at the
    // call site, as well as copy-on-write handlers that will be lifted out of
    // the kernel into the closure environment setup phase.
    SmallPtrSet<GlobalValue*, 16> GVs;
    SmallPtrSet<GlobalValue*, 16> DeclOnlyGVs;
    ValueToValueMapTy IndirectMap;
    ValueToValueMapTy CopyOnWriteMap;
    ExtractKernel(Context, CI, Body, Env, GVs, DeclOnlyGVs, IndirectMap, CopyOnWriteMap);

    // Generate PTX assembly for the (set of) functions called by the
    // `parallel_for` launcher, and embed the generated code into the module
    ArrayRef<uint8_t> Obj = CreateKernel(Context, M, Body->getName(), GVs, DeclOnlyGVs, IndirectMap);
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

    // Create temporary placeholders for the swifterror and throwerror terms
    // that can be passed to the GPU kernel. These will need to be
    // synchronised with the "real" swift error terms once the kernel
    // completes, but most likely we'll have to wait until we have this marked
    // as an 'async throws' function until that will work correctly.
    IntegerType *i64_t = IntegerType::getInt64Ty(Context);
    PointerType *ptr_t = PointerType::get(Context, 0);
    ConstantInt *size = ConstantInt::get(i64_t, M.getDataLayout().getTypeAllocSize(ptr_t).getFixedValue());
    Function *Alloc = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV5allocySvSiF");
    CallInst *KernelError = CallInst::Create(Alloc->getFunctionType(), Alloc, {size, Allocator0, Allocator1, Allocator2});
    KernelError->setCallingConv(CallingConv::Swift);
    KernelError->insertBefore(CI);
    new StoreInst(ConstantPointerNull::get(ptr_t), KernelError, CI);

    // Update the calling instruction to our placeholder `parallel_for` to our
    // kernel launcher. This assumes that the environment is set up correctly,
    // which we will do in the next step.
    std::vector<Value*> params = {Iterations, Context0, Context1, Context2, Stream, Kernel, Env, KernelError, ThrownError};
    CallInst* CIpar = CallInst::Create(Fpar->getFunctionType(), Fpar, params);
    CIpar->setCallingConv(CallingConv::Swift);
    CIpar->addParamAttr(5, Attribute::NonNull);
    CIpar->addParamAttr(5, Attribute::NoCapture);
    CIpar->addParamAttr(5, Attribute::getWithDereferenceableBytes(Context, 32));
    CIpar->addParamAttr(7, Attribute::NoAlias);
    CIpar->addParamAttr(7, Attribute::NoCapture);
    /* CIpar->addParamAttr(7, Attribute::SwiftError); */
    CIpar->addParamAttr(7, Attribute::getWithDereferenceableBytes(Context, 8));
    ReplaceInstWithInst(CI, CIpar);

    // Free the temporary swifterror term passed to the kernel. See above TODO
    Function *Free = M.getFunction("$s10SwiftToPTX20CachingHostAllocatorV4freeyySv_AA5EventCtF");
    CallInst *FreeI = CallInst::Create(Free->getFunctionType(), Free, {KernelError, CIpar, Allocator0, Allocator1, Allocator2});
    FreeI->setCallingConv(CallingConv::Swift);
    FreeI->insertAfter(CIpar);

    // Update the closure environment so that its contents are accessible from the device
    UpdateClosureEnvironment(Context, M, Body, Env, { Context0, Context1, Context2 }, { Allocator0, Allocator1, Allocator2 }, CIpar, CopyOnWriteMap);
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

