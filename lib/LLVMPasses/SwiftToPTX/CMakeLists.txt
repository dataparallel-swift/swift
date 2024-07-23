add_swift_host_library(swiftLLVMPassesSwiftToPTX STATIC
  ParallelFor.cpp

  LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitReader
  Core
  InstCombine
  MC
  ${LLVM_NATIVE_ARCH}CodeGen
  NVPTXCodeGen
  Passes
  ScalarOpts
  Support
  Target
  TransformUtils
)

target_link_libraries(swiftLLVMPassesSwiftToPTX PRIVATE
  swiftDemangling
)

target_include_directories(swiftLLVMPassesSwiftToPTX PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

set_swift_llvm_is_available(swiftLLVMPassesSwiftToPTX)

