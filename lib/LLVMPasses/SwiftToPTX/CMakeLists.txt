add_library(LiftToPTX MODULE
  ParallelFor.cpp
)

target_include_directories(LiftToPTX PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# allow undefined symbols in shared libraries on MacOS (this is the default
# behaviour on Linux)
target_link_libraries(LiftToPTX "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

