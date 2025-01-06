//===--- Heap.cpp - Swift Language Heap Logic -----------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// Implementations of the Swift heap
//
//===----------------------------------------------------------------------===//

#include "swift/Runtime/HeapObject.h"
#include "swift/Runtime/Heap.h"
#include "Private.h"
#include "swift/Runtime/Debug.h"
#include "swift/shims/RuntimeShims.h"
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__) && SWIFT_STDLIB_HAS_DARWIN_LIBMALLOC
#include "swift/Basic/Lazy.h"
#include <malloc/malloc.h>
#endif

using namespace swift;

#if defined(__APPLE__)
/// On Apple platforms, \c malloc() is always 16-byte aligned.
static constexpr size_t MALLOC_ALIGN_MASK = 15;

#elif defined(__linux__) || defined(_WIN32) || defined(__wasi__)
/// On Linux and Windows, \c malloc() returns 16-byte aligned pointers on 64-bit
/// and 8-byte aligned pointers on 32-bit.
/// On wasi-libc, pointers are 16-byte aligned even though 32-bit for SIMD access.
#if defined(__LP64) || defined(_WIN64) || defined(__wasi__)
static constexpr size_t MALLOC_ALIGN_MASK = 15;
#else
static constexpr size_t MALLOC_ALIGN_MASK = 7;
#endif

#else
/// This platform's \c malloc() constraints are unknown, so fall back to a value
/// derived from \c std::max_align_t that will be sufficient, but is not
/// necessarily optimal.
///
/// The C and C++ standards defined \c max_align_t as a type whose alignment is
/// at least that of every scalar type. It is the lower bound for the alignment
/// of any pointer returned from \c malloc().
static constexpr size_t MALLOC_ALIGN_MASK = alignof(std::max_align_t) - 1;
#endif


/* Define ALIASNAME as a strong alias for NAME.  */
#define strong_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((alias (#name))) \
    __attribute_copy__ (name);

// Define ALIASNAME as a weak alias for NAME.
// If weak aliases are not available, this defines a strong alias.
#define weak_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((weak, alias (#name))) \
    __attribute_copy__ (name);


// This assert ensures that manually allocated memory always uses the
// AlignedAlloc path. The stdlib will use "default" alignment for any user
// requested alignment less than or equal to _swift_MinAllocationAlignment. The
// runtime must ensure that any alignment > _swift_MinAllocationAlignment also
// uses the "aligned" deallocation path.
static_assert(_swift_MinAllocationAlignment > MALLOC_ALIGN_MASK,
              "Swift's default alignment must exceed platform malloc mask.");

// When alignMask == ~(size_t(0)), allocation uses the "default"
// _swift_MinAllocationAlignment. This is different than calling swift_slowAlloc
// with `alignMask == _swift_MinAllocationAlignment - 1` because it forces
// the use of AlignedAlloc. This allows manually allocated to memory to always
// be deallocated with AlignedFree without knowledge of its original allocation
// alignment.
static size_t computeAlignment(size_t alignMask) {
  return (alignMask == ~(size_t(0))) ? _swift_MinAllocationAlignment
                                     : alignMask + 1;
}

// Forward declaration so that we can provide asm("") naming shenanigans to
// force the compiler to not generate a mangled name
SWIFT_LIBRARY_VISIBILITY
void* __swift_slowAlloc(size_t size, size_t alignMask)
  asm("__swift_slowAlloc");

SWIFT_LIBRARY_VISIBILITY
void* __swift_slowAllocTyped(size_t size, size_t alignMask, MallocTypeId typeId)
  asm("__swift_slowAllocTyped");

SWIFT_LIBRARY_VISIBILITY
void __swift_slowDealloc(void *ptr, size_t bytes, size_t alignMask)
  asm("__swift_slowDealloc");

// For alignMask > (_minAllocationAlignment-1)
// i.e. alignment == 0 || alignment > _minAllocationAlignment:
//   The runtime must use AlignedAlloc, and the standard library must
//   deallocate using an alignment that meets the same condition.
//
// For alignMask <= (_minAllocationAlignment-1)
// i.e. 0 < alignment <= _minAllocationAlignment:
//   The runtime may use either malloc or AlignedAlloc, and the standard library
//   must deallocate using an identical alignment.
void* __swift_slowAlloc(size_t size, size_t alignMask) {
  void *p;
  // This check also forces "default" alignment to use AlignedAlloc.
  if (alignMask <= MALLOC_ALIGN_MASK) {
    p = malloc(size);
  } else {
    size_t alignment = computeAlignment(alignMask);
    p = AlignedAlloc(size, alignment);
  }
  if (!p) swift::crash("Could not allocate memory.");
  return p;
}
weak_alias(__swift_slowAlloc, swift::swift_slowAlloc)


void* __swift_slowAllocTyped(size_t size, size_t alignMask, MallocTypeId typeId) {
#if SWIFT_STDLIB_HAS_MALLOC_TYPE
  if (__builtin_available(macOS 9998, iOS 9998, tvOS 9998, watchOS 9998, *)) {
    void *p;
    // This check also forces "default" alignment to use malloc_memalign().
    if (alignMask <= MALLOC_ALIGN_MASK) {
      p = malloc_type_malloc(size, typeId);
    } else {
      size_t alignment = computeAlignment(alignMask);

      // Do not use malloc_type_aligned_alloc() here, because we want this
      // to work if `size` is not an integer multiple of `alignment`, which
      // was a requirement of the latter in C11 (but not C17 and later).
      int err = malloc_type_posix_memalign(&p, alignment, size, typeId);
      if (err != 0)
        p = nullptr;
    }
    if (!p) swift::crash("Could not allocate memory.");
    return p;
  }
#endif
  return __swift_slowAlloc(size, alignMask);
}
weak_alias(__swift_slowAllocTyped, swift::swift_slowAllocTyped)


// Unknown alignment is specified by passing alignMask == ~(size_t(0)), forcing
// the AlignedFree deallocation path for unknown alignment. The memory
// deallocated with unknown alignment must have been allocated with either
// "default" alignment, or alignment > _swift_MinAllocationAlignment, to
// guarantee that it was allocated with AlignedAlloc.
//
// The standard library assumes the following behavior:
//
// For alignMask > (_minAllocationAlignment-1)
// i.e. alignment == 0 || alignment > _minAllocationAlignment:
//   The runtime must use AlignedFree.
//
// For alignMask <= (_minAllocationAlignment-1)
// i.e. 0 < alignment <= _minAllocationAlignment:
//   The runtime may use either `free` or AlignedFree as long as it is
//   consistent with allocation with the same alignment.
void __swift_slowDealloc(void *ptr, size_t bytes, size_t alignMask) {
  if (alignMask <= MALLOC_ALIGN_MASK) {
    free(ptr);
  } else {
    AlignedFree(ptr);
  }
}
weak_alias(__swift_slowDealloc, swift::swift_slowDealloc)

void swift::swift_clearSensitive(void *ptr, size_t bytes) {
  // TODO: use memset_s if available
  // Though, it shouldn't make too much difference because the optimizer cannot remove
  // the following memset without inlining this library function.
  memset(ptr, 0, bytes);
}
