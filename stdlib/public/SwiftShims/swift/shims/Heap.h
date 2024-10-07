//===--- Heap.h -------------------------------------------------*- C++ -*-===//
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
#ifndef SWIFT_STDLIB_SHIMS_HEAP_H
#define SWIFT_STDLIB_SHIMS_HEAP_H

#include "SwiftStddef.h"
#include "Visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

SWIFT_ATTRIBUTE_FOR_IMPORTS
__swift_size_t swift_usableSize(const void* ptr);


#if defined(__APPLE__)
extern __swift_size_t malloc_size(const void *);
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__ANDROID__) \
   || defined(__HAIKU__) || defined(__FreeBSD__) || defined(__wasi__)
#if defined(__ANDROID__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 17)
extern __swift_size_t malloc_usable_size(const void * _Nullable ptr);
#else
extern __swift_size_t malloc_usable_size(void *ptr);
#endif
#elif defined(_WIN32)
extern __swift_size_t _msize(void *ptr);
#endif

#ifdef __cplusplus
}
#endif

#endif // SWIFT_STDLIB_SHIMS_HEAP_H

