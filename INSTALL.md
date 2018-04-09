Currently two platforms are known to work:

- Ubuntu 16.04
- CentOS 7.4

Ubuntu 16.04 Prerequisites
==========================

Compilers currently known to work are gcc-5.4.0, clang-3.9, and gcc-4.8.5.

If you are using gcc-5.4.0 or clang-3.9, it is recommended to add the
option `-DNGRAPH_USE_PREBUILT_LLVM=TRUE` to the `cmake` command. This causes
the build system to fetch a pre-built tarball of LLVM+Clang from `llvm.org`,
which substantially cuts down on build times.

If you are using gcc-4.8, it may be necessary to add symlinksfrom `gcc` to
`gcc-4.8`, and from `g++` to `g++-4.8`, in your PATH, even if you have
specify CMAKE_C_COMPILER and CMAKE_CXX_COMPILER when building. (You should
NOT supply the `-DNGRAPH_USE_PREBUILT_LLVM` flag in this case, because the
prebuilt tarball supplied on llvm.org is not compatible with a gcc-4.8
based build.)

CentOS 7.4 Prerequisites
========================

CentOS supplies an older version of CMake that is not compatible with
LLVM-5.0.1, which we build as an external dependency. There are two options:

1. (requires root privileges) install the the `cmake3` package from EPEL, or
2. (does not require root privileges) build CMake (3.1 or newer) from source,
   and run it from its build directory.

General Instructions
====================

These instructions assume that your system has been prepared in accordance
with the above prerequisites.

```
$ cd ngraph
$ mkdir build
$ cd build
$ cmake .. \
    -DCMAKE_C_COMPILER=<path to C compiler> \
    -DCMAKE_CXX_COMPILER=<path to C++ compiler>
$ make -j install
```
