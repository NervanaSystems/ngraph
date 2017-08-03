# Overview
TODO

# Building `libngraph`

## Build Environments

| Operating System            | Compiler  | Status    | Additional packages required      |
| --------------------------- | --------  | --------- | --------------------------------- |
| Ubuntu 14.04.5 (LTS) 64-bit | CLang 3.9 | supported | `build-essential cmake clang-3.9` |

## Steps

`libngraph` is build in the customary manner for a CMake-based project:

1. Create a build directory outside of source directory tree.
2. `cd` to the build directory.
3. Run CMake.  For example, `cmake ../private-ngraph-cpp -DCMAKE_CXX_COMPILER=clang++-3.9`
4. Run `make`.

# Testing `libngraph`

`libngraph` uses the GTest framework for unit tests.   CMake automatically downloads a
copy of the required GTest files when configuring the build directory.

To perform the unit tests

1. Configure the build directory as described above.
2. Change directory to the build directory.
3. Run `make check`

# Using `libngraph`

## System Requirements
TBD

## External library requirements
TBD
