# Overview
TODO

# Building `libngraph`

## Build Environments

| Operating System            | Compiler  | Build system           | Status                 | Additional packages required      |
| --------------------------- | --------- | ---------------------- | ---------------------- | --------------------------------- |
| Ubuntu 16.04 (LTS) 64-bit   | CLang 3.9 | CMake 3.5.1 + GNU Make | supported              | `build-essential cmake clang-3.9` |
| Ubuntu 16.04 (LTS) 64-bit   | CLang 4.0 | CMake 3.5.1 + GNU Make | unsupported, but works | `build-essential cmake clang-4.0` |

## Steps

_If you are developing ngraph on macOS (officially unsupported) please see the section "macOS Development Prerequisites" below._

`libngraph` is build in the customary manner for a CMake-based project:

1. Create a build directory outside of source directory tree.
2. `cd` to the build directory.
3. Run `cmake`.  For example, `cmake ../`
4. Run `make -j8`.
5. Run `make install`.
    * This will install `libngraph.so` and the header files to `$HOME/ngraph_dist`.
6. _(Optional, requires `doxygen`)_ Run `make doc`.
    * This will build API documentation in the directory `doc` inside the build directory.

## macOS Development Prerequisites

The repository includes two scripts (`maint/check-code-format.sh` and `maint/apply-code-format.sh`) that are used respectively to check adherence to `libngraph` code formatting conventions, and to automatically reformat code according to those conventions. These scripts require the command `clang-format-3.9` to be in your `PATH`. Run the following commands (you will need to adjust them if you are not using `bash`).

```
$ brew install llvm@3.9
$ mkdir -p $HOME/bin
$ ln -s /usr/local/opt/llvm@3.9/bin/clang-format $HOME/bin/clang-format-3.9
$ echo 'export PATH=$HOME/bin:$PATH' >> $HOME/.bash_profile
```

# Testing `libngraph`

`libngraph` uses the GTest framework for unit tests.   CMake automatically downloads a
copy of the required GTest files when configuring the build directory.

To perform the unit tests

1. Configure the build directory as described above.
2. Change directory to the build directory.
3. Run `make check`

# Using `libngraph`

## From Tensorflow as XLA plugin

:warning: Note: Work in Progress.

1. Get the Nervana's fork of the TF from this repo: ```git@github.com:NervanaSystems/ngraph-tensorflow.git```
2. Go to the end near the following snippet:
```
  native.new_local_repository(
    name = "ngraph_external",
    path = "/your/home/directory/where/ngraph_is_installed",
    build_file = str(Label("//tensorflow/compiler/plugin/ngraph:ngraph.BUILD")),
  )
```

Then modify the following line in `tensorflow/workspace.bzl` file and provide absolute path to `~/ngraph_dist` :
```
path = "/your/home/directory/where/ngraph_is_installed",
``` 
3. Now run `configure` and rest of the TF build.

## System Requirements
TBD

## External library requirements
TBD

# Maintaining `libngraph`

## Code formatting
All C/C++ source code in the `libngraph` repository, including the test code when practical,
should adhere to the project's source-code formatting guidelines.

The script `maint/apply-code-format.sh` enforces that formatting at the C/C++ syntactic level.

The script `maint/check-code-format.sh` verifies that the formatting rules are met by all C/C++
code (again, at the syntax level.)  The script has an exit code of 0 when this all code meets
the standard, and non-zero otherwise.  This script does _not_ modify the source code.

