---
# Build and Test


Prerequisites
=============

Release of supports Linux\*-based systems with the following packages
and prerequisites:

Other configurations may work, but should be considered experimental
with limited support. On Ubuntu 16.04 with gcc-5.4.0 or clang-3.9, for
example, we recommend adding `-DNGRAPH_USE_PREBUILT_LLVM=TRUE` to the
cmake command in step 4 below. This fetches a pre-built tarball of
LLVM+Clang from llvm.org, and it will substantially reduce build time.

If using `gcc` version 4.8, it may be necessary to add symlinks from
`gcc` to `gcc-4.8`, and from `g++` to `g++-4.8`, in your PATH, even if
you explicitly specify the `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`
flags when building. (**Do NOT** supply the `-DNGRAPH_USE_PREBUILT_LLVM`
flag in this case, because the prebuilt tarball supplied on llvm.org is
not compatible with a gcc 4.8-based build.)

The `default` build
-------------------

Running `cmake` with no build flags defaults to the following settings;
adjust as needed:

```
-- NGRAPH_UNIT_TEST_ENABLE:         ON
-- NGRAPH_TOOLS_ENABLE:             ON
-- NGRAPH_CPU_ENABLE:               ON
-- NGRAPH_INTELGPU_ENABLE:          OFF
-- NGRAPH_GPU_ENABLE:               OFF
-- NGRAPH_INTERPRETER_ENABLE:       ON
-- NGRAPH_NOP_ENABLE:               ON
-- NGRAPH_GPUH_ENABLE:              OFF
-- NGRAPH_GENERIC_CPU_ENABLE:       OFF
-- NGRAPH_DEBUG_ENABLE:             OFF
-- NGRAPH_ONNX_IMPORT_ENABLE:       OFF
-- NGRAPH_DEX_ONLY:                 OFF
-- NGRAPH_CODE_COVERAGE_ENABLE:     OFF
-- NGRAPH_LIB_VERSIONING_ENABLE:    OFF
-- NGRAPH_PYTHON_BUILD_ENABLE:      OFF
-- NGRAPH_USE_PREBUILT_LLVM:        OFF
-- NGRAPH_PLAIDML_ENABLE:           OFF
-- NGRAPH_DISTRIBUTED_ENABLE:       OFF
```


The default cmake procedure (no build flags) will install `ngraph_dist`
to an OS-level location like `/usr/bin/ngraph_dist` or
`/usr/lib/ngraph_dist`. Here we specify how to build locally to the
location of `~/ngraph_dist` with the cmake target
`-DCMAKE_INSTALL_PREFIX=~/ngraph_dist`.


All of the nGraph Library documentation presumes that `ngraph_dist` gets
installed locally. The system location can be used just as easily by
customizing paths on that system. See the ngraph/CMakeLists.txt file to
change or customize the default CMake procedure.

Install steps
-------------

### Ubuntu 16.04

The process documented here will work on Ubuntu\* 16.04 (LTS) or on
Ubuntu 18.04 (LTS).

1.  (Optional) Create something like `/opt/libraries` and (with sudo),
    give ownership of that directory to your user. Creating such a
    placeholder can be useful if you'd like to have a local reference
    for APIs and documentation, or if you are a developer who wants to
    experiment with how to core/constructing-graphs/execute using
    resources available through the code base.

    ```
    $ sudo mkdir -p /opt/libraries
    $ sudo chown -R username:username /opt/libraries
    $ cd /opt/libraries
    ```

2.  Clone the NervanaSystems `ngraph` repo:

    ``` {.sourceCode .console}
    $ git clone https://github.com/NervanaSystems/ngraph.git
    $ cd ngraph
    ```

3.  Create a build directory outside of the `ngraph/src` directory tree;
    somewhere like `ngraph/build`, for example:

    ```
    $ mkdir build && cd build
    ```

4.  Generate the GNU Makefiles in the customary manner (from within the
    `build` directory). This command enables ONNX support in the library
    and sets the target build location at `~/ngraph_dist`, where it can
    be found easily.

    ```
    $ cmake .. -DNGRAPH_ONNX_IMPORT_ENABLE=ON  -DCMAKE_INSTALL_PREFIX=~/ngraph_dist
    ```

    **Other optional build flags** -- If running `gcc-5.4.0` or
    `clang-3.9`, remember that you can also append `cmake` with the
    prebuilt LLVM option to speed-up the build. Another option if your
    deployment system has Intel® Advanced Vector Extensions (Intel® AVX)
    is to target the accelerations available directly by compiling the
    build as follows during the cmake step:
    `-DNGRAPH_TARGET_ARCH=skylake-avx512`.

    ```
    $ cmake .. [-DNGRAPH_USE_PREBUILT_LLVM=OFF] [-DNGRAPH_TARGET_ARCH=skylake-avx512]   
    ```

5.  Run `$ make` and `make install` to install `libngraph.so` and the
    header files to `~/ngraph_dist`:

    ```
    $ make   # note: make -j <N> may work, but sometimes results in out-of-memory errors if too many compilation processes are used
    $ make install          
    ```

6.  (Optional, requires
    [doxygen](https://www.stack.nl/~dimitri/doxygen/),
    [Sphinx](http://www.sphinx-doc.org/en/stable/), and
    [breathe](https://breathe.readthedocs.io/en/latest/)). Run
    `make html` inside the `doc/sphinx` directory of the cloned source
    to build a copy of the [website
    docs](http://ngraph.nervanasys.com/docs/latest/) locally. The
    low-level API docs with inheritance and collaboration diagrams can
    be found inside the `/docs/doxygen/` directory. See the
    project/doc-contributor-README for more details about how to build
    documentation for nGraph.

### CentOS 7.4

The process documented here will work on CentOS 7.4.

1.  (Optional) Create something like `/opt/libraries` and (with sudo),
    give ownership of that directory to your user. Creating such a
    placeholder can be useful if you'd like to have a local reference
    for APIs and documentation, or if you are a developer who wants to
    experiment with how to core/constructing-graphs/execute using
    resources available through the code base.

    ```
    $ sudo mkdir -p /opt/libraries
    $ sudo chown -R username:username /opt/libraries
    ```

2.  Update the system with yum and issue the following commands:

    ```
    $ sudo yum update
    $ sudo yum install zlib-devel install ncurses-libs ncurses-devel patch diffutils wget gcc-c++ make git perl-Data-Dumper
    ```

3.  Install Cmake 3.4:

    ```
    $ wget https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz      
    $ tar -xzvf cmake-3.4.3.tar.gz
    $ cd cmake-3.4.3
    $ ./bootstrap --system-curl --prefix=~/cmake
    $ make && make install     
    ```

4.  Clone the NervanaSystems `ngraph` repo via HTTPS and use Cmake 3.4.3
    to build nGraph Libraries to `~/ngraph_dist`. This command enables
    ONNX support in the library (optional).

    ```
    $ cd /opt/libraries 
    $ git clone https://github.com/NervanaSystems/ngraph.git
    $ cd ngraph && mkdir build && cd build
    $ ~/cmake/bin/cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_dist -DNGRAPH_ONNX_IMPORT_ENABLE=ON 
    $ make && sudo make install 
    ```

macOS\* development
-------------------

Although we do not currently offer full support for the macOS platform,
some configurations and features may work.

The repository includes two scripts (`maint/check-code-format.sh` and
`maint/apply-code-format.sh`) that are used respectively to check
adherence to `libngraph` code formatting conventions, and to
automatically reformat code according to those conventions. These
scripts require the command `clang-format-3.9` to be in your `PATH`. Run
the following commands (you will need to adjust them if you are not
using bash):

```
$ brew install llvm@3.9 automake
$ mkdir -p $HOME/bin
$ ln -s /usr/local/opt/llvm@3.9/bin/clang-format $HOME/bin/clang-format-3.9
$ echo 'export PATH=$HOME/bin:$PATH' >> $HOME/.bash_profile
```

Testing the build
=================

We use the [googletest
framework](https://github.com/google/googletest.git) from Google for
unit tests. The `cmake` command automatically downloaded a copy of the
needed `gtest` files when it configured the build directory.

To perform unit tests on the install:

1.  Create and configure the build directory as described in our buildlb
    guide.
2.  Enter the build directory and run `make check`:

    $ cd build/
    $ make check
