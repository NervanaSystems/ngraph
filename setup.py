# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# ----------------------------------------------------------------------------

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '0.0.1'


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


if "NGRAPH_CPP_BUILD_PATH" in os.environ:
    NGRAPH_CPP_INCLUDE_DIR = os.environ["NGRAPH_CPP_BUILD_PATH"] + "/include"
    NGRAPH_CPP_LIBRARY_DIR = os.environ["NGRAPH_CPP_BUILD_PATH"] + "/lib"
else:
    print("NGRAPH_CPP_BUILD_PATH must be defined exiting")
    exit()


sources = ['pyngraph/function.cpp',
           'pyngraph/node.cpp',
           'pyngraph/pyngraph.cpp',
           'pyngraph/util.cpp',
           'pyngraph/ops/add.cpp',
           'pyngraph/ops/broadcast.cpp',
           'pyngraph/ops/constant.cpp',
           'pyngraph/ops/convert.cpp',
           'pyngraph/ops/divide.cpp',
           'pyngraph/ops/dot.cpp',
           'pyngraph/ops/exp.cpp',
           'pyngraph/ops/greater.cpp',
           'pyngraph/ops/less.cpp',
           'pyngraph/ops/log.cpp',
           'pyngraph/ops/maximum.cpp',
           'pyngraph/ops/minimum.cpp',
           'pyngraph/ops/multiply.cpp',
           'pyngraph/ops/negative.cpp',
           'pyngraph/ops/op.cpp',
           'pyngraph/ops/one_hot.cpp',
           'pyngraph/ops/parameter.cpp',
           'pyngraph/ops/reduce.cpp',
           'pyngraph/ops/regmodule_pyngraph_op.cpp',
           'pyngraph/ops/reshape.cpp',
           'pyngraph/ops/subtract.cpp',
           'pyngraph/ops/sum.cpp',
           'pyngraph/runtime/backend.cpp',
           'pyngraph/runtime/call_frame.cpp',
           'pyngraph/runtime/external_function.cpp',
           'pyngraph/runtime/manager.cpp',
           'pyngraph/runtime/ndarray.cpp',
           'pyngraph/runtime/parameterized_tensor_view.cpp',
           'pyngraph/runtime/regmodule_pyngraph_runtime.cpp',
           'pyngraph/runtime/tensor_view.cpp',
           'pyngraph/runtime/utils.cpp',
           'pyngraph/runtime/value.cpp',
           'pyngraph/types/element_type.cpp',
           'pyngraph/types/regmodule_pyngraph_types.cpp',
           'pyngraph/types/type.cpp',
           ]

include_dirs = [# Path to pybind11 headers
                "pybind11/include",
                NGRAPH_CPP_INCLUDE_DIR,
                ".",
               ]

library_dirs = [NGRAPH_CPP_LIBRARY_DIR,
               ]

libraries    = ["ngraph",
               ]

extra_compile_args = []

extra_link_args = []

ext_modules = [Extension(
                   'pyngraph',
                   sources = sources,
                   include_dirs = include_dirs,
                   define_macros = [("VERSION_INFO", __version__)],
                   library_dirs = library_dirs,
                   libraries = libraries,
                   extra_link_args = extra_link_args,
                   language = "c++",
                   )
              ]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            ext.extra_compile_args += [cpp_flag(self.compiler)]
            if has_flag(self.compiler, '-fvisibility=hidden'):
                ext.extra_compile_args += ['-fvisibility=hidden']
            if sys.platform == 'darwin':
                ext.extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
            # else:
            #    ext.extra_link_args += ["-shared"]
            ext.extra_link_args += ["-Wl,-rpath,%s"%(NGRAPH_CPP_LIBRARY_DIR)]
        build_ext.build_extensions(self)


setup(
    name='pyngraph',
    version=__version__,
    author='Me',
    author_email='me@i.com',
    url='https://github.com/NervanaSystems/ngraph-neon',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
