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

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '0.0.1'


# Parallel build from http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool().imap(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile


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
    raise RuntimeError('NGRAPH_CPP_BUILD_PATH must be defined')


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
           'pyngraph/runtime/regmodule_pyngraph_runtime.cpp',
           'pyngraph/runtime/tensor_view.cpp',
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

data_files = [('lib', [NGRAPH_CPP_LIBRARY_DIR + "/" + library for library in os.listdir(NGRAPH_CPP_LIBRARY_DIR)]),]

ext_modules = [Extension(
                   '_pyngraph',
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
            if has_flag(self.compiler, '-frtti'):
                ext.extra_compile_args += ['-frtti']
            if sys.platform == 'darwin':
                ext.extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
                ext.extra_link_args += ["-Wl,-rpath,@loader_path/../.."]
            else:
                if has_flag(self.compiler, '-fvisibility=hidden'):
                    ext.extra_compile_args += ['-fvisibility=hidden']
                ext.extra_link_args += ["-Wl,-rpath,$ORIGIN/../.."]
        build_ext.build_extensions(self)


requirements = [
    "setuptools",
    "six",
]


setup(
    name='pyngraph',
    version=__version__,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='http://www.intelnervana.com',
    license='License :: OSI Approved :: Apache Software License',
    description='Python wrapper for ngraph',
    long_description='',
    ext_modules=ext_modules,
    packages = find_packages(exclude=['pybind11', 'build', 'test']),
    cmdclass={'build_ext': BuildExt},
    data_files = data_files,
    install_requires = requirements,
    zip_safe=False,
)
