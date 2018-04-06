# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import distutils.ccompiler

__version__ = '0.2.0'

PYNGRAPH_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))


def find_ngraph_dist_dir():
    """Return location of compiled ngraph library home."""
    if os.environ.get('NGRAPH_CPP_BUILD_PATH'):
        ngraph_dist_dir = os.environ.get('NGRAPH_CPP_BUILD_PATH')
    else:
        ngraph_dist_dir = os.path.join(PYNGRAPH_SOURCE_DIR, 'build/ngraph_dist')

    found = os.path.exists(os.path.join(ngraph_dist_dir, 'include/ngraph'))
    if not found:
        print("Cannot find nGraph library in {} make sure that "
              "NGRAPH_CPP_BUILD_PATH is set correctly".format(ngraph_dist_dir))
        sys.exit(1)
    else:
        print("nGraph library found in {}".format(ngraph_dist_dir))
        return ngraph_dist_dir


def find_pybind_headers_dir():
    if os.environ.get('PYBIND_HEADERS_PATH'):
        pybind_headers_dir = os.environ.get('PYBIND_HEADERS_PATH')
    else:
        pybind_headers_dir = os.path.join(PYNGRAPH_SOURCE_DIR, 'build/pybind11')

    found = os.path.exists(os.path.join(pybind_headers_dir, 'include/pybind11'))
    if not found:
        print("Cannot find pybind11 library in {} make sure that "
              "PYBIND_HEADERS_PATH is set correctly".format(pybind_headers_dir))
        sys.exit(1)
    else:
        print("pybind11 library found in {}".format(pybind_headers_dir))
        return pybind_headers_dir


NGRAPH_CPP_DIST_DIR = find_ngraph_dist_dir()
PYBIND11_INCLUDE_DIR = find_pybind_headers_dir() + "/include"
NGRAPH_CPP_INCLUDE_DIR = NGRAPH_CPP_DIST_DIR + "/include"
NGRAPH_CPP_LIBRARY_DIR = NGRAPH_CPP_DIST_DIR + "/lib"


# Parallel build from http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None,
                     debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    import multiprocessing.pool

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool().imap(_single_compile, objects))
    return objects


distutils.ccompiler.CCompiler.compile=parallelCCompile


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """
    Return a boolean indicating whether a flag name is supported on
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


sources = ['pyngraph/function.cpp',
           'pyngraph/serializer.cpp',
           'pyngraph/node.cpp',
           'pyngraph/node_vector.cpp',
           'pyngraph/shape.cpp',
           'pyngraph/strides.cpp',
           'pyngraph/coordinate_diff.cpp',
           'pyngraph/axis_set.cpp',
           'pyngraph/axis_vector.cpp',
           'pyngraph/coordinate.cpp',
           'pyngraph/pyngraph.cpp',
           'pyngraph/util.cpp',
           'pyngraph/ops/util/arithmetic_reduction.cpp',
           'pyngraph/ops/util/binary_elementwise_comparison.cpp',
           'pyngraph/ops/util/requires_tensor_view_args.cpp',
           'pyngraph/ops/util/binary_elementwise.cpp',
           'pyngraph/ops/util/op_annotations.cpp',
           'pyngraph/ops/util/unary_elementwise.cpp',
           'pyngraph/ops/util/binary_elementwise_arithmetic.cpp',
           'pyngraph/ops/util/regmodule_pyngraph_op_util.cpp',
           'pyngraph/ops/util/unary_elementwise_arithmetic.cpp',
           'pyngraph/ops/abs.cpp',
           'pyngraph/ops/acos.cpp',
           'pyngraph/ops/add.cpp',
           'pyngraph/ops/asin.cpp',
           'pyngraph/ops/atan.cpp',
           'pyngraph/ops/avg_pool.cpp',
           'pyngraph/ops/broadcast.cpp',
           'pyngraph/ops/concat.cpp',
           'pyngraph/ops/constant.cpp',
           'pyngraph/ops/convert.cpp',
           'pyngraph/ops/convolution.cpp',
           'pyngraph/ops/cos.cpp',
           'pyngraph/ops/cosh.cpp',
           'pyngraph/ops/ceiling.cpp',
           'pyngraph/ops/divide.cpp',
           'pyngraph/ops/dot.cpp',
           'pyngraph/ops/equal.cpp',
           'pyngraph/ops/exp.cpp',
           'pyngraph/ops/floor.cpp',
           'pyngraph/ops/greater.cpp',
           'pyngraph/ops/greater_eq.cpp',
           'pyngraph/ops/less.cpp',
           'pyngraph/ops/less_eq.cpp',
           'pyngraph/ops/log.cpp',
           'pyngraph/ops/maximum.cpp',
           'pyngraph/ops/max.cpp',
           'pyngraph/ops/product.cpp',
           'pyngraph/ops/max_pool.cpp',
           'pyngraph/ops/minimum.cpp',
           'pyngraph/ops/multiply.cpp',
           'pyngraph/ops/negative.cpp',
           'pyngraph/ops/not.cpp',
           'pyngraph/ops/not_equal.cpp',
           'pyngraph/ops/op.cpp',
           'pyngraph/ops/one_hot.cpp',
           'pyngraph/ops/pad.cpp',
           'pyngraph/ops/parameter.cpp',
           'pyngraph/ops/parameter_vector.cpp',
           'pyngraph/ops/power.cpp',
           'pyngraph/ops/reduce.cpp',
           'pyngraph/ops/regmodule_pyngraph_op.cpp',
           'pyngraph/ops/relu.cpp',
           'pyngraph/ops/replace_slice.cpp',
           'pyngraph/ops/reshape.cpp',
           'pyngraph/ops/reverse.cpp',
           'pyngraph/ops/select.cpp',
           'pyngraph/ops/sign.cpp',
           'pyngraph/ops/sin.cpp',
           'pyngraph/ops/sinh.cpp',
           'pyngraph/ops/slice.cpp',
           'pyngraph/ops/sqrt.cpp',
           'pyngraph/ops/subtract.cpp',
           'pyngraph/ops/sum.cpp',
           'pyngraph/ops/tan.cpp',
           'pyngraph/ops/tanh.cpp',
           'pyngraph/ops/allreduce.cpp',
           'pyngraph/ops/function_call.cpp',
           'pyngraph/ops/get_output_element.cpp',
           'pyngraph/ops/min.cpp',
           'pyngraph/ops/batch_norm.cpp',
           'pyngraph/ops/softmax.cpp',
           'pyngraph/runtime/backend.cpp',
           'pyngraph/runtime/call_frame.cpp',
           'pyngraph/runtime/external_function.cpp',
           'pyngraph/runtime/manager.cpp',
           'pyngraph/runtime/regmodule_pyngraph_runtime.cpp',
           'pyngraph/runtime/tensor_view.cpp',
           'pyngraph/passes/manager.cpp',
           'pyngraph/passes/regmodule_pyngraph_passes.cpp',
           'pyngraph/types/element_type.cpp',
           'pyngraph/types/regmodule_pyngraph_types.cpp',
           'pyngraph/types/type.cpp',
           ]

sources = [PYNGRAPH_SOURCE_DIR + "/" + source for source in sources]

include_dirs = [PYNGRAPH_SOURCE_DIR,
                NGRAPH_CPP_INCLUDE_DIR,
                PYBIND11_INCLUDE_DIR,
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
    """
    A custom build extension for adding compiler-specific options.
    """
    def build_extensions(self):
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            ext.extra_compile_args += [cpp_flag(self.compiler)]
            if has_flag(self.compiler, '-fstack-protector-strong'):
                ext.extra_compile_args += ['-fstack-protector-strong']
            else:
                ext.extra_compile_args += ['-fstack-protector']
            if has_flag(self.compiler, '-frtti'):
                ext.extra_compile_args += ['-frtti']
            if sys.platform == 'darwin':
                ext.extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
                ext.extra_link_args += ["-Wl,-rpath,@loader_path/../.."]
            else:
                if has_flag(self.compiler, '-fvisibility=hidden'):
                    ext.extra_compile_args += ['-fvisibility=hidden']
                ext.extra_link_args += ['-Wl,-rpath,$ORIGIN/../..']
            if sys.platform != 'darwin':
                ext.extra_link_args += ['-z', 'noexecstack']
                ext.extra_link_args += ['-z', 'relro']
                ext.extra_link_args += ['-z', 'now']
            ext.extra_compile_args += ['-Wformat', '-Wformat-security', '-Wno-comment']
            ext.extra_compile_args += ['-O2', '-D_FORTIFY_SOURCE=2']
        build_ext.build_extensions(self)


with open(os.path.join(PYNGRAPH_SOURCE_DIR, 'requirements.txt')) as req:
    requirements = req.read().splitlines()


setup(
    name='ngraph',
    version=__version__,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='http://www.intelnervana.com',
    license='License :: OSI Approved :: Apache Software License',
    description='Python wrapper for ngraph',
    long_description='',
    ext_modules=ext_modules,
    package_dir={'ngraph': PYNGRAPH_SOURCE_DIR + "/ngraph",
                 'ngraph.utils': PYNGRAPH_SOURCE_DIR + "/ngraph/utils",
                 'ngraph.impl': PYNGRAPH_SOURCE_DIR + "/ngraph/impl",
                 'ngraph.impl.op': PYNGRAPH_SOURCE_DIR + "/ngraph/impl/op",
                 'ngraph.impl.op.util': PYNGRAPH_SOURCE_DIR + "/ngraph/impl/op/util",
                 'ngraph.impl.passes': PYNGRAPH_SOURCE_DIR + "/ngraph/impl/passes",
                 'ngraph.impl.runtime': PYNGRAPH_SOURCE_DIR + "/ngraph/impl/runtime"},
    packages = ['ngraph', 'ngraph.utils', 'ngraph.impl', 'ngraph.impl.op',
                'ngraph.impl.op.util', 'ngraph.impl.passes', 'ngraph.impl.runtime'],
    cmdclass={'build_ext': BuildExt},
    data_files = data_files,
    install_requires = requirements,
    zip_safe=False,
)
