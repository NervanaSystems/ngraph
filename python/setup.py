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

__version__ = os.environ.get('NGRAPH_VERSION', '0.0.0-dev')
PYNGRAPH_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
NGRAPH_DEFAULT_INSTALL_DIR = os.environ.get('HOME')
NGRAPH_ONNX_IMPORT_ENABLE = os.environ.get('NGRAPH_ONNX_IMPORT_ENABLE')


def find_ngraph_dist_dir():
    """Return location of compiled ngraph library home."""
    if os.environ.get('NGRAPH_CPP_BUILD_PATH'):
        ngraph_dist_dir = os.environ.get('NGRAPH_CPP_BUILD_PATH')
    else:
        ngraph_dist_dir = os.path.join(NGRAPH_DEFAULT_INSTALL_DIR, 'ngraph_dist')

    found = os.path.exists(os.path.join(ngraph_dist_dir, 'include/ngraph'))
    if not found:
        print('Cannot find nGraph library in {} make sure that '
              'NGRAPH_CPP_BUILD_PATH is set correctly'.format(ngraph_dist_dir))
        sys.exit(1)
    else:
        print('nGraph library found in {}'.format(ngraph_dist_dir))
        return ngraph_dist_dir


def find_pybind_headers_dir():
    """Return location of pybind11 headers."""
    if os.environ.get('PYBIND_HEADERS_PATH'):
        pybind_headers_dir = os.environ.get('PYBIND_HEADERS_PATH')
    else:
        pybind_headers_dir = os.path.join(PYNGRAPH_ROOT_DIR, 'pybind11')

    found = os.path.exists(os.path.join(pybind_headers_dir, 'include/pybind11'))
    if not found:
        print('Cannot find pybind11 library in {} make sure that '
              'PYBIND_HEADERS_PATH is set correctly'.format(pybind_headers_dir))
        sys.exit(1)
    else:
        print('pybind11 library found in {}'.format(pybind_headers_dir))
        return pybind_headers_dir


NGRAPH_CPP_DIST_DIR = find_ngraph_dist_dir()
PYBIND11_INCLUDE_DIR = find_pybind_headers_dir() + '/include'
NGRAPH_CPP_INCLUDE_DIR = NGRAPH_CPP_DIST_DIR + '/include'
if os.path.exists(NGRAPH_CPP_DIST_DIR + '/lib'):
    NGRAPH_CPP_LIBRARY_DIR = NGRAPH_CPP_DIST_DIR + '/lib'
elif os.path.exists(NGRAPH_CPP_DIST_DIR + '/lib64'):
    NGRAPH_CPP_LIBRARY_DIR = NGRAPH_CPP_DIST_DIR + '/lib64'
else:
    print('Cannot find library directory in {}, make sure that nGraph is installed '
          'correctly'.format(NGRAPH_CPP_DIST_DIR))
    sys.exit(1)


def parallelCCompile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    """Build sources in parallel.

    Reference link:
    http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
    Monkey-patch for parallel compilation.
    """
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


distutils.ccompiler.CCompiler.compile = parallelCCompile


def has_flag(compiler, flagname):
    """Check whether a flag is supported by the specified compiler.

    As of Python 3.6, CCompiler has a `has_flag` method.
    cf http://bugs.python.org/issue26689
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
    """Check and return the -std=c++11 compiler flag."""
    if has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- C++11 support is needed!')


sources = [
    'pyngraph/function.cpp',
    'pyngraph/serializer.cpp',
    'pyngraph/node.cpp',
    'pyngraph/node_vector.cpp',
    'pyngraph/shape.cpp',
    'pyngraph/strides.cpp',
    'pyngraph/coordinate_diff.cpp',
    'pyngraph/axis_set.cpp',
    'pyngraph/axis_vector.cpp',
    'pyngraph/coordinate.cpp',
    'pyngraph/parameter_vector.cpp',
    'pyngraph/pyngraph.cpp',
    'pyngraph/util.cpp',
    'pyngraph/result_vector.cpp',
    'pyngraph/ops/util/arithmetic_reduction.cpp',
    'pyngraph/ops/util/binary_elementwise_comparison.cpp',
    'pyngraph/ops/util/op_annotations.cpp',
    'pyngraph/ops/util/binary_elementwise_arithmetic.cpp',
    'pyngraph/ops/util/binary_elementwise_logical.cpp',
    'pyngraph/ops/util/regmodule_pyngraph_op_util.cpp',
    'pyngraph/ops/util/unary_elementwise_arithmetic.cpp',
    'pyngraph/ops/util/index_reduction.cpp',
    'pyngraph/ops/abs.cpp',
    'pyngraph/ops/acos.cpp',
    'pyngraph/ops/add.cpp',
    'pyngraph/ops/and.cpp',
    'pyngraph/ops/argmax.cpp',
    'pyngraph/ops/argmin.cpp',
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
    'pyngraph/ops/lrn.cpp',
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
    'pyngraph/ops/or.cpp',
    'pyngraph/ops/pad.cpp',
    'pyngraph/ops/parameter.cpp',
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
    'pyngraph/ops/topk.cpp',
    'pyngraph/ops/allreduce.cpp',
    'pyngraph/ops/function_call.cpp',
    'pyngraph/ops/get_output_element.cpp',
    'pyngraph/ops/min.cpp',
    'pyngraph/ops/batch_norm.cpp',
    'pyngraph/ops/softmax.cpp',
    'pyngraph/ops/result.cpp',
    'pyngraph/runtime/backend.cpp',
    'pyngraph/runtime/regmodule_pyngraph_runtime.cpp',
    'pyngraph/runtime/tensor.cpp',
    'pyngraph/passes/manager.cpp',
    'pyngraph/passes/regmodule_pyngraph_passes.cpp',
    'pyngraph/types/element_type.cpp',
    'pyngraph/types/regmodule_pyngraph_types.cpp',
]

package_dir = {
    'ngraph': PYNGRAPH_ROOT_DIR + '/ngraph',
    'ngraph.utils': PYNGRAPH_ROOT_DIR + '/ngraph/utils',
    'ngraph.impl': PYNGRAPH_ROOT_DIR + '/ngraph/impl',
    'ngraph.impl.op': PYNGRAPH_ROOT_DIR + '/ngraph/impl/op',
    'ngraph.impl.op.util': PYNGRAPH_ROOT_DIR + '/ngraph/impl/op/util',
    'ngraph.impl.passes': PYNGRAPH_ROOT_DIR + '/ngraph/impl/passes',
    'ngraph.impl.runtime': PYNGRAPH_ROOT_DIR + '/ngraph/impl/runtime',
}
packages = [
    'ngraph',
    'ngraph.utils',
    'ngraph.impl',
    'ngraph.impl.op',
    'ngraph.impl.op.util',
    'ngraph.impl.passes',
    'ngraph.impl.runtime',
]

sources = [PYNGRAPH_ROOT_DIR + '/' + source for source in sources]

include_dirs = [PYNGRAPH_ROOT_DIR, NGRAPH_CPP_INCLUDE_DIR, PYBIND11_INCLUDE_DIR]

library_dirs = [NGRAPH_CPP_LIBRARY_DIR]

libraries = ['ngraph']

extra_compile_args = []

extra_link_args = []

data_files = [
    (
        'lib',
        [
            NGRAPH_CPP_LIBRARY_DIR + '/' + library
            for library in os.listdir(NGRAPH_CPP_LIBRARY_DIR)
        ],
    ),
    (
        'licenses',
        [
            NGRAPH_CPP_DIST_DIR + '/licenses/' + license
            for license in os.listdir(NGRAPH_CPP_DIST_DIR + '/licenses')
        ],
    ),
    (
        '',
        [NGRAPH_CPP_DIST_DIR + '/LICENSE'],
    ),
]

ext_modules = [
    Extension(
        '_pyngraph',
        sources=sources,
        include_dirs=include_dirs,
        define_macros=[('VERSION_INFO', __version__)],
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

if NGRAPH_ONNX_IMPORT_ENABLE in ['TRUE', 'ON', True]:
    onnx_sources = [
        'pyngraph/pyngraph_onnx_import.cpp',
        'pyngraph/onnx_import/onnx_import.cpp',
    ]
    onnx_sources = [PYNGRAPH_ROOT_DIR + '/' + source for source in onnx_sources]

    package_dir['ngraph.impl.onnx_import'] = (
        PYNGRAPH_ROOT_DIR + '/ngraph/impl/onnx_import'
    )
    packages.append('ngraph.impl.onnx_import')

    ext_modules.append(
        Extension(
            '_pyngraph_onnx_import',
            sources=onnx_sources,
            include_dirs=include_dirs,
            define_macros=[('VERSION_INFO', __version__)],
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
            language='c++',
        ),
    )


def add_platform_specific_link_args(link_args):
    """Add linker flags specific for actual OS."""
    if sys.platform.startswith('linux'):
        link_args += ['-Wl,-rpath,$ORIGIN/../..']
        link_args += ['-z', 'noexecstack']
        link_args += ['-z', 'relro']
        link_args += ['-z', 'now']
    elif sys.platform == 'darwin':
        link_args += ['-Wl,-rpath,@loader_path/../..']


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def _add_extra_compile_arg(self, flag, compile_args):
        """Return True if successfully added given flag to compiler args."""
        if has_flag(self.compiler, flag):
            compile_args += [flag]
            return True
        return False

    def build_extensions(self):
        """Build extension providing extra compiler flags."""
        if sys.platform == 'win32':
            raise RuntimeError('Unsupported platform: win32!')
        # -Wstrict-prototypes is not a valid option for c++
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass
        for ext in self.extensions:
            ext.extra_compile_args += [cpp_flag(self.compiler)]

            if not self._add_extra_compile_arg('-fstack-protector-strong', ext.extra_compile_args):
                self._add_extra_compile_arg('-fstack-protector', ext.extra_compile_args)

            self._add_extra_compile_arg('-fvisibility=hidden', ext.extra_compile_args)
            self._add_extra_compile_arg('-flto', ext.extra_compile_args)
            self._add_extra_compile_arg('-fPIC', ext.extra_compile_args)
            add_platform_specific_link_args(ext.extra_link_args)

            ext.extra_compile_args += ['-Wformat', '-Wformat-security']
            ext.extra_compile_args += ['-O2', '-D_FORTIFY_SOURCE=2']
        build_ext.build_extensions(self)


with open(os.path.join(PYNGRAPH_ROOT_DIR, 'requirements.txt')) as req:
    requirements = req.read().splitlines()

setup(
    name='ngraph-core',
    description='nGraph - Intel\'s graph compiler and runtime for Neural Networks',
    version=__version__,
    author='Intel',
    author_email='intelnervana@intel.com',
    url='https://github.com/NervanaSystems/ngraph/',
    license='License :: OSI Approved :: Apache Software License',
    long_description=open(os.path.join(PYNGRAPH_ROOT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    package_dir=package_dir,
    packages=packages,
    cmdclass={'build_ext': BuildExt},
    data_files=data_files,
    setup_requires=['numpy'],
    install_requires=requirements,
    zip_safe=False,
)
