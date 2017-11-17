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


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'wrapper.ngraph.types.clsTraitedType',
        ['wrapper/ngraph/types/element_type.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.ops.clsParameter',
       ['wrapper/ngraph/ops/parameter.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsTensorViewType',
       ['wrapper/ngraph/runtime/tensor_view.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.clsFunction',
       ['wrapper/ngraph/function.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsManager',
       ['wrapper/ngraph/runtime/manager.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsBackend',
       ['wrapper/ngraph/runtime/backend.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsExternalFunction',
       ['wrapper/ngraph/runtime/external_function.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsNDArray',
       ['wrapper/ngraph/runtime/ndarray.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsCallFrame',
       ['wrapper/ngraph/runtime/call_frame.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'wrapper.ngraph.runtime.clsParameterizedTensorView',
       ['wrapper/ngraph/runtime/parameterized_tensor_view.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            os.environ["PYBIND_HEADERS_PATH"],
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),


]



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


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append('-I')
        opts.append(os.environ['NGRAPH_CPP_BUILD_PATH'])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='ngraph neon',
    version=__version__,
    author='Me',
    author_email='me@i.com',
    url='https://github.com/NervanaSystems/ngraph-neon',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
