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

requirements = [
    "numpy",
    "pybind11",
]
include_dirs = [# Path to pybind11 headers
                os.environ["PYBIND_HEADERS_PATH"],
                get_pybind_include(),
                get_pybind_include(user=True)
               ]
ext_modules = [Extension('nwrapper.ngraph.types.Type',
                        ['nwrapper/ngraph/types/element_type.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Parameter',
                        ['nwrapper/ngraph/ops/parameter.cpp'], include_dirs),
               Extension('nwrapper.ngraph.types.TensorViewType',
                        ['nwrapper/ngraph/types/type.cpp'], include_dirs),
               Extension('nwrapper.ngraph.runtime.TensorView',
                        ['nwrapper/ngraph/runtime/tensor_view.cpp'], include_dirs),
               Extension('nwrapper.ngraph.Function',
                        ['nwrapper/ngraph/function.cpp'], include_dirs),
               Extension('nwrapper.ngraph.Node',
                        ['nwrapper/ngraph/node.cpp'], include_dirs),
               Extension('nwrapper.ngraph.runtime.Manager',
                        ['nwrapper/ngraph/runtime/manager.cpp'], include_dirs),
               Extension('nwrapper.ngraph.runtime.Backend',
                        ['nwrapper/ngraph/runtime/backend.cpp'], include_dirs),
               Extension('nwrapper.ngraph.runtime.ExternalFunction',
                        ['nwrapper/ngraph/runtime/external_function.cpp'], include_dirs),
               Extension('nwrapper.ngraph.runtime.CallFrame',
                        ['nwrapper/ngraph/runtime/call_frame.cpp'], include_dirs),
               Extension('nwrapper.ngraph.Util',
                        ['nwrapper/ngraph/util.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Op',
                        ['nwrapper/ngraph/ops/op.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Add',
                        ['nwrapper/ngraph/ops/add.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Dot',
                        ['nwrapper/ngraph/ops/dot.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Maximum',
                        ['nwrapper/ngraph/ops/maximum.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Divide',
                        ['nwrapper/ngraph/ops/divide.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Minimum',
                        ['nwrapper/ngraph/ops/minimum.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Exp',
                        ['nwrapper/ngraph/ops/exp.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Sum',
                        ['nwrapper/ngraph/ops/sum.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Greater',
                        ['nwrapper/ngraph/ops/greater.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Less',
                        ['nwrapper/ngraph/ops/less.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Log',
                        ['nwrapper/ngraph/ops/log.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Convert',
                        ['nwrapper/ngraph/ops/convert.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Subtract',
                        ['nwrapper/ngraph/ops/subtract.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Negative',
                        ['nwrapper/ngraph/ops/negative.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Reshape',
                        ['nwrapper/ngraph/ops/reshape.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Broadcast',
                        ['nwrapper/ngraph/ops/broadcast.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Reduce',
                        ['nwrapper/ngraph/ops/reduce.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.OneHot',
                        ['nwrapper/ngraph/ops/one_hot.cpp'], include_dirs),
               Extension('nwrapper.ngraph.ops.Constant',
                        ['nwrapper/ngraph/ops/constant.cpp'], include_dirs)]

# As of Python 3.6, CCompiler has a `has_flag` m ethod.
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
        if "NGRAPH_CPP_BUILD_PATH" in os.environ:
            NGRAPH_CPP_INSTALL_PATH = os.environ["NGRAPH_CPP_BUILD_PATH"]
            opts.append('-I')
            opts.append('%s/include'%(NGRAPH_CPP_INSTALL_PATH))
            if ct == 'unix':
                opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
                opts.append(cpp_flag(self.compiler))
                if has_flag(self.compiler, '-fvisibility=hidden'):
                    opts.append('-fvisibility=hidden')
            elif ct == 'msvc':
                opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            for ext in self.extensions:
                ext.library_dirs = ['%s/lib'%(NGRAPH_CPP_INSTALL_PATH)]
                ext.extra_compile_args = opts
                ext.extra_link_args = ["-shared", "-lngraph", "-Wl,-rpath,%s/lib"%(NGRAPH_CPP_INSTALL_PATH)]
            build_ext.build_extensions(self)
        else:
            print("NGRAPH_CPP_BUILD_PATH, not defined")


setup(
    name='ngraph neon',
    version=__version__,
    author='Me',
    author_email='me@i.com',
    url='https://github.com/NervanaSystems/ngraph-neon',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=requirements,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
