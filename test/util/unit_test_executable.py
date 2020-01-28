import logging as log
import sys
import os
import subprocess
# from pathlib import Path

import pytest

log.basicConfig(format="[ %(levelname)s ]  %(msg)s", stream=sys.stdout, level=log.INFO)

# paths_to_import = [str(Path(__file__).parent.parent.parent / 'model-optimizer')]


# def setup_module():
#     sys.path.extend(paths_to_import)
#     try:
#         from mo.utils.ir_engine.ir_engine import IREngine
#     except ImportError:
#         map(lambda x: sys.path.remove(x), paths_to_import)
#         raise ImportError('Can not import core module of comparator: IREngine\nPlease add following paths {} into '
#                           '`PYTHONPATH` environment variable manually and restart the test'.format(paths_to_import))
#
#
# def teardown_module():
#     map(lambda x: sys.path.remove(x), paths_to_import)


# def teardown_function(function):
#     """
#     Clean up ICNNs from current directory
#     :return: """
#     xmls = list(Path.cwd().rglob("*.{}".format('xml')))
#     xmls = [x for x in xmls if 'compare_report' not in x.name]
#     bins = list(Path.cwd().rglob("*.{}".format('bin')))
#     [fn.unlink() for fn in (xmls + bins)]


def shell(cmd, env=None, cwd=None):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :return:
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "unset OMP_NUM_THREADS; " + cmd]
    else:
        cmd = " ".join(cmd)

    sys.stdout.write("Running command:\n" + "".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        print(line.rstrip())
        if line == '' and p.poll() != None:
            break
    return p.returncode, ''.join(stdout)


def test(gtest_filter):
    executable = os.path.join("/home/abelyako/repositories/dldt/bin/intel64/Debug", "unit-test")
    cmd_line = executable + ' --gtest_filter=' + gtest_filter
    retcode, stdout = shell(cmd=cmd_line)
    # Parsing output
    stdout = stdout.split('\n')
    nothing_flag = 0
    for line in stdout:
        if 'UNSUPPORTED OPS DETECTED!' in line:
            pytest.skip('Skip from pytest because unit-test send error UNSUPPORTED OPS DETECTED!')
        elif 'Nodes in test:' in line:
            nothing_flag = 1
    if not nothing_flag:
        pytest.skip('Skip from pytest because inside test no one ngraph function created')

    # TODO Add check for the following cases (when nothing happened in test:
    # [ RUN      ] CPU.reverse_v1_incorrect_rev_axes_rank_index_mode
    # [       OK ] CPU.reverse_v1_incorrect_rev_axes_rank_index_mode (0 ms)

    # TODO make this check above
    assert retcode == 0, "unit-test execution failed. Return code: {}".format(retcode)
    print('a')
    # print(stdout)


if __name__ == '__main__':
    log.warning("Please run {} by pytest like so:\npytest {} --gtest_attr=<attributes for gtest>")
                # "".format(Path(__file__).name, Path(__file__).resolve()))
