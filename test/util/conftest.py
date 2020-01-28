import logging as log
import sys
import subprocess
import os
# import re
# from pathlib import Path
# from shutil import copyfile
# from typing import Iterable, List

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gtest_filter",
        help="Attributes to gtest",
        type=str,
        required=True,
    )

@pytest.fixture(scope="session")
def gtest_filter(request):
    return request.config.getoption('gtest_filter')


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


def create_list_test(stdout):
    # Example of stdout content:
    # 'CPU'
    # ' zero_sized_abs'
    # ' zero_sized_ceiling'
    # ...
    # So, list of test will be concatenation of 'CPU' and the second part (starting with ' '):
    # 'CPU.zero_sized_abs'
    # 'CPU.zero_sized_ceiling'
    list_test = []
    first_name, second_name = [''] * 2
    for line in stdout:
        if not line.startswith(' '):
            first_name = line
        else:
            second_name = line
            # Several test has gtest mark 'DISABLED' inside test - no test will be executed
            if not 'DISABLED' in line:
                list_test.append(first_name + second_name.strip())
    return list_test

def pytest_generate_tests(metafunc):
    gtest_filter = metafunc.config.getoption(name='gtest_filter')
    if 'gtest_filter' in metafunc.fixturenames and gtest_filter is not None:
        executable = os.path.join("/home/abelyako/repositories/dldt/bin/intel64/Debug", "unit-test")
        cmd_line = executable + ' --gtest_filter=' + gtest_filter + ' --gtest_list_tests'
        log.info('Executing {} for getting list of test'.format(executable))
        retcode, stdout = shell(cmd=cmd_line)
        assert retcode == 0, "unit-test --gtest_list_tests execution failed. Return code: {}".format(retcode)
        stdout = stdout.split('\n')
        list_test = create_list_test(stdout)
        print('a')
        metafunc.parametrize(argnames="gtest_filter", argvalues=list_test)