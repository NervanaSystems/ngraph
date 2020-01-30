import logging as log
import sys
import os
import subprocess
import csv
import pytest
import re

log.basicConfig(format="[ %(levelname)s ]  %(msg)s", stream=sys.stdout, level=log.INFO)

pytest.operation_dictionary = {}
pytest.avaliable_plugins = []


def setup_module():
    try:
        os.environ.get('PATH_TO_EXE')
    except KeyError:
        raise ImportError('PATH_TO_EXE is upsent in your environment variables. '
                          'Please, do "export PATH_TO_EXE=<path to unit-test>')


def teardown_module():
    """
    Creating CSV file at the end of test with nGraph nodes coverage
    :return:
    """
    csv_path = "nodes_coverage.csv"
    header = ["Operation"] + [p + " passed / total" for p in pytest.avaliable_plugins]
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(i for i in header)
        for key in sorted(pytest.operation_dictionary):
            line = [key]
            for plugin in pytest.avaliable_plugins:
                if not plugin in pytest.operation_dictionary[key]:
                    line.append('0')
                else:
                    line.append('/'.join(str(x) for x in pytest.operation_dictionary[key][plugin]))
            csv_writer.writerow(line)


def shell(cmd, env=None):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :return:
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "unset OMP_NUM_THREADS; " + cmd]
    else:
        cmd = " ".join(cmd)

    sys.stdout.write("Running command:\n" + "".join(cmd) + "\n")
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
    executable = os.path.join(os.environ.get('PATH_TO_EXE'), "unit-test")
    cmd_line = executable + ' --gtest_filter=' + gtest_filter
    retcode, stdout = shell(cmd=cmd_line)

    # Parsing output of single test
    stdout = stdout.split('\n')
    nodes_list = []
    for line in stdout:
        if 'UNSUPPORTED OPS DETECTED!' in line:
            pytest.skip('Skip from pytest because unit-test send error UNSUPPORTED OPS DETECTED!')
        elif 'Nodes in test:' in line:
            nodes_list = list(set(line.replace('Nodes in test:', '').strip().split(' ')))

    if not nodes_list:
        pytest.skip('Skip from pytest because inside test no one ngraph function created')

    # Added one more loop, because condition below must be executed only if some nodes_list found
    # (it means that test includes opset1 operations)
    for line in stdout:
        if re.match('.*1 test from\s([A-Z]+)', line):
            matches = re.search(r'.*1 test from\s([A-Z]+)', line)
            plugin = matches.group(1)
            if plugin not in pytest.avaliable_plugins:
                pytest.avaliable_plugins.append(plugin)

    # Filling dictionary with operation coverage
    # How many time one operation is tested
    for n in nodes_list:
        if not n in pytest.operation_dictionary:
            pytest.operation_dictionary[n] = {}
        if plugin in pytest.operation_dictionary[n]:
            numerator, denominator = pytest.operation_dictionary[n][plugin]
            pytest.operation_dictionary[n][plugin] = (numerator if retcode != 0 else numerator + 1,
                                                      denominator + 1)
        else:
            pytest.operation_dictionary[n][plugin] = (0, 1) if retcode != 0 else (1, 1)

    # This check is at the end, because with 99% it will return 0 or 1 (when function check of test failed)
    # Because the same cmd line executed by pytest_generate_tests with --gtest_list_tests.
    # So, most of the issue cached there.
    assert retcode == 0, "unit-test execution failed. Gtest failed. Return code: {}".format(retcode)


if __name__ == '__main__':
    log.warning("Please run {} by pytest like so:\npytest {} --gtest_filter=<attributes for gtest_filter>")
