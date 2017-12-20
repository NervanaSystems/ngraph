# Docker Builds for the NGraph-Cpp _Reference-OS_

## Introduction

This directory contains a basic build system for creating docker images of the _reference-OS_ on which NGraph-Cpp builds and unit tests are run.  The purpose is to provide reference builds for _Continuous Integration_ used in developing and testing NGraph-Cpp.

The `Makefile` provides targets for:

* Building the _reference-OS_ into a docker image
* Building NGraph-Cpp and running unit tests in this cloned repo, mounted into the docker image of the _reference-OS_
* Starting an interactive shell in the _reference-OS_ docker image, with the cloned repo available for manual builds and unit testing

The _make targets_ are designed to handle all aspects of building the _reference-OS_ docker image, running NGraph-Cpp builds and unit testing in it, and opening up a session in the docker image for interactive use.  You should not need to issue any manual commands (unless you want to).  In addition the `Dockerfile.ngraph_cpp_cpu` provides a description of how the _reference-OS_ is built, should you want to build your own server or docker image.

## Prerequisites

In order to use the make targets, you will need to do the following:

* Have docker installed on your computer with the docker daemon running.
* These scripts assume that you are able to run the `docker` command without using `sudo`.  You will need to add your account to the `docker` group so this is possible.
* If your computer (running docker) sits behind a firewall, you will need to have the docker daemon properly configured to use proxies to get through the firewall, so that public docker registries and git repositories can be accessed.
* You should _not_ run `make check_*` targets from a directory in an NFS filesystem, if that NFS filesystem uses _root squash_ (see **Notes** section below).  Instead, run `make check_*` targets from a cloned repo in a local filesystem.

## Make Targets

The _make targets_ are designed to provide easy commands to run actions using the docker image.  All _make targets_ should be issued on the host OS, and _not_ in a docker image.

Most make targets are structured in the form `<action>_<compiler>`.  The `<action>` indicates what you want to do (e.g. build, check, install), while the `<compiler>` indicates what you want to build with (i.e. gcc or clang).

* In general, you simply need to run the command **`make check_all`**.  This first makes the `build_docker_ngraph_cpp` target as a dependency.  Then it makes the `build_*` and `check_*` targets, which will build NGraph-Cpp using _cmake_ and _make_ and then run unit testing.  Please keep in mind that `make check_*` targets do not work when your working directory is in an NFS filesystem that uses _root squash_ (see **Notes** section below).

* Two builds are supported: building with `gcc` and `clang`.  Targets are named `*_gcc` and `*_clang` when they refer to building with a specific compiler, and the `*_all` targets are available to build with both compilers.  Output directories are BUILD-GCC and BUILD-CLANG, at the top level.

* You can also run the command **`make shell`** to start an interactive bash shell inside the docker image.  While this is not required for normal builds and unit testing, it allows you to run interactively within the docker image with the cloned repo mounted.  Again, `build_docker_ngraph_cpp` is made first as a dependency.  Please keep in mind that `make shell` does not work when your working directory is in an NFS filesystem that uses _root squash_ (see **Notes** section below).

* Running the command **`make build_docker_ngraph_cpp`** is also available, if you simply want to build the docker image.  This target does work properly when your working directory is in an NFS filesystem.

* Finally, **`make clean`** is available to clean up the BUILD-* and docker build directories.

Note that all operations performed inside the the docker image are run as a regular user, using the `run-as-user.sh` script.  This is done to avoid writing root-owned files in mounted filesystems.

## Helper Scripts

These helper scripts are included for use in the `Makefile` and automated (Jenkins) jobs.  **These scripts should _not_ be called directly unless you understand what they do.**

#### `docker_cleanup.sh`

A helper script for Jenkins jobs to clean up old exited docker containers and `ngraph_*` docker images.

#### `run_as_user.sh`

A helper script to run as a normal user within the docker container.  This is done to avoid writing root-owned files in mounted filesystems.

## Notes

* The top-level `Makefile` in this cloned repo can be used to build and unit-test NGraph-Cpp _outside_ of docker.  This directory is only for building and running unit tests for NGraph-Cpp in the _reference-OS_ docker image.

* Due to limitations in how docker mounts work, `make check_cpu` and `make shell` will fail if you try to run them while in a working directory that is in an NFS-mount that has _root squash_ enabled.  The cause results from the process in the docker container running as root.  When a file or directory is created by root in the mounted directory tree, from within the docker image, the NFS-mount (in the host OS) does not allow a root-created file, leading to a permissions error.  This is dependent on whether the host OS performs "root squash" when mounting NFS filesystems.  The fix to this is easy: run `make check_cpu` and `make shell` from a local filesystem.

* These make targets have been tested on Ubuntu 16.04 with docker installed and the docker daemon properly configured.  Some adjustments may be needed to run these on other OSes.
