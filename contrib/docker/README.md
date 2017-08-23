# Docker Builds for the NGraph-Cpp _Reference OS_

## Introduction

This directory contains a basic build system for creating docker images of the _reference OS_ on which NGraph-Cpp builds and unit tests are run.  The purpose is to provide reference builds for _Continuous Integration_ used in developing and testing NGraph-Cpp.

The `Makefile` provides targets for:

* Building the reference OS into a docker image
* Building NGraph-Cpp and running unit tests in this cloned repo, mounted into the docker image of the reference OS
* Starting a shell in the _reference OS_ docker image, with the cloned repo available for manual builds and unit testing

## Prerequisites

In order to use the make targets, you will need to do the following:

* Have docker installed on your computer with the docker daemon running
* If your computer (running docker) sits behind a firewall, you will need to have the docker daemon properly configured to use proxies to get through the firewall, so that public docker registries and git repositories can be accessed
* You should _not_ run `make check_cpu` from a directory in an NFS filesystem, if that NFS filesystem uses _root squash_ (see **Notes** section below)
  - Instead, run `make check_cpu` from a cloned repo in a local filesystem

## Make Targets

The following targets allow you to perform basic operations for building the _reference OS_ docker image, building NGraph-Cpp, running unit tests, and starting a shell in the _reference OS_ docker image.

### `make build_ngraph_cpp_cpu`

Build a docker image of the _reference OS_ used to build NGraph-Cpp and run unit tests

### `make check_cpu`

Build NGraph-Cpp in the _reference OS_ docker image.  The results are built in this cloned repo by mounting it into the docker image.  This target will set up a cmake BUILD directory and the top level of the cloned repo, run cmake and make, and finally run unit tests in the docker image.

### `make shell`

This target will start a bash shell in the _reference OS_ docker image, with a mount to the cloned repo available in the docker image.

## Helper Scripts

These helper scripts are included for use in the `Makefile` and Jenkins jobs.  **Generally, these scripts should _not_ be called directly.**

### `chown_files.sh`

Used in the `Makefile` to change the ownership of files while running _inside the docker image_.  This is used to fix a problem with docker where files written in a mounted filesystem, from within the docker image, will end up being owned by root in the host OS.  This leads to problems with Jenkins not being able to clean up its own job directories if docker images are used.

### `docker_cleanup.sh`

A helper script for Jenkins jobs to clean up old docker images.

## Notes

* The top-level `Makefile` in this cloned repo can be used outside of docker.  This directory is only for building and running unit tests for NGraph-Cpp in the _reference OS_ docker image.

* The `_cpu` in the targets refers to the docker image being built such that testing is in a _cpu-only_ environment.  Later, _gpu_ and other environments may be added.  This convention was chosen to stay consistent with the _NervanaSystems ngraph_ project on Github.

* Due to limitations in how docker mounts work, `make check_cpu` will fail
if you try to run it from an NFS-mounted directory which has _root squash_ enabled.  The cause results from the process in the docker container running as root.  When a file or directory is created by root in the mounted directory tree, from within the container, the NFS-mount (in the host OS) does not allow a root-created file, leading to a permissions error.  This is dependent on whether the host OS performs "root squash" when mounting NFS filesystems.  The fix to this is easy: run `make check_cpu` from a local filesystem.

* These make targets have been tested on Ubuntu 14.04 and Ubuntu 16.04 with docker installed and the docker daemon properly configured.  Some adjustments may be needed to run these on other OSes.
