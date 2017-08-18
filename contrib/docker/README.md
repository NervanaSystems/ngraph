## Usage

From this directory you can:

### make build_ngraph_cpp_cpu

will build Docker image for building NGraph-Cpp and running unit tests

### make check_cpu

will set up a cmake BUILD directory and run "make check" (unit tests) in
the docker image

### make shell

will put you into a bash shell with NGraph-Cpp installed

### Notes

Due to limitations in how docker mounts work, "make check_cpu" will fail
if you try to run it from an NFS-mounted directory.  The cause results
from the process in the docker container running as root.  When a file or
directory is created by root in the mounted directory tree, from within
the container, the NFS-mount (in the host OS) does not allow a root-created
file.  This is dependent on whether the host OS performs "root squash" when
mounting NFS filesystems.  The fix to this is easy: run "make check_cpu" from
a local filesystem.
