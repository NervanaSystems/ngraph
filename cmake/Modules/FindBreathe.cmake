find_program(BREATHE_EXECUTABLE
             NAMES breathe-apidoc
             DOC "Path to breathe executable")

# Handle REQUIRED and QUIET arguments
# this will also set SPHINX_FOUND to true if SPHINX_EXECUTABLE exists
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Breathe
                                  "Failed to locate breathe executable"
                                  BREATHE_EXECUTABLE)
