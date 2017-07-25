if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
  endif()
endif()

if(PYTHON_EXECUTABLE)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
        "import numpy as n; print(n.get_include());"
        RESULT_VARIABLE NUMPY_RESULT
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NUMPY_RESULT MATCHES 0)
        set(NUMPY_FOUND true)
        set(NUMPY_LIBRARIES "")
        set(NUMPY_DEFINITIONS "")
    endif(NUMPY_RESULT MATCHES 0)
endif(PYTHON_EXECUTABLE)

