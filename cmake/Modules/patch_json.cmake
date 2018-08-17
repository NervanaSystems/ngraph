set(FILE_NAME ${CMAKE_BINARY_DIR}/include/nlohmann/detail/macro_scope.hpp)
file(READ ${FILE_NAME} FILE_CONTENTS)
string(REPLACE
  "#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40900"
  "#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40805"
  REWRITTEN_FILE
  "${FILE_CONTENTS}"
)
file(WRITE ${FILE_NAME} "${REWRITTEN_FILE}")
message(STATUS "json library gcc minimum version number patched")
