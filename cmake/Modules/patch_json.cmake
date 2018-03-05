message("##################################### 2 patch here")
message(STATUS "CMAKE_BINARY_DIR  ${CMAKE_BINARY_DIR}")
set(FILE_NAME ${CMAKE_BINARY_DIR}/include/nlohmann/detail/macro_scope.hpp)
message(STATUS "FILE_NAME ${FILE_NAME}")
file(READ ${FILE_NAME} FILE_CONTENTS)
string(REPLACE
  "#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40900"
  "#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40805"
  REWRITTEN_FILE
  ${FILE_CONTENTS}
)
message(STATUS ${REWRITTEN_FILE})
file(WRITE ${FILE_NAME} ${REWRITTEN_FILE})
