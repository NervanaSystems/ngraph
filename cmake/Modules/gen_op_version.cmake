cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0007 NEW)

# Make the string OPSETS into a list by replacing spaces with semicolons
string(REPLACE " " ";" OPSETS "${OPSETS}")
set(ALL_OPS)
foreach(OPSET ${OPSETS})
    file(STRINGS ${CURRENT_DIR}/${OPSET} CONTENTS)

    # Remove // style comments
    list(TRANSFORM CONTENTS REPLACE "( |)//.*" "")

    # Remove # C preprocessor statements
    list(TRANSFORM CONTENTS REPLACE "#.*" "")

    # Convert ngraph::op::vN to N
    list(TRANSFORM CONTENTS REPLACE "ngraph::op::v" "")

    list(APPEND ALL_OPS ${CONTENTS})
endforeach()
list(REMOVE_DUPLICATES ALL_OPS)
list(SORT ALL_OPS)
string(REPLACE ";" "\n" NEW_CONTENTS "${ALL_OPS}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/op_version_tbl.hpp ${NEW_CONTENTS})
