set (SRC
    util.cpp
    transformers/exop.cpp
    transformers/op_graph.cpp
    transformers/axes.cpp
    transformers/mock_transformer.cpp
    element_type.cpp
    names.cpp
    transformers/ndarray.cpp
    strides.cpp
    tree.cpp
)

# file(GLOB DEPLOY_HEADERS_ABS *.hpp)
# foreach(DEPLOY_HEADER ${DEPLOY_HEADERS_ABS})
#     get_filename_component(FNAME ${DEPLOY_HEADER} NAME)
#     set(DEPLOY_HEADERS ${DEPLOY_HEADERS} ${FNAME})
# endforeach()

# if (PYTHON_FOUND)
# set(SRC ${SRC} api.cpp)
# endif(PYTHON_FOUND)

# set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
# set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/../setup.py")
# configure_file(${SETUP_PY_IN} ${SETUP_PY})

include_directories(.)
add_library(ngraph SHARED ${SRC})

# install(TARGETS aeon DESTINATION lib)
# install(FILES ${DEPLOY_HEADERS} DESTINATION include/aeon)