
add_mlir_dialect_library(MLIRNgraph
        NgraphDialect.cpp
        NgraphOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Ngraph

        DEPENDS
        MLIRNgraphOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
