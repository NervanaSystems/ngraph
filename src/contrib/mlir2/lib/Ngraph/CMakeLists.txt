
add_mlir_dialect_library(MLIRNgraph
        NgraphDialect.cpp
        NgraphOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Ngraph

        DEPENDS
        MLIRNgraphOpsIncGen
        MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
