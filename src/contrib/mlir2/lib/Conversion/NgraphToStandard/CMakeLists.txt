message(STATUS "*************** MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_INCLUDE_DIR}")

add_mlir_conversion_library(MLIRNgraphToStandard
    NgraphToStandard.cpp

    ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/NgraphToStandard

    DEPENDS
        MLIRNgraphOpsIncGen
        MLIRConversionPassIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRNgraph
)
