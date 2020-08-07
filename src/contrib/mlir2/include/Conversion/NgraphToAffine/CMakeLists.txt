add_mlir_conversion_library(MLIRNgraphToAffine
    NgraphToAffine.cpp

    ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

   LINK_LIBS PUBLIC
        # NgraphOps
        # AffineOps
        # StandardOps
)
