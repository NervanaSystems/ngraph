# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include( ../cmake/external_gtest.cmake )
include( ../cmake/external_eigen.cmake )
include( ../cmake/external_mkldnn.cmake )
if (NGRAPH_USE_PREBUILT_LLVM OR DEFINED LLVM_TARBALL_URL)
    include( ../cmake/external_llvm_prebuilt.cmake )
else()
    include( ../cmake/external_llvm.cmake )
endif()
include( ../cmake/external_tbb.cmake )
include( ../cmake/external_argon_transformer.cmake )
