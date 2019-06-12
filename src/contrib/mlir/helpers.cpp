//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <stdint.h>
#include "ngraph/ngraph_visibility.hpp"
#include <mlir/ExecutionEngine/MemRefUtils.h>

/// Call back to copy Index tensor to Int tensor
/// Can handle int tensors of bitwidth 8, 16, 32 and 64
/// Index width is always intptr_t
extern "C" NGRAPH_API void* __mlir_convert_index_to_int(mlir::StaticFloatMemRef dst, mlir::StaticFloatMemRef src, size_t numElements, size_t intWidth)
{
    size_t indexSize = sizeof(intptr_t);
    auto pSrc = reinterpret_cast<intptr_t*>(src.data);
    auto pDst = reinterpret_cast<char*>(dst.data);
    for (auto i = 0; i < numElements; i++)
    {
        switch(intWidth)
        {
        case 8:
            *pDst = static_cast<char>(pSrc[i]);
            pDst++;
            break;
        case 16:
            *(short*)pDst = static_cast<short>(pSrc[i]);
            pDst += sizeof(short);
            break;
        case 32:
            *(int*)pDst = static_cast<int>(pSrc[i]);
            pDst += sizeof(int);
            break;
        case 64:
            *(long*)pDst = static_cast<long>(pSrc[i]);
            pDst += sizeof(long);
            break;
        }
    }
}