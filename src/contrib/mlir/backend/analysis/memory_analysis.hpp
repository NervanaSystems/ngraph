//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#pragma once

#include <mlir/Pass/Pass.h>
#include <unordered_map>
#include "ngraph/check.hpp"

namespace mlir
{
    // BufferInfo
    struct BufferInfo
    {
        // Buffer Id. If -1 then invalid buffer.
        int m_bufferId;
        // Offset into the buffer
        int m_offset;
        bool isValid() const { return m_bufferId != -1; }
    };

    struct MemoryAnalysis
    {
        using BufferInfoMap = std::unordered_map<Operation*, BufferInfo>;
        using BufferSizeMap = std::unordered_map<unsigned, unsigned>;
        // Compute this analysis with the provided operation.
        MemoryAnalysis(Operation* op);
        BufferInfo getBufferInfo(Operation* op)
        {
            auto it = m_bufferInfo.find(op);
            if (it == m_bufferInfo.end())
            {
                return {-1, -1};
            }
            return it->second;
        }
        void setBufferInfo(Operation* op, BufferInfo bufferInfo) { m_bufferInfo[op] = bufferInfo; }
        void setBufferSize(unsigned bufferId, unsigned size)
        {
            auto it = m_bufferSize.find(bufferId);
            if (it != m_bufferSize.end())
            {
                it->second = (size > it->second) ? size : it->second;
            }
            else
            {
                m_bufferSize[bufferId] = size;
            }
        }
        unsigned getBufferSize(unsigned bufferId)
        {
            auto it = m_bufferSize.find(bufferId);
            NGRAPH_CHECK(it != m_bufferSize.end(), "Buffer has no size!");
            return it->second;
        }

    private:
        // Records assignment of BufferInfo to each inplace op
        BufferInfoMap m_bufferInfo;
        // Records buffer size required for each buffer id in bytes
        BufferSizeMap m_bufferSize;
    };
}
