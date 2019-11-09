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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#pragma once

#include <mlir/Pass/Pass.h>
#include <unordered_map>
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

    struct MemoryAnalysis {
        using BufferInfoMap = std::unordered_map<Operation*, BufferInfo>;
        // Compute this analysis with the provided operation.
        MemoryAnalysis(Operation *op);
        
        BufferInfo getBufferInfo(Operation *op)
        {
            auto it = m_bufferInfo.find(op);
            if (it == m_bufferInfo.end())
            {
                return {-1, -1};
            }
            return it->second;
        }
        void setBufferInfo(Operation *op, BufferInfo bufferInfo)
        {
            m_bufferInfo[op] = bufferInfo;
        }
        private:
        BufferInfoMap m_bufferInfo;
    };
}
