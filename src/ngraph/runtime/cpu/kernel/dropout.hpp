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

#pragma once

#include <random>

#include "ngraph/shape.hpp"
#include "ngraph/state/rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                // Note: this kernel is for doing upscale in train
                template <typename T>
                void generate_dropout(T* input,
                                      T* out0,
                                      T* out1_mask,
                                      size_t nelems,
                                      bool training,
                                      const double value,
                                      const std::vector<std::minstd_rand>& vmsr)
                {
                    if (training)
                    {
                        double dropout_prob = 1 - value;
                        size_t nthr = ngraph::runtime::cpu::executor::GetCPUExecutor().get_num_cores();
                        size_t chunk_size = (nelems + nthr - 1) / nthr;

#pragma omp parallel num_threads(nthr)
                        {
                            size_t tid = omp_get_thread_num();
                            std::minstd_rand msr = vmsr[tid];
                            std::uniform_int_distribution<> gen(0, 1);

                            size_t idx_start = tid * chunk_size;
                            size_t idx_end = std::min(idx_start + chunk_size, nelems);
                            for (size_t idx = idx_start; idx < idx_end; ++idx)
                            {
                                if (static_cast<T>(gen(msr)) < dropout_prob)
                                {
                                    out1_mask[idx] = 0;
                                    out0[idx] = 0;
                                }
                                else
                                {
                                    out1_mask[idx] = 1;
                                    out0[idx] = input[idx] / static_cast<T>(value);
                                }
                            }
                        }
                    }
                    else
                    {
                        // this is inference, ideally it should be optimized earlier
                        for (size_t i = 0; i < nelems; i++)
                        {
                            out1_mask[i] = 1;
                            out0[i] = static_cast<T>(1);
                        }
                    }
                }
            }
        }
    }
}
