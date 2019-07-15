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
                                      const size_t nelems,
                                      const bool training,
                                      const double keep_prob,
                                      const std::vector<std::minstd_rand>& vmsr,
                                      const bool use_seed)
                {
                    if (training)

                    {
                        int32_t rnd_seed = rand();
                        double dropout_prob = 1 - keep_prob;
#ifdef _OPENMP
                        size_t nthr =
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_num_cores();
                        size_t chunk_size = (nelems + nthr - 1) / nthr;

#pragma omp parallel num_threads(nthr)
                        {
                            size_t tid = omp_get_thread_num();
#else
                        size_t chunk_size = nelems;
                        {
                            size_t tid = 0;
#endif
                            // Note :
                            // In this implementation of dropout, we are trying to be same as PDPD
                            // native implementation (and other frameworks).
                            // https://github.com/NervanaSystems/ngraph-paddle/blob/14d88829b386c9f7601788c5539c08326dcbe2fe/paddle/fluid/operators/dropout_op.h#L58-L78
                            // So, if framework passes same seed, then we will get same mask.
                            std::minstd_rand msr;
                            if (use_seed)
                            {
                                msr = vmsr[tid];
                            }
                            else
                            {
                                msr.seed(rnd_seed + tid);
                            }
                            std::uniform_real_distribution<> gen(0, 1);

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
                                    out0[idx] = input[idx] / static_cast<T>(keep_prob);
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
