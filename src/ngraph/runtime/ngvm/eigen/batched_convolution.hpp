// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/eigen/utils.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            namespace eigen
            {
                template <typename ET, size_t IMG_DIMENSIONS>
                class BatchedConvolutionInstruction : public Instruction
                {
                public:
                    BatchedConvolutionInstruction(const TensorViewInfo& arg0, // imgs
                                                  const TensorViewInfo& arg1, // kernels
                                                  const TensorViewInfo& out,
                                                  size_t n_imgs,
                                                  size_t n_input_channels,
                                                  size_t n_output_channels)
                        : m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_out(out)
                        , m_n_imgs(n_imgs)
                        , m_n_input_channels(n_input_channels)
                        , m_n_output_channels(n_output_channels)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        auto imgs_in = EigenTensor<ET, IMG_DIMENSIONS + 2>(call_frame, m_arg0);
                        auto kernels = EigenTensor<ET, IMG_DIMENSIONS + 2>(call_frame, m_arg1);
                        auto imgs_out = EigenTensor<ET, IMG_DIMENSIONS + 2>(call_frame, m_out);

                        for (size_t img_idx = 0; img_idx < m_n_imgs; img_idx++)
                        {
                            auto img_in = imgs_in.chip(img_idx, 0);
                            auto img_out = imgs_out.chip(img_idx, 0);

                            for (size_t co = 0; co < m_n_output_channels; co++)
                            {
                                auto kernel = kernels.chip(co, 0);

                                // We convolve on all dimensions including the input channels, hence the +1.
                                auto conv_dims = Eigen::array<ptrdiff_t, IMG_DIMENSIONS + 1>();

                                for (size_t i = 0; i < IMG_DIMENSIONS + 1; i++)
                                {
                                    conv_dims[i] = ptrdiff_t(i);
                                }

                                img_out.chip(co, 0) = img_in.convolve(kernel, conv_dims).chip(0, 0);
                            }
                        }
                    }

                protected:
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    TensorViewInfo m_out;
                    size_t m_n_imgs;
                    size_t m_n_input_channels;
                    size_t m_n_output_channels;
                };
            }
        }
    }
}
