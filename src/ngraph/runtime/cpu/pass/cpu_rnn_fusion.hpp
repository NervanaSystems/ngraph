/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class LSTMFusion;
                class RNNFusion;
                class MultiLayerRNNFusion;
            }
        }
    }
}

class ngraph::runtime::cpu::pass::LSTMFusion : public ngraph::pass::GraphRewrite
{
public:
    LSTMFusion()
        : GraphRewrite()
    {
        construct_sigmoid();
        construct_lstm_fprop();
    }

private:
    void construct_sigmoid();
    void construct_lstm_fprop();
};

class ngraph::runtime::cpu::pass::RNNFusion : public ngraph::pass::RecurrentGraphRewrite
{
public:
    RNNFusion()
        : RecurrentGraphRewrite()
    {
        construct_rnn_lstm_fprop();
    }

private:
    void construct_rnn_lstm_fprop();
};

class ngraph::runtime::cpu::pass::MultiLayerRNNFusion : public ngraph::pass::RecurrentGraphRewrite
{
public:
    MultiLayerRNNFusion()
        : RecurrentGraphRewrite()
    {
        construct_multi_layer_rnn_fusion_fprop();
    }

private:
    void construct_multi_layer_rnn_fusion_fprop();
};
