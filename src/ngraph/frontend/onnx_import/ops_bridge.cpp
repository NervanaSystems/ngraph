//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "core/attribute.hpp"
#include "op/abs.hpp"
#include "op/add.hpp"
#include "op/and.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/cast.hpp"
#include "op/ceil.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/div.hpp"
#include "op/elu.hpp"
#include "op/equal.hpp"
#include "op/exp.hpp"
#include "op/flatten.hpp"
#include "op/floor.hpp"
#include "op/gemm.hpp"
#include "op/global_average_pool.hpp"
#include "op/global_max_pool.hpp"
#include "op/greater.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/identity.hpp"
#include "op/leaky_relu.hpp"
#include "op/less.hpp"
#include "op/log.hpp"
#include "op/log_softmax.hpp"
#include "op/lrn.hpp"
#include "op/matmul.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/mean.hpp"
#include "op/min.hpp"
#include "op/mul.hpp"
#include "op/neg.hpp"
#include "op/not.hpp"
#include "op/or.hpp"
#include "op/pow.hpp"
#include "op/prelu.hpp"
#include "op/reciprocal.hpp"
#include "op/reduce.hpp"
#include "op/relu.hpp"
#include "op/reshape.hpp"
#include "op/selu.hpp"
#include "op/shape.hpp"
#include "op/sigmoid.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/softplus.hpp"
#include "op/softsign.hpp"
#include "op/split.hpp"
#include "op/sqrt.hpp"
#include "op/squeeze.hpp"
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/tanh.hpp"
#include "op/thresholded_relu.hpp"
#include "op/transpose.hpp"
#include "op/unsqueeze.hpp"
#include "op/xor.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        const OperatorSet& OperatorsBridge::get_operator_set_version_1() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                for (const auto& op : m_map)
                {
                    for (const auto& it : op.second)
                    {
                        if (it.first == 1)
                        {
                            operator_set.emplace(op.first, it.second);
                        }
                    }
                }
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_2() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_1();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_3() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_2();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_4() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_3();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_5() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_4();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_6() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_5();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_7() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_6();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_8() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_7();
            }
            return operator_set;
        }

        const OperatorSet& OperatorsBridge::get_operator_set_version_9() const
        {
            static OperatorSet operator_set;
            if (operator_set.empty())
            {
                operator_set = get_operator_set_version_8();
            }
            return operator_set;
        }

#define OPERATOR_SET_NAME(version_) get_operator_set_version_##version_()

#define GET_OPERATOR_SET(version_)                                                                 \
    case version_:                                                                                 \
        return OPERATOR_SET_NAME(version_)

#define OPERATOR_SET_NAME_HELPER(version_) OPERATOR_SET_NAME(version_)

#define DEFAULT_OPERATOR_SET() return OPERATOR_SET_NAME_HELPER(ONNX_OPSET_VERSION)

        const OperatorSet& OperatorsBridge::get_operator_set_version(std::int64_t version) const
        {
            switch (version)
            {
                GET_OPERATOR_SET(1);
                GET_OPERATOR_SET(2);
                GET_OPERATOR_SET(3);
                GET_OPERATOR_SET(4);
                GET_OPERATOR_SET(5);
                GET_OPERATOR_SET(6);
                GET_OPERATOR_SET(7);
                GET_OPERATOR_SET(8);
                GET_OPERATOR_SET(9);
            default: DEFAULT_OPERATOR_SET();
            }
        }

#define REGISTER_OPERATOR(name_, version_, fn_)                                                    \
    m_map[name_].emplace(version_, std::bind(op::set_##version_::fn_, std::placeholders::_1))

        OperatorsBridge::OperatorsBridge()
        {
            REGISTER_OPERATOR("Abs", 1, abs);
            REGISTER_OPERATOR("Add", 1, add);
            REGISTER_OPERATOR("And", 1, logical_and);
            REGISTER_OPERATOR("AveragePool", 1, average_pool);
            REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
            REGISTER_OPERATOR("Cast", 1, cast);
            REGISTER_OPERATOR("Ceil", 1, ceil);
            REGISTER_OPERATOR("Clip", 1, clip);
            REGISTER_OPERATOR("Concat", 1, concat);
            REGISTER_OPERATOR("Constant", 1, constant);
            REGISTER_OPERATOR("Conv", 1, conv);
            REGISTER_OPERATOR("Div", 1, div);
            REGISTER_OPERATOR("Dropout", 1, identity);
            REGISTER_OPERATOR("Elu", 1, elu);
            REGISTER_OPERATOR("Equal", 1, equal);
            REGISTER_OPERATOR("Exp", 1, exp);
            REGISTER_OPERATOR("Flatten", 1, flatten);
            REGISTER_OPERATOR("Floor", 1, floor);
            REGISTER_OPERATOR("Gemm", 1, gemm);
            REGISTER_OPERATOR("GlobalAveragePool", 1, global_average_pool);
            REGISTER_OPERATOR("GlobalMaxPool", 1, global_max_pool);
            REGISTER_OPERATOR("Greater", 1, greater);
            REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
            REGISTER_OPERATOR("Identity", 1, identity);
            REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
            REGISTER_OPERATOR("Less", 1, less);
            REGISTER_OPERATOR("Log", 1, log);
            REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
            REGISTER_OPERATOR("LRN", 1, lrn);
            REGISTER_OPERATOR("MatMul", 1, matmul);
            REGISTER_OPERATOR("MaxPool", 1, max_pool);
            REGISTER_OPERATOR("Max", 1, max);
            REGISTER_OPERATOR("Mean", 1, mean);
            REGISTER_OPERATOR("Min", 1, min);
            REGISTER_OPERATOR("Mul", 1, mul);
            REGISTER_OPERATOR("Neg", 1, neg);
            REGISTER_OPERATOR("Not", 1, logical_not);
            REGISTER_OPERATOR("Or", 1, logical_or);
            REGISTER_OPERATOR("Pow", 1, pow);
            REGISTER_OPERATOR("PRelu", 1, prelu);
            REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
            REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
            REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
            REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
            REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
            REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
            REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
            REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
            REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
            REGISTER_OPERATOR("ReduceSum", 1, reduce_sum);
            REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
            REGISTER_OPERATOR("Relu", 1, relu);
            REGISTER_OPERATOR("Reshape", 1, reshape);
            REGISTER_OPERATOR("Selu", 1, selu);
            REGISTER_OPERATOR("Shape", 1, shape);
            REGISTER_OPERATOR("Sigmoid", 1, sigmoid);
            REGISTER_OPERATOR("Slice", 1, slice);
            REGISTER_OPERATOR("Softmax", 1, softmax);
            REGISTER_OPERATOR("Softplus", 1, softplus);
            REGISTER_OPERATOR("Softsign", 1, softsign);
            REGISTER_OPERATOR("Split", 1, split);
            REGISTER_OPERATOR("Sqrt", 1, sqrt);
            REGISTER_OPERATOR("Squeeze", 1, squeeze);
            REGISTER_OPERATOR("Sub", 1, sub);
            REGISTER_OPERATOR("Sum", 1, sum);
            REGISTER_OPERATOR("Tanh", 1, tanh);
            REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
            REGISTER_OPERATOR("Transpose", 1, transpose);
            REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
            REGISTER_OPERATOR("Xor", 1, logical_xor);
        }

    } // namespace onnx_import

} // namespace ngraph
