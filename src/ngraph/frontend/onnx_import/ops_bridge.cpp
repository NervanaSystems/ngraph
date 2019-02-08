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

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "core/attribute.hpp"
#include "ngraph/log.hpp"
#include "op/abs.hpp"
#include "op/acos.hpp"
#include "op/add.hpp"
#include "op/and.hpp"
#include "op/argmax.hpp"
#include "op/argmin.hpp"
#include "op/asin.hpp"
#include "op/atan.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/cast.hpp"
#include "op/ceil.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/conv_transpose.hpp"
#include "op/cos.hpp"
#include "op/cosh.hpp"
#include "op/depth_to_space.hpp"
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
#include "op/lstm.hpp"
#include "op/matmul.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/mean.hpp"
#include "op/min.hpp"
#include "op/mul.hpp"
#include "op/neg.hpp"
#include "op/not.hpp"
#include "op/or.hpp"
#include "op/pad.cpp"
#include "op/pad.hpp"
#include "op/pow.hpp"
#include "op/prelu.hpp"
#include "op/reciprocal.hpp"
#include "op/reduce.hpp"
#include "op/relu.hpp"
#include "op/reshape.hpp"
#include "op/selu.hpp"
#include "op/shape.hpp"
#include "op/sigmoid.hpp"
#include "op/sin.hpp"
#include "op/sinh.hpp"
#include "op/size.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/softplus.hpp"
#include "op/softsign.hpp"
#include "op/space_to_depth.hpp"
#include "op/split.hpp"
#include "op/sqrt.hpp"
#include "op/squeeze.hpp"
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/tan.hpp"
#include "op/tanh.hpp"
#include "op/thresholded_relu.hpp"
#include "op/topk.hpp"
#include "op/transpose.hpp"
#include "op/unsqueeze.hpp"
#include "op/xor.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            const std::map<std::int64_t, Operator>::const_iterator
                find(std::int64_t version, const std::map<std::int64_t, Operator>& map)
            {
                std::map<std::int64_t, Operator>::const_iterator it{};
                // Get the latest version.
                if (version == -1)
                {
                    return map.empty() ? std::end(map) : --std::end(map);
                }
                while (version > 0)
                {
                    it = map.find(version--);
                    if (it != std::end(map))
                    {
                        return it;
                    }
                }
                return it;
            }
        }

        void OperatorsBridge::_register_operator(const std::string& name,
                                                 std::int64_t version,
                                                 const std::string& domain,
                                                 Operator fn)
        {
            auto result = m_map[domain][name].emplace(version, std::move(fn));
            if (result.second)
            {
                NGRAPH_WARN << "Overwriting existing operator: "
                            << domain + "." + name + ":" + std::to_string(version);
            }
        }

        OperatorSet OperatorsBridge::_get_operator_set(const std::string& domain,
                                                       std::int64_t version)
        {
            OperatorSet result;

            auto dm = m_map.find(domain);
            if (dm == std::end(m_map))
            {
                throw error::UnknownDomain{domain};
            }
            if (domain == "" && version > OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION)
            {
                NGRAPH_WARN << "Currently ONNX operator set version: " << version
                            << " is unsupported. Falling back to: "
                            << OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION;
            }
            for (const auto& op : dm->second)
            {
                const auto& it = detail::find(version, op.second);
                if (it == std::end(op.second))
                {
                    throw error::UnsupportedVersion{op.first, version, domain};
                }
                result.emplace(op.first, it->second);
            }
            return result;
        }

        bool OperatorsBridge::_is_operator_registered(const std::string& name,
                                                      std::int64_t version,
                                                      const std::string& domain)
        {
            // search for domain
            auto dm_map = m_map.find(domain);
            if (dm_map == std::end(m_map))
            {
                return false;
            }
            // search for name
            auto op_map = dm_map->second.find(name);
            if (op_map == std::end(dm_map->second))
            {
                return false;
            }

            if (detail::find(version, op_map->second) != std::end(op_map->second))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

#define REGISTER_OPERATOR(name_, ver_, fn_)                                                        \
    m_map[""][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1))

        OperatorsBridge::OperatorsBridge()
        {
            REGISTER_OPERATOR("Abs", 1, abs);
            REGISTER_OPERATOR("Acos", 1, acos);
            REGISTER_OPERATOR("Add", 1, add);
            REGISTER_OPERATOR("Add", 7, add);
            REGISTER_OPERATOR("And", 1, logical_and);
            REGISTER_OPERATOR("ArgMin", 1, argmin);
            REGISTER_OPERATOR("ArgMax", 1, argmax);
            REGISTER_OPERATOR("Asin", 1, asin);
            REGISTER_OPERATOR("Atan", 1, atan);
            REGISTER_OPERATOR("AveragePool", 1, average_pool);
            REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
            REGISTER_OPERATOR("Cast", 1, cast);
            REGISTER_OPERATOR("Ceil", 1, ceil);
            REGISTER_OPERATOR("Clip", 1, clip);
            REGISTER_OPERATOR("Concat", 1, concat);
            REGISTER_OPERATOR("Constant", 1, constant);
            REGISTER_OPERATOR("Conv", 1, conv);
            REGISTER_OPERATOR("ConvTranspose", 1, conv_transpose);
            REGISTER_OPERATOR("Cos", 1, cos);
            REGISTER_OPERATOR("Cosh", 1, cosh);
            REGISTER_OPERATOR("DepthToSpace", 1, depth_to_space);
            REGISTER_OPERATOR("Div", 1, div);
            REGISTER_OPERATOR("Div", 7, div);
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
            REGISTER_OPERATOR("LSTM", 1, lstm);
            REGISTER_OPERATOR("MatMul", 1, matmul);
            REGISTER_OPERATOR("MaxPool", 1, max_pool);
            REGISTER_OPERATOR("Max", 1, max);
            REGISTER_OPERATOR("Max", 8, max);
            REGISTER_OPERATOR("Mean", 1, mean);
            REGISTER_OPERATOR("Mean", 8, mean);
            REGISTER_OPERATOR("Min", 1, min);
            REGISTER_OPERATOR("Min", 8, min);
            REGISTER_OPERATOR("Mul", 1, mul);
            REGISTER_OPERATOR("Mul", 7, mul);
            REGISTER_OPERATOR("Neg", 1, neg);
            REGISTER_OPERATOR("Not", 1, logical_not);
            REGISTER_OPERATOR("Or", 1, logical_or);
            REGISTER_OPERATOR("Pad", 1, pad);
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
            REGISTER_OPERATOR("Sin", 1, sin);
            REGISTER_OPERATOR("Sinh", 1, sinh);
            REGISTER_OPERATOR("Size", 1, size);
            REGISTER_OPERATOR("Slice", 1, slice);
            REGISTER_OPERATOR("Softmax", 1, softmax);
            REGISTER_OPERATOR("Softplus", 1, softplus);
            REGISTER_OPERATOR("Softsign", 1, softsign);
            REGISTER_OPERATOR("SpaceToDepth", 1, space_to_depth);
            REGISTER_OPERATOR("Split", 1, split);
            REGISTER_OPERATOR("Sqrt", 1, sqrt);
            REGISTER_OPERATOR("Squeeze", 1, squeeze);
            REGISTER_OPERATOR("Sub", 1, sub);
            REGISTER_OPERATOR("Sub", 7, sub);
            REGISTER_OPERATOR("Sum", 1, sum);
            REGISTER_OPERATOR("Sum", 8, sum);
            REGISTER_OPERATOR("Tan", 1, tan);
            REGISTER_OPERATOR("Tanh", 1, tanh);
            REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
            REGISTER_OPERATOR("TopK", 1, topk);
            REGISTER_OPERATOR("Transpose", 1, transpose);
            REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
            REGISTER_OPERATOR("Xor", 1, logical_xor);
        }

    } // namespace onnx_import

} // namespace ngraph
