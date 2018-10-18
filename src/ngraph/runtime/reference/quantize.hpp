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

#pragma once

#include <cfenv>
#include <cmath>

#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/quantize.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename REAL, typename QUANT>
            void quantize(const REAL* input,
                          const REAL* scale,
                          const QUANT* offset,
                          QUANT* output,
                          const Shape& input_shape,
                          const Shape& scale_offset_shape,
                          const AxisSet& axes,
                          op::Quantize::RoundMode round_mode)
            {
                CoordinateTransform input_transform(input_shape);
                CoordinateTransform scale_offset_transform(scale_offset_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate scale_offset_coord = project(input_coord, axes);

                    // apply scale
                    REAL qvalue = input[input_transform.index(input_coord)] /
                                  scale[scale_offset_transform.index(scale_offset_coord)];

                    auto fe_round_mode = std::fegetround();

                    // round
                    if (round_mode == op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO)
                    {
                        qvalue = std::round(qvalue);
                    }
                    else if (round_mode == op::Quantize::RoundMode::HALF_TO_EVEN)
                    {
                        std::fesetround(FE_TONEAREST);
                        qvalue = std::nearbyint(qvalue);
                    }
                    else if (round_mode == op::Quantize::RoundMode::ALL_TOWARD_POSITIVE_INFINITY)
                    {
                        std::fesetround(FE_UPWARD);
                        qvalue = std::nearbyint(qvalue);
                    }
                    else if (round_mode == op::Quantize::RoundMode::ALL_TOWARD_NEGATIVE_INFINITY)
                    {
                        std::fesetround(FE_DOWNWARD);
                        qvalue = std::nearbyint(qvalue);
                    }
                    else if (round_mode == op::Quantize::RoundMode::ALL_TOWARD_ZERO)
                    {
                        std::fesetround(FE_TOWARDZERO);
                        qvalue = std::nearbyint(qvalue);
                    }

                    std::fesetround(fe_round_mode);

                    // apply offset
                    qvalue += offset[scale_offset_transform.index(scale_offset_coord)];

                    // clamp
                    qvalue = std::max<REAL>(qvalue,
                                            static_cast<REAL>(std::numeric_limits<QUANT>::min()));
                    qvalue = std::min<REAL>(qvalue,
                                            static_cast<REAL>(std::numeric_limits<QUANT>::max()));

                    // cast
                    output[input_transform.index(input_coord)] = static_cast<QUANT>(qvalue);
                }
            }
        }
    }
}
