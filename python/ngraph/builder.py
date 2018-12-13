# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from ngraph.impl.builder import ScaledQuantize, ScaledDequantize, ScaledQuantizedConvolution
from ngraph.impl.Quantize import RoundMode
from ngraph.impl import Node
from ngraph.utils.types import NumericType


def scaledquantize(data, # type: Node
        min_val,         # type: Node
        max_val,         # type: Node
        quant_type,      # type: NumericType
        axes,            # type: List[int]
        round_mode,      # type: int
        name=None,       # type: str
        ):

    return ScaledQuantize(data,
            min_val,
            max_val,
            quant_type,
            axes,
            round_mode)

def scaleddequantize(data,  # type: Node
        min_val,            # type: Node
        max_val,            # type: Node
        real_type,          # type: NumericType
        axes,               # type: List[int]
        name=None,          # type: str
        ):
    return ScaledDequantize(data,
            min_val,
            max_val,
            real_type,
            axes)

def scaledquantizedconvolution(data,    # type: Node
        filters,                        # type: Node
        window_movement_strides=None,   # type: List[int]
        window_dilation_strides=None,   # type: List[int]
        padding_below=None,             # type: List[int]
        padding above=None,             # type: List[int]
        data_dilation_stride=None,      # type: List[int]
        min_input,                      # type: Node
        max_output,                     # type: Node
        min_filter,                     # type: Node
        max_filter,                     # type: Node
        min_freezed_output,             # type: Node
        max_freezed_output              # type: Node
        name=None,                      # type: str
        ):
    spatial_dim_count = len(data.shape) - 2
    if window_movement_strides is None:
        window_movement_strides = [1] * spatial_dim_count
    if window_dilation_strides is None:
        window_dilation_strides = [1] * spatial_dim_count
    if padding_above is None:
        padding_above = [0] * spatial_dim_count
    if padding_below is None:
        padding_below = [0] * spatial_dim_count
    if data_dilation_strides is None:
        data_dilation_strides = [1] * spatial_dim_count
    return ScaledQuantizedConvolution(data,
            filters,
            window_movement_strides,
            window_dilation_strides,
            padding_below,
            padding above,
            data_dilation_stride,
            min_input,
            max_output,
            min_filter,
            max_filter,
            min_freezed_output,
            max_freezed_output)
