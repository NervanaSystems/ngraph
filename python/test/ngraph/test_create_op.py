# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type
from _pyngraph import PartialShape
import test

np_types = [np.float32, np.int32]
integral_np_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@pytest.mark.parametrize("dtype", np_types)
def test_binary_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    mode = "xnor-popcount"
    pad_value = 0.0

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.binary_convolution(
        parameter_input0,
        parameter_input1,
        strides,
        pads_begin,
        pads_end,
        dilations,
        mode,
        pad_value,
    )

    assert node.get_type_name() == "BinaryConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_ctc_greedy_decoder(dtype):
    input0_shape = [20, 8, 128]
    input1_shape = [20, 8]
    expected_shape = [8, 20, 1, 1]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.ctc_greedy_decoder(parameter_input0, parameter_input1)

    assert node.get_type_name() == "CTCGreedyDecoder"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 1, 9, 9]
    input2_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)

    node = ng.deformable_convolution(
        parameter_input0,
        parameter_input1,
        parameter_input2,
        strides,
        pads_begin,
        pads_end,
        dilations,
    )

    assert node.get_type_name() == "DeformableConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_psroi_pooling(dtype):
    output_dim = 8
    spatial_scale = 0.0625
    group_size = 7
    mode = "bilinear_deformable"
    spatial_bins_x = 4
    spatial_bins_y = 4
    trans_std = 0.1
    part_size = 7

    input0_shape = [1, 392, 38, 63]
    input1_shape = [300, 5]
    input2_shape = [300, 2, 7, 7]
    expected_shape = [300, 8, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)

    node = ng.deformable_psroi_pooling(
        parameter_input0,
        parameter_input1,
        output_dim,
        spatial_scale,
        group_size,
        mode,
        spatial_bins_x,
        spatial_bins_y,
        trans_std,
        part_size,
        offsets=parameter_input2,
    )

    assert node.get_type_name() == "DeformablePSROIPooling"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_floor_mod(dtype):
    input0_shape = [8, 1, 6, 1]
    input1_shape = [7, 1, 5]
    expected_shape = [8, 7, 6, 5]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.floor_mod(parameter_input0, parameter_input1)

    assert node.get_type_name() == "FloorMod"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_gather_tree(dtype):
    input0_shape = [100, 1, 10]
    input1_shape = [100, 1, 10]
    input2_shape = [1]
    input3_shape = []
    expected_shape = [100, 1, 10]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)
    parameter_input3 = ng.parameter(input3_shape, name="Input3", dtype=dtype)

    node = ng.gather_tree(parameter_input0, parameter_input1, parameter_input2, parameter_input3)

    assert node.get_type_name() == "GatherTree"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_cell_operator(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    C_t_shape = [batch_size, hidden_size]
    W_shape = [4 * hidden_size, input_size]
    R_shape = [4 * hidden_size, hidden_size]
    B_shape = [4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    expected_shape = [1, 128]

    node_default = ng.lstm_cell(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
    )

    assert node_default.get_type_name() == "LSTMCell"
    assert node_default.get_output_size() == 2
    assert list(node_default.get_output_shape(0)) == expected_shape
    assert list(node_default.get_output_shape(1)) == expected_shape

    activations = ["tanh", "Sigmoid", "RELU"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 0.5

    node_param = ng.lstm_cell(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMCell"
    assert node_param.get_output_size() == 2
    assert list(node_param.get_output_shape(0)) == expected_shape
    assert list(node_param.get_output_shape(1)) == expected_shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_bidirectional(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128
    num_directions = 2
    seq_length = 2

    X_shape = [seq_length, batch_size, input_size]
    H_t_shape = [num_directions, batch_size, hidden_size]
    C_t_shape = [num_directions, batch_size, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "BIDIRECTIONAL"
    node = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_reverse(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [seq_length, batch_size, input_size]
    H_t_shape = [num_directions, batch_size, hidden_size]
    C_t_shape = [num_directions, batch_size, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "REVERSE"

    node_default = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_forward(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [seq_length, batch_size, input_size]
    H_t_shape = [num_directions, batch_size, hidden_size]
    C_t_shape = [num_directions, batch_size, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "forward"

    node_default = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [2.0]
    activation_beta = [1.0]
    clip = 0.5

    node = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3


def test_gru_cell_operator():
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    W_shape = [3 * hidden_size, input_size]
    R_shape = [3 * hidden_size, hidden_size]
    B_shape = [3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    expected_shape = [1, 128]

    node_default = ng.gru_cell(
        parameter_X, parameter_H_t, parameter_W, parameter_R, parameter_B, hidden_size
    )

    assert node_default.get_type_name() == "GRUCell"
    assert node_default.get_output_size() == 1
    assert list(node_default.get_output_shape(0)) == expected_shape

    activations = ["tanh", "relu"]
    activations_alpha = [1.0, 2.0]
    activations_beta = [1.0, 2.0]
    clip = 0.5
    linear_before_reset = True

    # If *linear_before_reset* is set True, then B tensor shape must be [4 * hidden_size]
    B_shape = [4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    node_param = ng.gru_cell(
        parameter_X,
        parameter_H_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activations_alpha,
        activations_beta,
        clip,
        linear_before_reset,
    )

    assert node_param.get_type_name() == "GRUCell"
    assert node_param.get_output_size() == 1
    assert list(node_param.get_output_shape(0)) == expected_shape


def test_roi_pooling():
    inputs = ng.parameter([2, 3, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.roi_pooling(inputs, coords, [6, 6], 0.0625, "Max")

    assert node.get_type_name() == "ROIPooling"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [150, 3, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_psroi_pooling():
    inputs = ng.parameter([1, 3, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.psroi_pooling(inputs, coords, 2, 6, 0.0625, 0, 0, "Avg")

    assert node.get_type_name() == "PSROIPooling"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [150, 2, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_convert_like():
    parameter_data = ng.parameter([1, 2, 3, 4], name="data", dtype=np.float32)
    like = ng.constant(1, dtype=np.int8)

    node = ng.convert_like(parameter_data, like)

    assert node.get_type_name() == "ConvertLike"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2, 3, 4]
    assert node.get_output_element_type(0) == Type.i8


def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1
    excepted = [[5, 10], [10, 5], [10, 10]]

    result = test.ngraph.util.run_op_node([data, depth, on_value, off_value], ng.ops.one_hot, axis)
    assert np.allclose(result, excepted)


def test_reverse():
    parameter_data = ng.parameter([3, 10, 100, 200], name="data", dtype=np.float32)
    parameter_axis = ng.parameter([1], name="axis", dtype=np.int64)
    expected_shape = [3, 10, 100, 200]

    node = ng.reverse(parameter_data, parameter_axis, "index")

    assert node.get_type_name() == "Reverse"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_select():
    cond = [[False, False], [True, False], [True, True]]
    then_node = [[-1, 0], [1, 2], [3, 4]]
    else_node = [[11, 10], [9, 8], [7, 6]]
    excepted = [[11, 10], [1, 8], [3, 4]]

    result = test.ngraph.util.run_op_node([cond, then_node, else_node], ng.ops.select)
    assert np.allclose(result, excepted)


def test_result():
    node = [[11, 10], [1, 8], [3, 4]]

    result = test.ngraph.util.run_op_node([node], ng.ops.result)
    assert np.allclose(result, node)


def test_bucketize():
    data = ng.parameter([4, 3, 2, 1], name="data", dtype=np.float32)
    buckets = ng.parameter([5], name="buckets", dtype=np.int64)

    node = ng.bucketize(data, buckets, "i32")

    assert node.get_type_name() == "Bucketize"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [4, 3, 2, 1]
    assert node.get_output_element_type(0) == Type.i32


@pytest.mark.skip_on_interpreter
def test_range():
    start = 5
    stop = 35
    step = 5

    result = test.ngraph.util.run_op_node([start, stop, step], ng.ops.range)
    assert np.allclose(result, [5, 10, 15, 20, 25, 30])


def test_region_yolo():
    data = ng.parameter([1, 125, 13, 13], name="input", dtype=np.float32)
    num_coords = 4
    num_classes = 80
    num_regions = 1
    mask = [6, 7, 8]
    axis = 0
    end_axis = 3
    do_softmax = False

    node = ng.region_yolo(data, num_coords, num_classes, num_regions,
                          do_softmax, mask, axis, end_axis)

    assert node.get_type_name() == "RegionYolo"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, (80 + 4 + 1) * 3, 13, 13]
    assert node.get_output_element_type(0) == Type.f32


def test_reorg_yolo():
    data = ng.parameter([2, 24, 34, 62], name="input", dtype=np.int32)
    stride = [2]

    node = ng.reorg_yolo(data, stride)

    assert node.get_type_name() == "ReorgYolo"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 96, 17, 31]
    assert node.get_output_element_type(0) == Type.i32


def test_embedding_bag_offsets_sum_1():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    offsets = ng.parameter([3], name="offsets", dtype=np.int64)
    default_index = ng.parameter([], name="default_index", dtype=np.int64)

    node = ng.embedding_bag_offsets_sum(emb_table, indices, offsets, default_index)

    assert node.get_type_name() == "EmbeddingBagOffsetsSum"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_segments_sum_all_inputs():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    segment_ids = ng.parameter([4], name="segment_ids", dtype=np.int64)
    num_segments = ng.parameter([], name="num_segments", dtype=np.int64)
    default_index = ng.parameter([], name="default_index", dtype=np.int64)
    per_sample_weights = ng.parameter([4], name="per_sample_weights", dtype=np.float32)

    node = ng.embedding_segments_sum(
        emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights
    )

    assert node.get_type_name() == "EmbeddingSegmentsSum"
    assert node.get_output_size() == 1
    assert node.get_output_partial_shape(0).same_scheme(PartialShape([-1, 2]))
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_segments_sum_with_some_opt_inputs():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    segment_ids = ng.parameter([4], name="segment_ids", dtype=np.int64)
    num_segments = ng.parameter([], name="num_segments", dtype=np.int64)

    # only 1 out of 3 optional inputs
    node = ng.embedding_segments_sum(emb_table, indices, segment_ids, num_segments)

    assert node.get_type_name() == "EmbeddingSegmentsSum"
    assert node.get_output_size() == 1
    assert node.get_output_partial_shape(0).same_scheme(PartialShape([-1, 2]))
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_bag_packed_sum():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([3, 3], name="indices", dtype=np.int64)
    per_sample_weights = ng.parameter([3, 3], name="per_sample_weights", dtype=np.float32)

    # only 1 out of 3 optional inputs
    node = ng.embedding_bag_packed_sum(emb_table, indices, per_sample_weights)

    assert node.get_type_name() == "EmbeddingBagPackedSum"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize("dtype", integral_np_types)
def test_interpolate(dtype):
    image_shape = [1, 3, 1024, 1024]
    output_shape = [64, 64]
    attributes = {
        "attrs.axes": [2, 3],
        "attrs.mode": "cubic",
        "attrs.pads_begin": np.array([2, 2], dtype=dtype),
    }

    image_node = ng.parameter(image_shape, dtype, name="Image")

    node = ng.interpolate(image_node, output_shape, attributes)
    expected_shape = [1, 3, 64, 64]

    assert node.get_type_name() == "Interpolate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_prior_box(int_dtype, fp_dtype):
    image_shape = np.array([64, 64], dtype=int_dtype)
    attributes = {
        "attrs.offset": fp_dtype(0),
        "attrs.min_size": np.array([2, 3], dtype=fp_dtype),
        "attrs.aspect_ratio": np.array([1.5, 2.0, 2.5], dtype=fp_dtype),
    }

    layer_shape = ng.constant(np.array([32, 32], dtype=int_dtype), int_dtype)

    node = ng.prior_box(layer_shape, image_shape, attributes)

    assert node.get_type_name() == "PriorBox"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 20480]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_prior_box_clustered(int_dtype, fp_dtype):
    image_size = np.array([64, 64], dtype=int_dtype)
    attributes = {
        "attrs.offset": fp_dtype(0.5),
        "attrs.widths": np.array([4.0, 2.0, 3.2], dtype=fp_dtype),
        "attrs.heights": np.array([1.0, 2.0, 1.0], dtype=fp_dtype),
    }

    output_size = ng.constant(np.array([19, 19], dtype=int_dtype), int_dtype)

    node = ng.prior_box_clustered(output_size, image_size, attributes)

    assert node.get_type_name() == "PriorBoxClustered"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 4332]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_detection_output(int_dtype, fp_dtype):
    attributes = {
        "attrs.num_classes": int_dtype(85),
        "attrs.keep_top_k": np.array([64], dtype=int_dtype),
        "attrs.nms_threshold": fp_dtype(0.645),
    }

    box_logits = ng.parameter([4, 1, 5, 5], fp_dtype, "box_logits")
    class_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "class_preds")
    proposals = ng.parameter([2, 1, 4, 5], fp_dtype, "proposals")
    aux_class_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "aux_class_preds")
    aux_box_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "aux_box_preds")

    node = ng.detection_output(
        box_logits, class_preds, proposals, attributes, aux_class_preds, aux_box_preds
    )

    assert node.get_type_name() == "DetectionOutput"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 256, 7]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.uint32, np.float16),
        (np.uint32, np.float64),
    ],
)
def test_proposal(int_dtype, fp_dtype):
    attributes = {
        "attrs.base_size": int_dtype(1),
        "attrs.pre_nms_topn": int_dtype(20),
        "attrs.post_nms_topn": int_dtype(64),
        "attrs.nms_thresh": fp_dtype(0.34),
        "attrs.feat_stride": int_dtype(16),
        "attrs.min_size": int_dtype(32),
        "attrs.ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=fp_dtype),
        "attrs.scale": np.array([2, 3, 3, 4], dtype=fp_dtype),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], fp_dtype, "class_probs")
    class_logits = ng.parameter([batch_size, 24, 34, 62], fp_dtype, "class_logits")
    image_shape = ng.parameter([3], fp_dtype, "image_shape")
    node = ng.proposal(class_probs, class_logits, image_shape, attributes)

    assert node.get_type_name() == "Proposal"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [batch_size * attributes["attrs.post_nms_topn"], 5]
