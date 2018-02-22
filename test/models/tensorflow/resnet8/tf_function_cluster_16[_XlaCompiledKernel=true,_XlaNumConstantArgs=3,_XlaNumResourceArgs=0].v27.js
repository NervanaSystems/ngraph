[
    {
        "name": "Function_19",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1052",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1052_0"
                ],
                "shape": [
                    2,
                    16,
                    16,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1051",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1051_0"
                ],
                "shape": [
                    2,
                    32,
                    32,
                    16
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1050",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1050_0"
                ],
                "shape": [
                    2,
                    16,
                    16,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1048",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1048_0"
                ],
                "shape": [
                    3,
                    3,
                    32,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1045",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1045_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_1076",
                "op": "Constant",
                "outputs": [
                    "Constant_1076_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "input_order": [
                    3,
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_1052"
                ],
                "name": "Reshape_1099",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1099_0"
                ]
            },
            {
                "input_order": [
                    3,
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_1051"
                ],
                "name": "Reshape_1060",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1060_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2,
                    3
                ],
                "inputs": [
                    "Parameter_1050"
                ],
                "name": "Reshape_1055",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_1055_0"
                ]
            },
            {
                "input_order": [
                    0,
                    1,
                    2,
                    3
                ],
                "inputs": [
                    "Parameter_1050"
                ],
                "name": "Reshape_1090",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_1090_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1048"
                ],
                "name": "Reverse_1091",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1091_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2,
                    3
                ],
                "inputs": [
                    "Parameter_1045"
                ],
                "name": "Broadcast_1097",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1097_0"
                ],
                "shape": [
                    3,
                    3,
                    32,
                    32
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2,
                    3
                ],
                "inputs": [
                    "Constant_1076"
                ],
                "name": "Broadcast_1085",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1085_0"
                ],
                "shape": [
                    2,
                    16,
                    16,
                    32
                ]
            },
            {
                "input_order": [
                    3,
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Reshape_1055"
                ],
                "name": "Reshape_1062",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1062_0"
                ]
            },
            {
                "input_order": [
                    3,
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Reshape_1090"
                ],
                "name": "Reshape_1100",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1100_0"
                ]
            },
            {
                "input_order": [
                    0,
                    3,
                    1,
                    2
                ],
                "inputs": [
                    "Reshape_1090"
                ],
                "name": "Reshape_1092",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1092_0"
                ]
            },
            {
                "input_order": [
                    2,
                    3,
                    0,
                    1
                ],
                "inputs": [
                    "Reverse_1091"
                ],
                "name": "Reshape_1093",
                "op": "Reshape",
                "output_shape": [
                    32,
                    32,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_1093_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1048",
                    "Broadcast_1097"
                ],
                "name": "Multiply_1098",
                "op": "Multiply",
                "outputs": [
                    "Multiply_1098_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1052",
                    "Broadcast_1085"
                ],
                "name": "Greater_1089",
                "op": "Greater",
                "outputs": [
                    "Greater_1089_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1060",
                    "Reshape_1062"
                ],
                "name": "Convolution_1064",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1064_0"
                ],
                "padding_above": [
                    -1,
                    -1
                ],
                "padding_below": [
                    0,
                    0
                ],
                "window_dilation_strides": [
                    2,
                    2
                ],
                "window_movement_strides": [
                    1,
                    1
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1099",
                    "Reshape_1100"
                ],
                "name": "Convolution_1101",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1101_0"
                ],
                "padding_above": [
                    1,
                    1
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_dilation_strides": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    1,
                    1
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1092",
                    "Reshape_1093"
                ],
                "name": "Convolution_1094",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1094_0"
                ],
                "padding_above": [
                    1,
                    1
                ],
                "padding_below": [
                    1,
                    1
                ],
                "window_dilation_strides": [
                    1,
                    1
                ],
                "window_movement_strides": [
                    1,
                    1
                ]
            },
            {
                "input_order": [
                    1,
                    2,
                    3,
                    0
                ],
                "inputs": [
                    "Convolution_1064"
                ],
                "name": "Reshape_1067",
                "op": "Reshape",
                "output_shape": [
                    32,
                    1,
                    1,
                    16
                ],
                "outputs": [
                    "Reshape_1067_0"
                ]
            },
            {
                "input_order": [
                    1,
                    2,
                    3,
                    0
                ],
                "inputs": [
                    "Convolution_1101"
                ],
                "name": "Reshape_1102",
                "op": "Reshape",
                "output_shape": [
                    32,
                    3,
                    3,
                    32
                ],
                "outputs": [
                    "Reshape_1102_0"
                ]
            },
            {
                "input_order": [
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_1094"
                ],
                "name": "Reshape_1095",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_1095_0"
                ]
            },
            {
                "input_order": [
                    1,
                    2,
                    3,
                    0
                ],
                "inputs": [
                    "Reshape_1067"
                ],
                "name": "Reshape_1072",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_1072_0"
                ]
            },
            {
                "input_order": [
                    1,
                    2,
                    3,
                    0
                ],
                "inputs": [
                    "Reshape_1102"
                ],
                "name": "Reshape_1103",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1103_0"
                ]
            },
            {
                "inputs": [
                    "Greater_1089",
                    "Reshape_1095",
                    "Broadcast_1085"
                ],
                "name": "Select_1096",
                "op": "Select",
                "outputs": [
                    "Select_1096_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_1098",
                    "Reshape_1103"
                ],
                "name": "Add_1104",
                "op": "Add",
                "outputs": [
                    "Add_1104_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_1045",
            "Parameter_1048",
            "Parameter_1050",
            "Parameter_1051",
            "Parameter_1052"
        ],
        "result": [
            "Reshape_1055",
            "Reshape_1072",
            "Select_1096",
            "Add_1104"
        ]
    }
]