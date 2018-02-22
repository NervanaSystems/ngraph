[
    {
        "name": "Function_22",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1232",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1232_0"
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
                "name": "Parameter_1231",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1231_0"
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
                "name": "Parameter_1230",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1230_0"
                ],
                "shape": [
                    2,
                    34,
                    34,
                    16
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1229",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1229_0"
                ],
                "shape": [
                    1,
                    1,
                    16,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1228",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1228_0"
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
                "name": "Parameter_1227",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1227_0"
                ],
                "shape": [
                    1,
                    1,
                    16,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1226",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1226_0"
                ],
                "shape": [
                    3,
                    3,
                    16,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1225",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1225_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_1233",
                "op": "Constant",
                "outputs": [
                    "Constant_1233_0"
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
                    "Parameter_1231"
                ],
                "name": "Reshape_1252",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1252_0"
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
                    "Parameter_1231"
                ],
                "name": "Reshape_1242",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1242_0"
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
                    "Parameter_1230"
                ],
                "name": "Reshape_1251",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    34,
                    34
                ],
                "outputs": [
                    "Reshape_1251_0"
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
                    "Parameter_1228"
                ],
                "name": "Reshape_1237",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1237_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1227"
                ],
                "name": "Reverse_1236",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1236_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "inputs": [
                    "Parameter_1226"
                ],
                "name": "Reverse_1241",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1241_0"
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
                    "Parameter_1225"
                ],
                "name": "Broadcast_1249",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1249_0"
                ],
                "shape": [
                    3,
                    3,
                    16,
                    32
                ]
            },
            {
                "input_order": [],
                "inputs": [
                    "Parameter_1225"
                ],
                "name": "Reshape_1257",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_1257_0"
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
                    "Constant_1233"
                ],
                "name": "Broadcast_1234",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1234_0"
                ],
                "shape": [
                    2,
                    32,
                    32,
                    16
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1251",
                    "Reshape_1252"
                ],
                "name": "Convolution_1253",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1253_0"
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
                "input_order": [
                    2,
                    3,
                    0,
                    1
                ],
                "inputs": [
                    "Reverse_1236"
                ],
                "name": "Reshape_1238",
                "op": "Reshape",
                "output_shape": [
                    16,
                    32,
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_1238_0"
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
                    "Reverse_1241"
                ],
                "name": "Reshape_1243",
                "op": "Reshape",
                "output_shape": [
                    16,
                    32,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_1243_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1226",
                    "Broadcast_1249"
                ],
                "name": "Multiply_1250",
                "op": "Multiply",
                "outputs": [
                    "Multiply_1250_0"
                ]
            },
            {
                "axes": [
                    2,
                    3
                ],
                "inputs": [
                    "Reshape_1257"
                ],
                "name": "Broadcast_1258",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1258_0"
                ],
                "shape": [
                    1,
                    1,
                    16,
                    32
                ]
            },
            {
                "inputs": [
                    "Parameter_1232",
                    "Broadcast_1234"
                ],
                "name": "Greater_1235",
                "op": "Greater",
                "outputs": [
                    "Greater_1235_0"
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
                    "Convolution_1253"
                ],
                "name": "Reshape_1254",
                "op": "Reshape",
                "output_shape": [
                    32,
                    3,
                    3,
                    16
                ],
                "outputs": [
                    "Reshape_1254_0"
                ]
            },
            {
                "data_dilation_strides": [
                    2,
                    2
                ],
                "inputs": [
                    "Reshape_1237",
                    "Reshape_1238"
                ],
                "name": "Convolution_1239",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1239_0"
                ],
                "padding_above": [
                    1,
                    1
                ],
                "padding_below": [
                    0,
                    0
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
                    2,
                    2
                ],
                "inputs": [
                    "Reshape_1242",
                    "Reshape_1243"
                ],
                "name": "Convolution_1244",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1244_0"
                ],
                "padding_above": [
                    3,
                    3
                ],
                "padding_below": [
                    2,
                    2
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
                "inputs": [
                    "Parameter_1227",
                    "Broadcast_1258"
                ],
                "name": "Multiply_1259",
                "op": "Multiply",
                "outputs": [
                    "Multiply_1259_0"
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
                    "Reshape_1254"
                ],
                "name": "Reshape_1255",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_1255_0"
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
                    "Convolution_1239"
                ],
                "name": "Reshape_1240",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1240_0"
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
                    "Convolution_1244"
                ],
                "name": "Reshape_1245",
                "op": "Reshape",
                "output_shape": [
                    2,
                    34,
                    34,
                    16
                ],
                "outputs": [
                    "Reshape_1245_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_1259",
                    "Parameter_1229"
                ],
                "name": "Add_1260",
                "op": "Add",
                "outputs": [
                    "Add_1260_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_1250",
                    "Reshape_1255"
                ],
                "name": "Add_1256",
                "op": "Add",
                "outputs": [
                    "Add_1256_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_1245"
                ],
                "lower_bounds": [
                    0,
                    1,
                    1,
                    0
                ],
                "name": "Slice_1246",
                "op": "Slice",
                "outputs": [
                    "Slice_1246_0"
                ],
                "strides": [
                    1,
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    2,
                    33,
                    33,
                    16
                ]
            },
            {
                "inputs": [
                    "Reshape_1240",
                    "Slice_1246"
                ],
                "name": "Add_1247",
                "op": "Add",
                "outputs": [
                    "Add_1247_0"
                ]
            },
            {
                "inputs": [
                    "Greater_1235",
                    "Add_1247",
                    "Broadcast_1234"
                ],
                "name": "Select_1248",
                "op": "Select",
                "outputs": [
                    "Select_1248_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_1225",
            "Parameter_1226",
            "Parameter_1227",
            "Parameter_1228",
            "Parameter_1229",
            "Parameter_1230",
            "Parameter_1231",
            "Parameter_1232"
        ],
        "result": [
            "Select_1248",
            "Add_1256",
            "Add_1260"
        ]
    }
]