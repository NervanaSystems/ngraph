[
    {
        "name": "Function_4",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_178",
                "op": "Parameter",
                "outputs": [
                    "Parameter_178_0"
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
                "name": "Parameter_177",
                "op": "Parameter",
                "outputs": [
                    "Parameter_177_0"
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
                "name": "Constant_182",
                "op": "Constant",
                "outputs": [
                    "Constant_182_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_179",
                "op": "Constant",
                "outputs": [
                    "Constant_179_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_188",
                "op": "Constant",
                "outputs": [
                    "Constant_188_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "input_order": [
                    3,
                    2,
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_178"
                ],
                "name": "Reshape_185",
                "op": "Reshape",
                "output_shape": [
                    32,
                    16,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_185_0"
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
                    "Constant_179"
                ],
                "name": "Broadcast_180",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_180_0"
                ],
                "shape": [
                    2,
                    32,
                    32,
                    16
                ]
            },
            {
                "inputs": [
                    "Broadcast_180",
                    "Parameter_177"
                ],
                "name": "Maximum_181",
                "op": "Maximum",
                "outputs": [
                    "Maximum_181_0"
                ]
            },
            {
                "inputs": [
                    "Maximum_181",
                    "Constant_188"
                ],
                "name": "Pad_189",
                "op": "Pad",
                "outputs": [
                    "Pad_189_0"
                ],
                "padding_above": [
                    0,
                    0,
                    0,
                    0
                ],
                "padding_below": [
                    0,
                    0,
                    0,
                    0
                ],
                "padding_interior": [
                    0,
                    0,
                    0,
                    0
                ]
            },
            {
                "inputs": [
                    "Maximum_181",
                    "Constant_182"
                ],
                "name": "Pad_183",
                "op": "Pad",
                "outputs": [
                    "Pad_183_0"
                ],
                "padding_above": [
                    0,
                    1,
                    1,
                    0
                ],
                "padding_below": [
                    0,
                    1,
                    1,
                    0
                ],
                "padding_interior": [
                    0,
                    0,
                    0,
                    0
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
                    "Pad_183"
                ],
                "name": "Reshape_184",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    34,
                    34
                ],
                "outputs": [
                    "Reshape_184_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_184",
                    "Reshape_185"
                ],
                "name": "Convolution_186",
                "op": "Convolution",
                "outputs": [
                    "Convolution_186_0"
                ],
                "padding_above": [
                    0,
                    0
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
                    2,
                    2
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
                    "Convolution_186"
                ],
                "name": "Reshape_187",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_187_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_177",
            "Parameter_178"
        ],
        "result": [
            "Reshape_187",
            "Pad_189",
            "Maximum_181",
            "Pad_183"
        ]
    }
]