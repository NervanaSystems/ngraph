[
    {
        "name": "Function_3",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_132",
                "op": "Parameter",
                "outputs": [
                    "Parameter_132_0"
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
                "name": "Parameter_131",
                "op": "Parameter",
                "outputs": [
                    "Parameter_131_0"
                ],
                "shape": [
                    3,
                    3,
                    16,
                    16
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_130",
                "op": "Parameter",
                "outputs": [
                    "Parameter_130_0"
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
                "name": "Constant_133",
                "op": "Constant",
                "outputs": [
                    "Constant_133_0"
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
                    "Parameter_131"
                ],
                "name": "Reshape_137",
                "op": "Reshape",
                "output_shape": [
                    16,
                    16,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_137_0"
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
                    "Constant_133"
                ],
                "name": "Broadcast_134",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_134_0"
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
                    "Broadcast_134",
                    "Parameter_130"
                ],
                "name": "Maximum_135",
                "op": "Maximum",
                "outputs": [
                    "Maximum_135_0"
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
                    "Maximum_135"
                ],
                "name": "Reshape_136",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_136_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_136",
                    "Reshape_137"
                ],
                "name": "Convolution_138",
                "op": "Convolution",
                "outputs": [
                    "Convolution_138_0"
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
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_138"
                ],
                "name": "Reshape_139",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_139_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_139",
                    "Parameter_132"
                ],
                "name": "Add_140",
                "op": "Add",
                "outputs": [
                    "Add_140_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_130",
            "Parameter_131",
            "Parameter_132"
        ],
        "result": [
            "Add_140",
            "Maximum_135",
            "Reshape_139"
        ]
    }
]