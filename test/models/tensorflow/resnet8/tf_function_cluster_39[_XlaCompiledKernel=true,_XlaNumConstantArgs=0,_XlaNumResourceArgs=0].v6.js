[
    {
        "name": "Function_1",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_38",
                "op": "Parameter",
                "outputs": [
                    "Parameter_38_0"
                ],
                "shape": [
                    3,
                    3,
                    3,
                    16
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_37",
                "op": "Parameter",
                "outputs": [
                    "Parameter_37_0"
                ],
                "shape": [
                    2,
                    32,
                    32,
                    3
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
                    "Parameter_38"
                ],
                "name": "Reshape_41",
                "op": "Reshape",
                "output_shape": [
                    16,
                    3,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_41_0"
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
                    "Parameter_37"
                ],
                "name": "Reshape_39",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    3
                ],
                "outputs": [
                    "Reshape_39_0"
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
                    "Reshape_39"
                ],
                "name": "Reshape_40",
                "op": "Reshape",
                "output_shape": [
                    2,
                    3,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_40_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_40",
                    "Reshape_41"
                ],
                "name": "Convolution_42",
                "op": "Convolution",
                "outputs": [
                    "Convolution_42_0"
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
                    "Convolution_42"
                ],
                "name": "Reshape_43",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_43_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_37",
            "Parameter_38"
        ],
        "result": [
            "Reshape_43",
            "Reshape_39"
        ]
    }
]