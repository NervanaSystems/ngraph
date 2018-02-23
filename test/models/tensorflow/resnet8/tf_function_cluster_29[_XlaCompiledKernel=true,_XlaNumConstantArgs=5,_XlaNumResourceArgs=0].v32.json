[
    {
        "name": "Function_16",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_859",
                "op": "Parameter",
                "outputs": [
                    "Parameter_859_0"
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
                "name": "Parameter_858",
                "op": "Parameter",
                "outputs": [
                    "Parameter_858_0"
                ],
                "shape": [
                    2,
                    8,
                    8,
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_857",
                "op": "Parameter",
                "outputs": [
                    "Parameter_857_0"
                ],
                "shape": [
                    2,
                    18,
                    18,
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_856",
                "op": "Parameter",
                "outputs": [
                    "Parameter_856_0"
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
                "name": "Parameter_855",
                "op": "Parameter",
                "outputs": [
                    "Parameter_855_0"
                ],
                "shape": [
                    3,
                    3,
                    32,
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_854",
                "op": "Parameter",
                "outputs": [
                    "Parameter_854_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_860",
                "op": "Constant",
                "outputs": [
                    "Constant_860_0"
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
                    "Parameter_858"
                ],
                "name": "Reshape_916",
                "op": "Reshape",
                "output_shape": [
                    64,
                    2,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_916_0"
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
                    "Parameter_858"
                ],
                "name": "Reshape_864",
                "op": "Reshape",
                "output_shape": [
                    2,
                    64,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_864_0"
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
                    "Parameter_857"
                ],
                "name": "Reshape_915",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    18,
                    18
                ],
                "outputs": [
                    "Reshape_915_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_855"
                ],
                "name": "Reverse_863",
                "op": "Reverse",
                "outputs": [
                    "Reverse_863_0"
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
                    "Parameter_854"
                ],
                "name": "Broadcast_913",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_913_0"
                ],
                "shape": [
                    3,
                    3,
                    32,
                    64
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
                    "Constant_860"
                ],
                "name": "Broadcast_861",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_861_0"
                ],
                "shape": [
                    2,
                    16,
                    16,
                    32
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_915",
                    "Reshape_916"
                ],
                "name": "Convolution_917",
                "op": "Convolution",
                "outputs": [
                    "Convolution_917_0"
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
                    "Reverse_863"
                ],
                "name": "Reshape_865",
                "op": "Reshape",
                "output_shape": [
                    32,
                    64,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_865_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_855",
                    "Broadcast_913"
                ],
                "name": "Multiply_914",
                "op": "Multiply",
                "outputs": [
                    "Multiply_914_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_859",
                    "Broadcast_861"
                ],
                "name": "Greater_862",
                "op": "Greater",
                "outputs": [
                    "Greater_862_0"
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
                    "Convolution_917"
                ],
                "name": "Reshape_918",
                "op": "Reshape",
                "output_shape": [
                    64,
                    3,
                    3,
                    32
                ],
                "outputs": [
                    "Reshape_918_0"
                ]
            },
            {
                "data_dilation_strides": [
                    2,
                    2
                ],
                "inputs": [
                    "Reshape_864",
                    "Reshape_865"
                ],
                "name": "Convolution_866",
                "op": "Convolution",
                "outputs": [
                    "Convolution_866_0"
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
                "input_order": [
                    1,
                    2,
                    3,
                    0
                ],
                "inputs": [
                    "Reshape_918"
                ],
                "name": "Reshape_919",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    32,
                    64
                ],
                "outputs": [
                    "Reshape_919_0"
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
                    "Convolution_866"
                ],
                "name": "Reshape_867",
                "op": "Reshape",
                "output_shape": [
                    2,
                    18,
                    18,
                    32
                ],
                "outputs": [
                    "Reshape_867_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_914",
                    "Reshape_919"
                ],
                "name": "Add_920",
                "op": "Add",
                "outputs": [
                    "Add_920_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_867"
                ],
                "lower_bounds": [
                    0,
                    1,
                    1,
                    0
                ],
                "name": "Slice_868",
                "op": "Slice",
                "outputs": [
                    "Slice_868_0"
                ],
                "strides": [
                    1,
                    1,
                    1,
                    1
                ],
                "upper_bounds": [
                    2,
                    17,
                    17,
                    32
                ]
            },
            {
                "inputs": [
                    "Parameter_856",
                    "Slice_868"
                ],
                "name": "Add_869",
                "op": "Add",
                "outputs": [
                    "Add_869_0"
                ]
            },
            {
                "inputs": [
                    "Greater_862",
                    "Add_869",
                    "Broadcast_861"
                ],
                "name": "Select_870",
                "op": "Select",
                "outputs": [
                    "Select_870_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_854",
            "Parameter_855",
            "Parameter_856",
            "Parameter_857",
            "Parameter_858",
            "Parameter_859"
        ],
        "result": [
            "Select_870",
            "Add_920"
        ]
    }
]