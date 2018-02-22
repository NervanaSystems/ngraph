[
    {
        "name": "Function_25",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1384",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1384_0"
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
                "name": "Parameter_1383",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1383_0"
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
                "name": "Parameter_1382",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1382_0"
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
                "name": "Parameter_1381",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1381_0"
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
                "name": "Constant_1386",
                "op": "Constant",
                "outputs": [
                    "Constant_1386_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1383"
                ],
                "name": "Reverse_1390",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1390_0"
                ],
                "reversed_axes": [
                    0,
                    1
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
                    "Parameter_1382"
                ],
                "name": "Reshape_1396",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1396_0"
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
                    "Parameter_1381"
                ],
                "name": "Reshape_1385",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1385_0"
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
                    "Parameter_1381"
                ],
                "name": "Reshape_1389",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1389_0"
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
                    "Constant_1386"
                ],
                "name": "Broadcast_1387",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1387_0"
                ],
                "shape": [
                    2,
                    32,
                    32,
                    16
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
                    "Reverse_1390"
                ],
                "name": "Reshape_1392",
                "op": "Reshape",
                "output_shape": [
                    16,
                    16,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_1392_0"
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
                    "Reshape_1389"
                ],
                "name": "Reshape_1397",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1397_0"
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
                    "Reshape_1389"
                ],
                "name": "Reshape_1391",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1391_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1382",
                    "Broadcast_1387"
                ],
                "name": "Greater_1388",
                "op": "Greater",
                "outputs": [
                    "Greater_1388_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1396",
                    "Reshape_1397"
                ],
                "name": "Convolution_1398",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1398_0"
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
                    "Reshape_1391",
                    "Reshape_1392"
                ],
                "name": "Convolution_1393",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1393_0"
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
                    "Convolution_1398"
                ],
                "name": "Reshape_1399",
                "op": "Reshape",
                "output_shape": [
                    16,
                    3,
                    3,
                    16
                ],
                "outputs": [
                    "Reshape_1399_0"
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
                    "Convolution_1393"
                ],
                "name": "Reshape_1394",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1394_0"
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
                    "Reshape_1399"
                ],
                "name": "Reshape_1400",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1400_0"
                ]
            },
            {
                "inputs": [
                    "Greater_1388",
                    "Reshape_1394",
                    "Broadcast_1387"
                ],
                "name": "Select_1395",
                "op": "Select",
                "outputs": [
                    "Select_1395_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1384",
                    "Reshape_1400"
                ],
                "name": "Add_1401",
                "op": "Add",
                "outputs": [
                    "Add_1401_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_1381",
            "Parameter_1382",
            "Parameter_1383",
            "Parameter_1384"
        ],
        "result": [
            "Reshape_1385",
            "Select_1395",
            "Add_1401"
        ]
    }
]