[
    {
        "name": "Function_27",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1485",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1485_0"
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
                "name": "Parameter_1484",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1484_0"
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
                "name": "Parameter_1483",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1483_0"
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
                "name": "Parameter_1482",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1482_0"
                ],
                "shape": [
                    1,
                    1,
                    16,
                    16
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_1481",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1481_0"
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
                "name": "Parameter_1480",
                "op": "Parameter",
                "outputs": [
                    "Parameter_1480_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_1486",
                "op": "Constant",
                "outputs": [
                    "Constant_1486_0"
                ],
                "shape": [],
                "value": [
                    "0"
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
                    "Parameter_1485"
                ],
                "name": "Reshape_1495",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1495_0"
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
                    "Parameter_1485"
                ],
                "name": "Reshape_1504",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1504_0"
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
                    "Parameter_1484"
                ],
                "name": "Reshape_1503",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1503_0"
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
                    "Parameter_1484"
                ],
                "name": "Reshape_1512",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1512_0"
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
                    "Parameter_1483"
                ],
                "name": "Reshape_1513",
                "op": "Reshape",
                "output_shape": [
                    16,
                    2,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1513_0"
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
                    "Parameter_1483"
                ],
                "name": "Reshape_1490",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    32,
                    32
                ],
                "outputs": [
                    "Reshape_1490_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1482"
                ],
                "name": "Reverse_1489",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1489_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "inputs": [
                    "Parameter_1481"
                ],
                "name": "Reverse_1494",
                "op": "Reverse",
                "outputs": [
                    "Reverse_1494_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "input_order": [],
                "inputs": [
                    "Parameter_1480"
                ],
                "name": "Reshape_1509",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_1509_0"
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
                    "Parameter_1480"
                ],
                "name": "Broadcast_1501",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1501_0"
                ],
                "shape": [
                    3,
                    3,
                    16,
                    16
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
                    "Constant_1486"
                ],
                "name": "Broadcast_1487",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1487_0"
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
                    "Reshape_1503",
                    "Reshape_1504"
                ],
                "name": "Convolution_1505",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1505_0"
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
                    "Reshape_1512",
                    "Reshape_1513"
                ],
                "name": "Convolution_1514",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1514_0"
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
                    "Reverse_1489"
                ],
                "name": "Reshape_1491",
                "op": "Reshape",
                "output_shape": [
                    16,
                    16,
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_1491_0"
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
                    "Reverse_1494"
                ],
                "name": "Reshape_1496",
                "op": "Reshape",
                "output_shape": [
                    16,
                    16,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_1496_0"
                ]
            },
            {
                "axes": [
                    2,
                    3
                ],
                "inputs": [
                    "Reshape_1509"
                ],
                "name": "Broadcast_1510",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_1510_0"
                ],
                "shape": [
                    1,
                    1,
                    16,
                    16
                ]
            },
            {
                "inputs": [
                    "Parameter_1481",
                    "Broadcast_1501"
                ],
                "name": "Multiply_1502",
                "op": "Multiply",
                "outputs": [
                    "Multiply_1502_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_1484",
                    "Broadcast_1487"
                ],
                "name": "Greater_1488",
                "op": "Greater",
                "outputs": [
                    "Greater_1488_0"
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
                    "Convolution_1505"
                ],
                "name": "Reshape_1506",
                "op": "Reshape",
                "output_shape": [
                    16,
                    3,
                    3,
                    16
                ],
                "outputs": [
                    "Reshape_1506_0"
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
                    "Convolution_1514"
                ],
                "name": "Reshape_1515",
                "op": "Reshape",
                "output_shape": [
                    16,
                    1,
                    1,
                    16
                ],
                "outputs": [
                    "Reshape_1515_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_1490",
                    "Reshape_1491"
                ],
                "name": "Convolution_1492",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1492_0"
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
                    "Reshape_1495",
                    "Reshape_1496"
                ],
                "name": "Convolution_1497",
                "op": "Convolution",
                "outputs": [
                    "Convolution_1497_0"
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
                "inputs": [
                    "Parameter_1482",
                    "Broadcast_1510"
                ],
                "name": "Multiply_1511",
                "op": "Multiply",
                "outputs": [
                    "Multiply_1511_0"
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
                    "Reshape_1506"
                ],
                "name": "Reshape_1507",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1507_0"
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
                    "Reshape_1515"
                ],
                "name": "Reshape_1516",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_1516_0"
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
                    "Convolution_1492"
                ],
                "name": "Reshape_1493",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1493_0"
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
                    "Convolution_1497"
                ],
                "name": "Reshape_1498",
                "op": "Reshape",
                "output_shape": [
                    2,
                    32,
                    32,
                    16
                ],
                "outputs": [
                    "Reshape_1498_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_1502",
                    "Reshape_1507"
                ],
                "name": "Add_1508",
                "op": "Add",
                "outputs": [
                    "Add_1508_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_1511",
                    "Reshape_1516"
                ],
                "name": "Add_1517",
                "op": "Add",
                "outputs": [
                    "Add_1517_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_1493",
                    "Reshape_1498"
                ],
                "name": "Add_1499",
                "op": "Add",
                "outputs": [
                    "Add_1499_0"
                ]
            },
            {
                "inputs": [
                    "Greater_1488",
                    "Add_1499",
                    "Broadcast_1487"
                ],
                "name": "Select_1500",
                "op": "Select",
                "outputs": [
                    "Select_1500_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_1480",
            "Parameter_1481",
            "Parameter_1482",
            "Parameter_1483",
            "Parameter_1484",
            "Parameter_1485"
        ],
        "result": [
            "Select_1500",
            "Add_1508",
            "Add_1517"
        ]
    }
]