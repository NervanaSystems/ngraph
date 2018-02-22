[
    {
        "name": "Function_11",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_681",
                "op": "Parameter",
                "outputs": [
                    "Parameter_681_0"
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
                "name": "Parameter_680",
                "op": "Parameter",
                "outputs": [
                    "Parameter_680_0"
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
                "name": "Parameter_679",
                "op": "Parameter",
                "outputs": [
                    "Parameter_679_0"
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
                "name": "Parameter_678",
                "op": "Parameter",
                "outputs": [
                    "Parameter_678_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_677",
                "op": "Parameter",
                "outputs": [
                    "Parameter_677_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_676",
                "op": "Parameter",
                "outputs": [
                    "Parameter_676_0"
                ],
                "shape": [
                    64,
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_675",
                "op": "Parameter",
                "outputs": [
                    "Parameter_675_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_674",
                "op": "Parameter",
                "outputs": [
                    "Parameter_674_0"
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
                "name": "Parameter_673",
                "op": "Parameter",
                "outputs": [
                    "Parameter_673_0"
                ],
                "shape": [
                    1,
                    1,
                    32,
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_672",
                "op": "Parameter",
                "outputs": [
                    "Parameter_672_0"
                ],
                "shape": [
                    3,
                    3,
                    64,
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_671",
                "op": "Parameter",
                "outputs": [
                    "Parameter_671_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_670",
                "op": "Parameter",
                "outputs": [
                    "Parameter_670_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_669",
                "op": "Parameter",
                "outputs": [
                    "Parameter_669_0"
                ],
                "shape": [
                    64,
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_668",
                "op": "Parameter",
                "outputs": [
                    "Parameter_668_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_667",
                "op": "Parameter",
                "outputs": [
                    "Parameter_667_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_666",
                "op": "Parameter",
                "outputs": [
                    "Parameter_666_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_665",
                "op": "Parameter",
                "outputs": [
                    "Parameter_665_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_689",
                "op": "Constant",
                "outputs": [
                    "Constant_689_0"
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
                    "Parameter_681"
                ],
                "name": "Reshape_715",
                "op": "Reshape",
                "output_shape": [
                    64,
                    2,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_715_0"
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
                    "Parameter_680"
                ],
                "name": "Reshape_724",
                "op": "Reshape",
                "output_shape": [
                    32,
                    2,
                    16,
                    16
                ],
                "outputs": [
                    "Reshape_724_0"
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
                    "Parameter_679"
                ],
                "name": "Reshape_683",
                "op": "Reshape",
                "output_shape": [
                    2,
                    8,
                    8,
                    64
                ],
                "outputs": [
                    "Reshape_683_0"
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
                    "Parameter_679"
                ],
                "name": "Reshape_692",
                "op": "Reshape",
                "output_shape": [
                    2,
                    8,
                    8,
                    64
                ],
                "outputs": [
                    "Reshape_692_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_673"
                ],
                "name": "Reverse_684",
                "op": "Reverse",
                "outputs": [
                    "Reverse_684_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "inputs": [
                    "Parameter_672"
                ],
                "name": "Reverse_693",
                "op": "Reverse",
                "outputs": [
                    "Reverse_693_0"
                ],
                "reversed_axes": [
                    0,
                    1
                ]
            },
            {
                "input_order": [],
                "inputs": [
                    "Parameter_665"
                ],
                "name": "Reshape_682",
                "op": "Reshape",
                "output_shape": [],
                "outputs": [
                    "Reshape_682_0"
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
                    "Constant_689"
                ],
                "name": "Broadcast_690",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_690_0"
                ],
                "shape": [
                    2,
                    8,
                    8,
                    64
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
                    "Reshape_683"
                ],
                "name": "Reshape_725",
                "op": "Reshape",
                "output_shape": [
                    64,
                    2,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_725_0"
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
                    "Reshape_683"
                ],
                "name": "Reshape_685",
                "op": "Reshape",
                "output_shape": [
                    2,
                    64,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_685_0"
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
                    "Reshape_692"
                ],
                "name": "Reshape_694",
                "op": "Reshape",
                "output_shape": [
                    2,
                    64,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_694_0"
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
                    "Reshape_692"
                ],
                "name": "Reshape_716",
                "op": "Reshape",
                "output_shape": [
                    64,
                    2,
                    8,
                    8
                ],
                "outputs": [
                    "Reshape_716_0"
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
                    "Reverse_684"
                ],
                "name": "Reshape_686",
                "op": "Reshape",
                "output_shape": [
                    32,
                    64,
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_686_0"
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
                    "Reverse_693"
                ],
                "name": "Reshape_695",
                "op": "Reshape",
                "output_shape": [
                    64,
                    64,
                    3,
                    3
                ],
                "outputs": [
                    "Reshape_695_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Reshape_682"
                ],
                "name": "Broadcast_707",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_707_0"
                ],
                "shape": [
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
                    "Reshape_682"
                ],
                "name": "Broadcast_713",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_713_0"
                ],
                "shape": [
                    3,
                    3,
                    64,
                    64
                ]
            },
            {
                "axes": [
                    0,
                    1
                ],
                "inputs": [
                    "Reshape_682"
                ],
                "name": "Broadcast_704",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_704_0"
                ],
                "shape": [
                    64,
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Reshape_682"
                ],
                "name": "Broadcast_701",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_701_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Reshape_682"
                ],
                "name": "Broadcast_710",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_710_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "input_order": [],
                "inputs": [
                    "Reshape_682"
                ],
                "name": "Reshape_721",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1
                ],
                "outputs": [
                    "Reshape_721_0"
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
                    "Reshape_682"
                ],
                "name": "Broadcast_699",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_699_0"
                ],
                "shape": [
                    3,
                    3,
                    16,
                    16
                ]
            },
            {
                "inputs": [
                    "Parameter_681",
                    "Broadcast_690"
                ],
                "name": "Greater_691",
                "op": "Greater",
                "outputs": [
                    "Greater_691_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_724",
                    "Reshape_725"
                ],
                "name": "Convolution_726",
                "op": "Convolution",
                "outputs": [
                    "Convolution_726_0"
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
                    "Reshape_715",
                    "Reshape_716"
                ],
                "name": "Convolution_717",
                "op": "Convolution",
                "outputs": [
                    "Convolution_717_0"
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
                    2,
                    2
                ],
                "inputs": [
                    "Reshape_685",
                    "Reshape_686"
                ],
                "name": "Convolution_687",
                "op": "Convolution",
                "outputs": [
                    "Convolution_687_0"
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
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_694",
                    "Reshape_695"
                ],
                "name": "Convolution_696",
                "op": "Convolution",
                "outputs": [
                    "Convolution_696_0"
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
                    "Parameter_670",
                    "Broadcast_707"
                ],
                "name": "Multiply_708",
                "op": "Multiply",
                "outputs": [
                    "Multiply_708_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_672",
                    "Broadcast_713"
                ],
                "name": "Multiply_714",
                "op": "Multiply",
                "outputs": [
                    "Multiply_714_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_669",
                    "Broadcast_704"
                ],
                "name": "Multiply_705",
                "op": "Multiply",
                "outputs": [
                    "Multiply_705_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_668",
                    "Broadcast_701"
                ],
                "name": "Multiply_702",
                "op": "Multiply",
                "outputs": [
                    "Multiply_702_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_671",
                    "Broadcast_710"
                ],
                "name": "Multiply_711",
                "op": "Multiply",
                "outputs": [
                    "Multiply_711_0"
                ]
            },
            {
                "axes": [
                    2,
                    3
                ],
                "inputs": [
                    "Reshape_721"
                ],
                "name": "Broadcast_722",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_722_0"
                ],
                "shape": [
                    1,
                    1,
                    32,
                    64
                ]
            },
            {
                "inputs": [
                    "Parameter_674",
                    "Broadcast_699"
                ],
                "name": "Multiply_700",
                "op": "Multiply",
                "outputs": [
                    "Multiply_700_0"
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
                    "Convolution_726"
                ],
                "name": "Reshape_727",
                "op": "Reshape",
                "output_shape": [
                    64,
                    1,
                    1,
                    32
                ],
                "outputs": [
                    "Reshape_727_0"
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
                    "Convolution_717"
                ],
                "name": "Reshape_718",
                "op": "Reshape",
                "output_shape": [
                    64,
                    3,
                    3,
                    64
                ],
                "outputs": [
                    "Reshape_718_0"
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
                    "Convolution_687"
                ],
                "name": "Reshape_688",
                "op": "Reshape",
                "output_shape": [
                    2,
                    16,
                    16,
                    32
                ],
                "outputs": [
                    "Reshape_688_0"
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
                    "Convolution_696"
                ],
                "name": "Reshape_697",
                "op": "Reshape",
                "output_shape": [
                    2,
                    8,
                    8,
                    64
                ],
                "outputs": [
                    "Reshape_697_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_708",
                    "Parameter_678"
                ],
                "name": "Add_709",
                "op": "Add",
                "outputs": [
                    "Add_709_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_705",
                    "Parameter_676"
                ],
                "name": "Add_706",
                "op": "Add",
                "outputs": [
                    "Add_706_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_702",
                    "Parameter_675"
                ],
                "name": "Add_703",
                "op": "Add",
                "outputs": [
                    "Add_703_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_711",
                    "Parameter_677"
                ],
                "name": "Add_712",
                "op": "Add",
                "outputs": [
                    "Add_712_0"
                ]
            },
            {
                "inputs": [
                    "Parameter_673",
                    "Broadcast_722"
                ],
                "name": "Multiply_723",
                "op": "Multiply",
                "outputs": [
                    "Multiply_723_0"
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
                    "Reshape_727"
                ],
                "name": "Reshape_728",
                "op": "Reshape",
                "output_shape": [
                    1,
                    1,
                    32,
                    64
                ],
                "outputs": [
                    "Reshape_728_0"
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
                    "Reshape_718"
                ],
                "name": "Reshape_719",
                "op": "Reshape",
                "output_shape": [
                    3,
                    3,
                    64,
                    64
                ],
                "outputs": [
                    "Reshape_719_0"
                ]
            },
            {
                "inputs": [
                    "Greater_691",
                    "Reshape_697",
                    "Broadcast_690"
                ],
                "name": "Select_698",
                "op": "Select",
                "outputs": [
                    "Select_698_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_723",
                    "Reshape_728"
                ],
                "name": "Add_729",
                "op": "Add",
                "outputs": [
                    "Add_729_0"
                ]
            },
            {
                "inputs": [
                    "Multiply_714",
                    "Reshape_719"
                ],
                "name": "Add_720",
                "op": "Add",
                "outputs": [
                    "Add_720_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_665",
            "Parameter_666",
            "Parameter_667",
            "Parameter_668",
            "Parameter_669",
            "Parameter_670",
            "Parameter_671",
            "Parameter_672",
            "Parameter_673",
            "Parameter_674",
            "Parameter_675",
            "Parameter_676",
            "Parameter_677",
            "Parameter_678",
            "Parameter_679",
            "Parameter_680",
            "Parameter_681"
        ],
        "result": [
            "Reshape_682",
            "Reshape_688",
            "Select_698",
            "Multiply_700",
            "Add_703",
            "Add_706",
            "Add_709",
            "Add_712",
            "Add_720",
            "Add_729"
        ]
    }
]