[
    {
        "name": "Function_5",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_184",
                "op": "Parameter",
                "outputs": [
                    "Parameter_184_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_183",
                "op": "Parameter",
                "outputs": [
                    "Parameter_183_0"
                ],
                "shape": []
            },
            {
                "inputs": [
                    "Parameter_183",
                    "Parameter_184"
                ],
                "name": "Maximum_185",
                "op": "Maximum",
                "outputs": [
                    "Maximum_185_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_183",
            "Parameter_184"
        ],
        "result": [
            "Maximum_185"
        ]
    },
    {
        "name": "Function_4",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_171",
                "op": "Parameter",
                "outputs": [
                    "Parameter_171_0"
                ],
                "shape": []
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_170",
                "op": "Parameter",
                "outputs": [
                    "Parameter_170_0"
                ],
                "shape": []
            },
            {
                "inputs": [
                    "Parameter_170",
                    "Parameter_171"
                ],
                "name": "Maximum_172",
                "op": "Maximum",
                "outputs": [
                    "Maximum_172_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_170",
            "Parameter_171"
        ],
        "result": [
            "Maximum_172"
        ]
    },
    {
        "name": "Function_6",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_156",
                "op": "Parameter",
                "outputs": [
                    "Parameter_156_0"
                ],
                "shape": [
                    50,
                    784
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_155",
                "op": "Parameter",
                "outputs": [
                    "Parameter_155_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_154",
                "op": "Parameter",
                "outputs": [
                    "Parameter_154_0"
                ],
                "shape": [
                    1024,
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_153",
                "op": "Parameter",
                "outputs": [
                    "Parameter_153_0"
                ],
                "shape": [
                    1024
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_152",
                "op": "Parameter",
                "outputs": [
                    "Parameter_152_0"
                ],
                "shape": [
                    3136,
                    1024
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_151",
                "op": "Parameter",
                "outputs": [
                    "Parameter_151_0"
                ],
                "shape": [
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_150",
                "op": "Parameter",
                "outputs": [
                    "Parameter_150_0"
                ],
                "shape": [
                    5,
                    5,
                    32,
                    64
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_149",
                "op": "Parameter",
                "outputs": [
                    "Parameter_149_0"
                ],
                "shape": [
                    32
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_148",
                "op": "Parameter",
                "outputs": [
                    "Parameter_148_0"
                ],
                "shape": [
                    5,
                    5,
                    1,
                    32
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
                    "-INFINITY"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_169",
                "op": "Constant",
                "outputs": [
                    "Constant_169_0"
                ],
                "shape": [],
                "value": [
                    "-INFINITY"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_159",
                "op": "Constant",
                "outputs": [
                    "Constant_159_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_158",
                "op": "Constant",
                "outputs": [
                    "Constant_158_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Constant_157",
                "op": "Constant",
                "outputs": [
                    "Constant_157_0"
                ],
                "shape": [],
                "value": [
                    "0"
                ]
            },
            {
                "input_order": [
                    0,
                    1
                ],
                "inputs": [
                    "Parameter_156"
                ],
                "name": "Reshape_160",
                "op": "Reshape",
                "output_shape": [
                    50,
                    28,
                    28,
                    1
                ],
                "outputs": [
                    "Reshape_160_0"
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_155"
                ],
                "name": "Broadcast_194",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_194_0"
                ],
                "shape": [
                    50,
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_153"
                ],
                "name": "Broadcast_189",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_189_0"
                ],
                "shape": [
                    50,
                    1024
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_151"
                ],
                "name": "Broadcast_178",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_178_0"
                ],
                "shape": [
                    50,
                    14,
                    14,
                    64
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
                    "Parameter_150"
                ],
                "name": "Reshape_175",
                "op": "Reshape",
                "output_shape": [
                    64,
                    32,
                    5,
                    5
                ],
                "outputs": [
                    "Reshape_175_0"
                ]
            },
            {
                "axes": [
                    0,
                    1,
                    2
                ],
                "inputs": [
                    "Parameter_149"
                ],
                "name": "Broadcast_165",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_165_0"
                ],
                "shape": [
                    50,
                    28,
                    28,
                    32
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
                    "Parameter_148"
                ],
                "name": "Reshape_162",
                "op": "Reshape",
                "output_shape": [
                    32,
                    1,
                    5,
                    5
                ],
                "outputs": [
                    "Reshape_162_0"
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
                    "Constant_159"
                ],
                "name": "Broadcast_167",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_167_0"
                ],
                "shape": [
                    50,
                    28,
                    28,
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
                    "Constant_158"
                ],
                "name": "Broadcast_180",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_180_0"
                ],
                "shape": [
                    50,
                    14,
                    14,
                    64
                ]
            },
            {
                "axes": [
                    0,
                    1
                ],
                "inputs": [
                    "Constant_157"
                ],
                "name": "Broadcast_191",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_191_0"
                ],
                "shape": [
                    50,
                    1024
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
                    "Reshape_160"
                ],
                "name": "Reshape_161",
                "op": "Reshape",
                "output_shape": [
                    50,
                    1,
                    28,
                    28
                ],
                "outputs": [
                    "Reshape_161_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_161",
                    "Reshape_162"
                ],
                "name": "Convolution_163",
                "op": "Convolution",
                "outputs": [
                    "Convolution_163_0"
                ],
                "padding_above": [
                    2,
                    2
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
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_163"
                ],
                "name": "Reshape_164",
                "op": "Reshape",
                "output_shape": [
                    50,
                    28,
                    28,
                    32
                ],
                "outputs": [
                    "Reshape_164_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_164",
                    "Broadcast_165"
                ],
                "name": "Add_166",
                "op": "Add",
                "outputs": [
                    "Add_166_0"
                ]
            },
            {
                "inputs": [
                    "Broadcast_167",
                    "Add_166"
                ],
                "name": "Maximum_168",
                "op": "Maximum",
                "outputs": [
                    "Maximum_168_0"
                ]
            },
            {
                "function": "Function_4",
                "inputs": [
                    "Maximum_168",
                    "Constant_169"
                ],
                "name": "ReduceWindow_173",
                "op": "ReduceWindow",
                "outputs": [
                    "ReduceWindow_173_0"
                ],
                "window_movement_strides": [
                    1,
                    2,
                    2,
                    1
                ],
                "window_shape": [
                    1,
                    2,
                    2,
                    1
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
                    "ReduceWindow_173"
                ],
                "name": "Reshape_174",
                "op": "Reshape",
                "output_shape": [
                    50,
                    32,
                    14,
                    14
                ],
                "outputs": [
                    "Reshape_174_0"
                ]
            },
            {
                "data_dilation_strides": [
                    1,
                    1
                ],
                "inputs": [
                    "Reshape_174",
                    "Reshape_175"
                ],
                "name": "Convolution_176",
                "op": "Convolution",
                "outputs": [
                    "Convolution_176_0"
                ],
                "padding_above": [
                    2,
                    2
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
                    0,
                    2,
                    3,
                    1
                ],
                "inputs": [
                    "Convolution_176"
                ],
                "name": "Reshape_177",
                "op": "Reshape",
                "output_shape": [
                    50,
                    14,
                    14,
                    64
                ],
                "outputs": [
                    "Reshape_177_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_177",
                    "Broadcast_178"
                ],
                "name": "Add_179",
                "op": "Add",
                "outputs": [
                    "Add_179_0"
                ]
            },
            {
                "inputs": [
                    "Broadcast_180",
                    "Add_179"
                ],
                "name": "Maximum_181",
                "op": "Maximum",
                "outputs": [
                    "Maximum_181_0"
                ]
            },
            {
                "function": "Function_5",
                "inputs": [
                    "Maximum_181",
                    "Constant_182"
                ],
                "name": "ReduceWindow_186",
                "op": "ReduceWindow",
                "outputs": [
                    "ReduceWindow_186_0"
                ],
                "window_movement_strides": [
                    1,
                    2,
                    2,
                    1
                ],
                "window_shape": [
                    1,
                    2,
                    2,
                    1
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
                    "ReduceWindow_186"
                ],
                "name": "Reshape_187",
                "op": "Reshape",
                "output_shape": [
                    50,
                    3136
                ],
                "outputs": [
                    "Reshape_187_0"
                ]
            },
            {
                "inputs": [
                    "Reshape_187",
                    "Parameter_152"
                ],
                "name": "Dot_188",
                "op": "Dot",
                "outputs": [
                    "Dot_188_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_188",
                    "Broadcast_189"
                ],
                "name": "Add_190",
                "op": "Add",
                "outputs": [
                    "Add_190_0"
                ]
            },
            {
                "inputs": [
                    "Broadcast_191",
                    "Add_190"
                ],
                "name": "Maximum_192",
                "op": "Maximum",
                "outputs": [
                    "Maximum_192_0"
                ]
            },
            {
                "inputs": [
                    "Maximum_192",
                    "Parameter_154"
                ],
                "name": "Dot_193",
                "op": "Dot",
                "outputs": [
                    "Dot_193_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_193",
                    "Broadcast_194"
                ],
                "name": "Add_195",
                "op": "Add",
                "outputs": [
                    "Add_195_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_148",
            "Parameter_149",
            "Parameter_150",
            "Parameter_151",
            "Parameter_152",
            "Parameter_153",
            "Parameter_154",
            "Parameter_155",
            "Parameter_156"
        ],
        "result": [
            "Add_195"
        ]
    }
]