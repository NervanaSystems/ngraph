[
    {
        "name": "Function_1",
        "ops": [
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_98",
                "op": "Parameter",
                "outputs": [
                    "Parameter_98_0"
                ],
                "shape": [
                    10000,
                    784
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_97",
                "op": "Parameter",
                "outputs": [
                    "Parameter_97_0"
                ],
                "shape": [
                    10
                ]
            },
            {
                "element_type": "float",
                "inputs": [],
                "name": "Parameter_96",
                "op": "Parameter",
                "outputs": [
                    "Parameter_96_0"
                ],
                "shape": [
                    784,
                    10
                ]
            },
            {
                "axes": [
                    0
                ],
                "inputs": [
                    "Parameter_97"
                ],
                "name": "Broadcast_100",
                "op": "Broadcast",
                "outputs": [
                    "Broadcast_100_0"
                ],
                "shape": [
                    10000,
                    10
                ]
            },
            {
                "inputs": [
                    "Parameter_98",
                    "Parameter_96"
                ],
                "name": "Dot_99",
                "op": "Dot",
                "outputs": [
                    "Dot_99_0"
                ],
                "reduction_axes_count": 1
            },
            {
                "inputs": [
                    "Dot_99",
                    "Broadcast_100"
                ],
                "name": "Add_101",
                "op": "Add",
                "outputs": [
                    "Add_101_0"
                ]
            }
        ],
        "parameters": [
            "Parameter_96",
            "Parameter_97",
            "Parameter_98"
        ],
        "result": [
            "Add_101"
        ]
    }
]