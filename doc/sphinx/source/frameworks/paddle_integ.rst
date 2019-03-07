.. paddle_integ.rst:

PaddlePaddle
============

nGraph PaddlePaddle integration overview
----------------------------------------
PaddlePaddle is an open source deep learning framework developed by Baidu. It aims to enable performant large scale distributed computation for deep learning. nGraph Compiler stack is integrated with the current version of PaddlePaddle (Fluid v1.4) and respects PaddlePaddle’s design philosophies to minimize switching cost for users. In order to access nGraph from PaddlePaddle, we added three modules to PaddlePaddle: nGrah engine operator (op), nGraph engine, and nGraph bridge. 

nGraph engine op inherits the PaddlePaddle operator class to allow nGraph engine op to be called using methods consistent with other PaddlePaddle operators. When the nGraph engine is called by the aforementioned op, the nGraph bridge takes and converts PaddlePaddle operators into nGraph operators. nGraph will then build a computational graph based on the converted ops according to the input topology. 

Integration design
----------------------------------------

Here are key design criteria for nGraph PaddlePaddle integration:

1. Minimal intermediate links between nGraph and PaddlePaddle to reduce latency and improve performance
2. Close to no switching cost for end users of PaddlePaddle framework
3. Ease of maintenance 


To satisfy the first design criteria, nGraph designed its operator to match PaddlePaddle implementation. nGraph is triggered in the executor exactly in the same way as MKL-DNN and requires one line of code. 

Once nGraph engine is called, performance optimization is handled by nGraph engine and its C++ backend. There is no change made to the PaddlePaddle's python frontend, and end users are not required to change their code to take advantage of nGraph's performance. This design fulfills the second criteria.

Lastly, the code contributed by the nGraph team to PaddlePaddle repository mainly resides in the fluid/operator/ngraph directory, and having the nGraph bridge code in one place allows for easy maintenance. 

![](https://github.com/NervanaSystems/ngraph-paddle/raw/ngraph/doc/fluid/design/ngraph/images/ngraph_flow.png)

The diagram above depicts nGraph access from PaddlePaddle. The PaddlePaddle executor generates executable operator according to the procedure description (ProgDesc). The ngraph scans the operator sequence before execution and replaces the supported operator subgraph with ngraph. Engine operator. The next execution is returned to Paddle. The paddle can call and execute the ngraph engine operator with a uniform interface.




Integration details 
-------------------



