FAQs
----

### Why nGraph?

We developed nGraph to simplify the realization of optimized deep learning 
performance across frameworks and hardware platforms. The value we're offering 
to the developer community is empowerment: we are confident that Intel® 
Architecture already provides the best computational resources available 
for the breadth of ML/DL tasks.


### How do I connect a framework?

The nGraph Library manages framework bridges for some of the more widely-known 
frameworks. A bridge acts as an intermediary between the nGraph core and the 
framework, and the result is a function that can be compiled from a framework. 
A fully-compiled function that makes use of bridge code thus becomes a 
"function graph", or what we sometimes call an **nGraph graph**.

Low-level nGraph APIs are not accessible *dynamically* via bridge code; this 
is the nature of stateless graphs. However, do note that a graph with a 
"saved" checkpoint can be "continued" to run from a previously-applied checkpoint, 
or it can loaded as static graph for further inspection.

For a more detailed dive into how custom bridge code can be implemented, see our 
documentation on [Working with other frameworks]. To learn how TensorFlow and MXNet 
currently make use of custom bridge code, see [Integrate supported frameworks].

![](doc/sphinx/source/graphics/bridge-to-graph-compiler.png) 

<alt="JiT Compiling for computation" width="733" />

Although we only directly support a few frameworks at this time, we provide 
documentation to help developers and engineers create custom solutions. 


### How do I run an inference model?

Framework bridge code is *not* the only way to connect a model (function graph) to 
nGraph's ../ops/index. We've also built an importer for models that have been 
exported from a framework and saved as serialized file, such as ONNX. To learn 
how to convert such serialized files to an nGraph model, please see the "How to" 
documentation.

### What's next?

The Gold release is targeted for April 2019; it will feature broader workload 
coverage, including support for quantized graphs, and more detail on our 
advanced support for ``int8``.  We developed nGraph to simplify the realization 
of optimized deep learning performance across frameworks and hardware platforms. 
You can read more about design decisions and what is tentatively in the pipeline 
for development in our [arXiv paper](https://arxiv.org/pdf/1801.08058.pdf) from 
the 2018 SysML conference.



[Working with other frameworks]: http://ngraph.nervanasys.com/docs/latest/frameworks/index.html
[Integrate supported frameworks]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
