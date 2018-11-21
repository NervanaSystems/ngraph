
# Framework & runtime support

One of nGraph’s key features is framework neutrality. We currently support 
popular deep learning frameworks such as TensorFlow and MXNet with stable 
bridges to pass computational graphs to nGraph. Additionally nGraph 
Compiler has functional bridges to PaddlePaddle and PyTorch (via [ONNXIFI]). 
For these frameworks, we have successfully tested functionality with a few 
deep learning workloads, and we plan to bring stable support for them in the 
upcoming releases. 

To further promote framework neutrality, the nGraph team has been actively 
contributing to the ONNX project. Developers who already have a "trained" 
DNN (Deep Neural Network) model can use nGraph to bypass significant 
framework-based complexity and [import it] to test or run on targeted and 
efficient backends with our user-friendly Python-based API.

nGraph is also integrated as an computation provider for [ONNX Runtime], 
which is a runtime for [WinML] on Windows OS and Azure to accelerate DL 
workloads. 

The table below summarizes our current progress on supported frameworks. 
If you are an architect of a framework wishing to take advantage of speed 
and multi-device support of nGraph Compiler, please refer to “how to 
connect custom framework” section on this page. 


|  Framework & Runtime       | Supported          |  Validated 
|----------------------------|--------------------|-------------
| TensorFlow* 1.12           | :heavy_check_mark: |  :heavy_check_mark:
| MXNet* 1.4                 | :heavy_check_mark: |  :heavy_check_mark:
| ONNX 1.3                   | :heavy_check_mark: |  :heavy_check_mark:
| ONNX Runtime  Functional   | Functional         |  No
| PyTorch (via ONNXIFI)      | Functional         |  No
| PaddlePaddle               | Functional         |  No



## Hardware & backend support

The current release of nGraph primarily focuses on accelerating inference 
performance on CPU. However we are also working on adding support for more 
hardware and backends. As with the frameworks, we believe in providing 
freedom to AI developers to deploy their deep learning workloads to the 
desired hardware without a lock in. We currently have functioning backends 
for Intel, Nvidia*, and AMD* GPU either leveraging kernel libraries 
such as clDNN and cuDNN directly or utilizing PlaidML to compile for codegen 
and emit OpenCL, OpenGL, LLVM, Cuda, and Metal. (Please refer to architecture 
and features section to learn more about how we plan to take advantage of 
both solutions using hybrid transformer). In the similar time frame, we plan 
to release multinode support. 

We are excited about providing support for our upcoming deep learning 
accelerators such as NNP (Neural Network Processor) via nGraph compiler 
stack, and early adopters will be able test them in 2019.



| Backend                                       | supported         
|-----------------------------------------------|-------------------
| Intel® Architecture CPU                       | :heavy_check_mark:               
| Intel® Architecture GPUs                      | Functional via clDNN and PlaidML      
| AMD* GPUs                                     | Functional via PlaidML                 
| Nvidia* GPUs                                  | Functional via cuDNN and PlaidML        
| Intel® Nervana™ Neural Network Processor (NNP)| Functional               
| [Upcoming DL accelerators]                    | see details on [Upcoming DL accelerators]       


## How do I connect a framework?

The nGraph Library manages framework bridges for some of the more widely-known 
frameworks. A bridge acts as an intermediary between the nGraph core and the 
framework, and the result is a function that can be compiled from a framework. 
A fully-compiled function that makes use of bridge code thus becomes a "function 
graph", or what we sometimes call an nGraph graph.

For a more detailed dive into how custom bridge code can be implemented, see our 
documentation [How to]. To learn how TensorFlow and MXNet currently make use of 
custom bridge code, see the section on [framework-integration-guides].

:note-caption: **Important** -- Intel's compilers may or may not optimize to the 
same degree for non-Intel microprocessors for optimizations that are not unique 
to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 
instruction sets and other optimizations. Intel does not guarantee the availability, 
functionality, or effectiveness of any optimization on microprocessors not 
manufactured by Intel. Microprocessor-dependent optimizations in this product 
are intended for use with Intel microprocessors. Certain optimizations not specific 
to Intel microarchitecture  are reserved for Intel microprocessors. Please refer 
to the applicable product User and Reference Guides for more information regarding 
the specific instruction sets covered by this notice.






[Upcoming DL accelerators]: https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/vision-accelerator-design-product-brief.pdf
[import it]: http://ngraph.nervanasys.com/docs/latest/howto/import.html
[ONNXIFI]: https://github.com/onnx/onnx/blob/master/docs/ONNXIFI.md
[ONNX Runtime]:https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx
[WinML]: http://docs.microsoft.com/en-us/windows/ai
[How to]: https://ngraph.nervanasys.com/docs/latest/howto/index.html
[framework-integration-guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
