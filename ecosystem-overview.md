
# Framework & runtime support

One of nGraph’s key features is framework neutrality. We currently support 
popular deep learning frameworks such as TensorFlow and MXNet with stable 
bridges to pass computational graphs to nGraph. Additionally nGraph 
Compiler has a functional bridge to PaddlePaddle. 
For these frameworks, we have successfully tested functionality with a few 
deep learning workloads, and we plan to bring stable support for them in the 
upcoming releases. 

To further promote framework neutrality, the nGraph team has been actively 
contributing to the ONNX project. Developers who already have a "trained" 
DNN (Deep Neural Network) model can use nGraph to bypass significant 
framework-based complexity and [import it] to test or run on targeted and 
efficient backends with our user-friendly Python-based API.

nGraph is also integrated as an execution provider for [ONNX Runtime], 
which is the first publicly available inference engine for ONNX.

The table below summarizes our current progress on supported frameworks. 
If you are an architect of a framework wishing to take advantage of speed 
and multi-device support of nGraph Compiler, please refer to [Framework integration guide] section.  


|  Framework & Runtime       | Supported          |  Validated 
|----------------------------|--------------------|-------------
| TensorFlow* 1.12           | :heavy_check_mark: |  :heavy_check_mark:
| MXNet* 1.3                 | :heavy_check_mark: |  :heavy_check_mark:
| ONNX 1.3                   | :heavy_check_mark: |  :heavy_check_mark:
| ONNX Runtime               | Functional         |  No
| PaddlePaddle               | Functional         |  No



## Hardware & backend support

The current release of nGraph primarily focuses on accelerating inference 
performance on CPU. However we are also working on adding support for more 
hardware and backends. As with the frameworks, we believe in providing 
freedom to AI developers to deploy their deep learning workloads to the 
desired hardware without a lock in. We currently have functioning backends 
for Intel, Nvidia*, and AMD* GPU either leveraging kernel libraries 
such as clDNN and cuDNN directly or utilizing PlaidML to compile for codegen 
and emit OpenCL, OpenGL, LLVM, Cuda, and Metal. Please refer to [Architecture 
and features] section to learn more about how we plan to take advantage of 
both solutions using hybrid transformer. We expect to have stable support for aformentioned GPUs
in the early second half of 2019. In the similar time frame, we plan 
to release multinode support. 

Additionally, we are excited about providing support for our upcoming deep learning 
accelerators such as NNP (Neural Network Processor) via nGraph compiler 
stack, and early adopters will be able test them in 2019.



| Backend                                       | Supported         
|-----------------------------------------------|-------------------
| Intel® Architecture CPU                       | :heavy_check_mark:               
| Intel® Architecture GPUs                      | Functional via clDNN and PlaidML      
| AMD* GPUs                                     | Functional via PlaidML                 
| Nvidia* GPUs                                  | Functional via cuDNN and PlaidML        
| Intel® Nervana™ Neural Network Processor (NNP)| Functional               
| Upcoming DL accelerators                      | Functional and will be announced in the near future       




[Architecture and features]: ./ABOUT.md
[Upcoming DL accelerators]: https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/vision-accelerator-design-product-brief.pdf
[import it]: https://ngraph.nervanasys.com/docs/latest/core/constructing-graphs/import.html
[ONNX Runtime]: https://azure.microsoft.com/en-us/blog/onnx-runtime-is-now-open-source/
[WinML]: http://docs.microsoft.com/en-us/windows/ai
[How to]: https://ngraph.nervanasys.com/docs/latest/howto/index.html
[Framework integration guide]: https://ngraph.nervanasys.com/docs/latest/frameworks/index.html
