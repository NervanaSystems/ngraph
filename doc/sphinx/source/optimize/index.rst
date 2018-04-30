.. optimize/index: 

#############################
Integrate Generic Frameworks   
#############################

This section, written for framework architects or engineers who want 
to optimize a generic, brand new or less widely-supported framework, we
provide some of our learnings from the work we've done in developing 
"framework direct optimizations (DO)" and custom bridge code, such as 
that for our `ngraph tensorflow bridge`_ code.

.. important:: This section contains articles for framework owners or developers
   who want to incorporate the nGraph library directly into their framework and 
   optimize for some specific compute-time characteristic. 


.. toctree::
   :maxdepth: 1 

   generic.rst


When using a framework to run a model or deploy an algorithm on nGraph 
devices, there are some additional configuration options that can be 
incorporated -- manually on the command line or via scripting -- to improve 
performance. Fine-tuning an nGraph-enabled device is as much of an art as it 
is a science; there are virtually limitless ways to do so. 

Since a framework is typically designed around some feature, such as fast 
training using image data, inference on a mobile device, or support for voice 
and speech pattern recognition, a framework cannot optimize for all 
possibilities at the same time.   

In general, the larger and more complex a framework is, the harder it becomes 
to navigate and extract the best performance; configuration options that are 
enabled by "default" from the framework side can sometimes slow down compilation 
without the developer being any the wiser. Sometimes only `a few small`_ 
adjustments can increase performance. Likewise, a minimalistic framework that 
is designed around one specific kind of model can sometimes offer significant 
performance-improvement opportunities by lowering overhead. 

Right now the preferred way for a data scientist to get better performance is 
to shop around and select the framework that is "already" designed or optimized 
for some characteristic or trait of the model they want to build, test, tweak, 
or run. One challenge of the framework developer, then, is to differentiate from 
the pack by providing a means for the data scientist to obtain reproducible 
results. The other challenge is to provide sufficient documentation, or to 
provide sufficient hints for how to do any "fine-tuning" for specific use cases. 

How this has worked in creating the :doc:`the direct optimizations <../framework-integration-guides>` 
we've shared with the developer community, our `engineering teams carefully tune the workload to extract best performance`_ 
from a specific :abbr:`DL (Deep Learning)` model embedded in a specific framework 
that is training a specific dataset. Our forks of the frameworks adjust the code 
and/or explain how to set the parameters that achieve reproducible results. 

Some of the ways we attempt to improve performance include: 

* Testing and recording the results of various system-level configuration options
  or enabled or disabled flags,
* Compiling with a mix of custom environment variables, 
* Finding semi-related comparisons for benchmarking [#1]_, and 
* Tuning lower levels of the system so that the machine-learning algorithm can 
  learn faster or more accurately that it did on previous runs, 
* Incorporating various :doc:`../ops/index` to build graphs more efficiently. 

This approach, however, is obviously not a scalable solution for developers on  
the framework side who are trying to support multiple use cases. Nor is it ideal 
for teams looking to pivot or innovate multi-layer solutions based on something 
**other than training speed**, things like accuracy or precision. Chasing 
performance improvements does eventually yield a diminishing 
:abbr:`Return on Investment (ROI)`, though it is up to the framework 
developer to decide when that is for each of their customers.    

For these reasons, we're providing some of the more commonly-used options for 
fine-tuning various code deployments to the nGraph-enabled devices we 
currently support. Watch this section as we enable new devices and post new 
updates. 

.. rubric:: Footnotes

.. [#1] Benchmarking performance of DL systems is a young discipline; it is a
   good idea to be vigilant for results based on atypical distortions in the 
   configuration parameters. Every topology is different, and performance 
   increases or slowdowns can be attributed to multiple means.    


.. _ngraph tensorflow bridge: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html#tensorflow
.. _engineering teams carefully tune the workload to extract best performance: https://ai.intel.com/accelerating-deep-learning-training-inference-system-level-optimizations
.. _a few small: https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi
.. _Movidius: https://www.movidius.com/