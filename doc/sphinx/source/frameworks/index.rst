.. frameworks/index: 

Framework Customizations   
########################

.. important:: This section contains articles for framework owners or developers
   who want to incorporate the nGraph Compiler stack and optimize for some 
   specific compute, runtime, or compile-time characteristic. 

.. toctree::
   :maxdepth: 1 

   generic-configs.rst
   genre-validation.rst
   testing-latency.rst


Data scientists and ML engineers may not initially know which framework is the 
"best" framework to use to start working on his or her problem set. While there 
are several to choose from, it can be daunting and time-consuming to scope the 
wide array of features and customization options offered by some of the more 
popular frameworks. One of the more common approaches is the three-pronged 
approach: 

#. First **find** a tested and working DL model that does something *similar* 
   to what the data scientist or ML engineer wants to do. To assist with this 
   stage, we've already provided several :doc:`genre-validation` examples 
   organized by framework and what we call *Genre of Deep Learning*.  
#. Next, **replicate** that result using well-known datasets to confirm that the 
   model does indeed work. To assist wtih this stage, we've released several  
   :doc:`pip installation options <../framework-integration-guides>` that can 
   be used to test a very simple example.
#. Finally, **modify** some aspect: add new datasets, or adjust an algorithm's 
   parameters to hone in on specifics that can better train, forecast, or predict 
   scenarios modeling the real-world problem.

   .. warning:: nGraph does not provide an interface for "users" of frameworks 
      (that is, we cannot dictate or control how Tensorflow* or MXNet* presents 
      outputs to users). Please keep in mind that designing and documenting 
      the :abbr:`User Interface (UI)` of step 3 above is entirely in the realm 
      of the framework owner or developer and beyond the scope of the nGraph 
      Compiler stack. Any framework can be designed to make direct use of 
      specific nGraph features. See the :doc:`generic-configs` doc for some 
      ideas.    

Enabling new genres of Deep Learning 
====================================

For framework architects or engineers who can't quite find what they need among 
the existing DL tools, we have incorporated several ways to build or optimize 
frameworks that are built-from-scratch, generic, "stock", or less widely-supported 
See :doc:`generic-configs`. 

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

How this has worked in creating the 
:doc:`the direct optimizations <../framework-integration-guides>` we've shared 
with the developer community, our engineering teams carefully 
`tune the workload to extract best performance`_ 
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
   changes can be attributed to multiple causes. Also watch out for the word "theoretical" in comparisons; actual performance should not be 
   compared to theoretical performance.     


.. _ngraph tensorflow bridge: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html#tensorflow
.. _tune the workload to extract best performance: https://ai.intel.com/accelerating-deep-learning-training-inference-system-level-optimizations
.. _a few small: https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi
.. _Movidius: https://www.movidius.com/