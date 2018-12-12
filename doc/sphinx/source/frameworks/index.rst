.. frameworks/index.rst: 

########################
Framework Customizations   
########################

While a :abbr:`Deep Learning (DL)`:term:`framework` is ultimately meant for end 
use by data scientists, or for deployment in cloud container environments, the 
nGraph Core and C++ Library is designed for the framework builders themselves. 
We invite anyone building new and novel frameworks to explore our highly modular 
stack of components that can be implemented in virtually limitless ways -- from 
creating a lightweight design that does "just one thing" extremely well to 
architecting a complex and robust framework design around an entirely new *genre* 
of deep learning application on new silicon -- technologies that are too brand 
new to be well-known.  

Read the articles in this section if you considering incorporating components 
from the nGraph Compiler stack in your framework. If you are working on something 
built-from-scratch, or on an existing framework that is less widely-supported 
than the popular frameworks like TensorFlow and PyTorch, see :doc:`generic-configs`.

.. toctree::
   :maxdepth: 1 

   generic-configs.rst
   testing-latency.rst
   validation.rst


Understanding users of frameworks
=================================

A data scientist or ML engineer may not initially know which framework is the 
"best" framework to use to start working on his or her problem set. While there 
are several to choose from, it can be daunting and time-consuming to scope the 
wide array of features and customization options offered by some of the more 
popular frameworks. One of the more common approaches is the three-pronged 
approach: 

#. First **find** a tested and working DL model that does something *similar* 
   to what the data scientist or ML engineer wants to do. To assist with this 
   stage, we've already provided several :doc:`validation` examples 
   organized by framework and what we call *Genre of Deep Learning*.  
#. Next, **replicate** that result using well-known datasets to confirm that the 
   model does indeed work. To assist with this stage, we've released several  
   :doc:`pip installation options <../framework-integration-guides>` that can 
   be used to test basic examples.
#. Finally, **modify** some aspect: add new datasets, or adjust an algorithm's 
   parameters to hone in on specifics that can better train, forecast, or predict 
   scenarios modeling the real-world problem.

   .. important:: nGraph does not provide an interface for "users" of frameworks 
      (for example, we cannot dictate or control how Tensorflow* or MXNet* presents 
      interfaces to users). Please keep in mind that designing and documenting 
      the :abbr:`User Interface (UI)` of step 3 above is entirely in the realm 
      of the framework owner or developer and beyond the scope of the nGraph 
      Compiler stack. However, any framework can be designed to make direct use 
      of specific nGraph features; see the :doc:`generic-configs` doc for some 
      ideas.

Clearly, one challenge of the framework developer, is to differentiate from 
the pack by providing a means for the data scientist to obtain reproducible 
results. The other challenge is to provide sufficient documentation, or to 
provide sufficient hints for how to do any "fine-tuning" for specific use cases. 
With the nGraph Compiler stack powering your framework, it becomes much easier 
to help your users get reproducible results with nothing more complex than the 
CPU that powers their Operating System.   

In general, the larger and more complex a framework is, the harder it becomes 
to navigate and extract the best performance; configuration options that are 
enabled by "default" from the framework side can sometimes slow down compilation 
without the developer being any the wiser. Sometimes only `a few small`_ 
adjustments can increase performance. Likewise, a minimalistic framework that 
is designed around one specific kind of model can sometimes offer significant 
performance-improvement opportunities by lowering overhead. 

See :doc:`generic-configs` to get started.   


.. _ngraph tensorflow bridge: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html#tensorflow
.. _tune the workload to extract best performance: https://ai.intel.com/accelerating-deep-learning-training-inference-system-level-optimizations
.. _a few small: https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi
