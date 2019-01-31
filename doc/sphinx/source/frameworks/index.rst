.. frameworks/index.rst: 

#################
Framework Support   
#################

While a :abbr:`Deep Learning (DL)` :term:`framework` is ultimately meant for 
end use by data scientists, or for deployment in cloud container environments, 
nGraph Core ops and the nGraph C++ Library are designed for framework builders 
themselves. We invite anyone working on new and novel frameworks or neural 
network designs to explore our highly-modular stack of components that can 
be implemented or integrated in virtually limitless ways.

Please read the articles in this section if you are considering incorporating 
components from the nGraph Compiler stack in your framework or neural network 
design. Articles here are also useful if you are working on something 
built-from-scratch, or on an existing framework that is less widely-supported 
than the popular frameworks like TensorFlow and PyTorch. 


.. toctree::
   :maxdepth: 1

   ../framework-integration-guides.rst
   validation.rst
   generic-configs.rst
   testing-latency.rst



Understanding users of frameworks
=================================

A data scientist or ML engineer may not initially know which framework is the 
"best" framework to use to start working on his or her problem set. While there 
are several to choose from, it can be daunting and time consuming to scope the 
wide array of features and customization options offered by some of the more 
popular frameworks:

#. First **find** a tested and working DL model that does something *similar* 
   to what the data scientist or ML engineer wants to do. To assist with this 
   stage, we've already provided organized tables of :doc:`validation` examples.
#. Next, **replicate** that result using well-known datasets to confirm that the 
   model does indeed work. To assist with this stage, we've released several  
   :doc:`pip installation options <../framework-integration-guides>` that can 
   be used to test basic examples.
#. Finally, **modify** some aspect: add new datasets, or adjust an algorithm's 
   parameters to hone in on specifics that can better train, forecast, or predict 
   scenarios modeling the real-world problem. This is also the stage where it 
   makes sense to `tune the workload to extract best performance`_.

   .. important:: nGraph does not provide an interface for "users" of frameworks 
      (for example, we cannot dictate or control how Tensorflow* or MXNet* presents 
      interfaces to users). Please keep in mind that designing and documenting 
      the :abbr:`User Interface (UI)` is entirely in the realm of the framework owner 
      or developer and beyond the scope of the nGraph Compiler stack. However, any 
      framework can be designed to make direct use of nGraph Compiler stack-based 
      features and then expose an accompanying UI, output message, or other detail 
      to a user.

Clearly, one challenge of the framework developer is to differentiate from 
the pack by providing a means for the data scientist to obtain reproducible 
results. The other challenge is to provide sufficient documentation, or to 
provide sufficient hints for how to do any "fine-tuning" for specific use cases. 
With the nGraph Compiler stack powering your framework, it becomes much easier 
to help your users get reproducible results with nothing more complex than the 
CPU that powers their operating system.   

In general, the larger and more complex a framework is, the harder it becomes 
to navigate and extract the best performance; configuration options that are 
enabled by "default" from the framework side can sometimes slow down compilation 
without the developer being any the wiser. Sometimes only `a few small`_ 
adjustments can increase performance. Likewise, a minimalistic framework that 
is designed around one specific kind of model can sometimes offer significant 
performance-improvement opportunities by lowering overhead. 

See :doc:`generic-configs` to get started.   


.. _tune the workload to extract best performance: https://ai.intel.com/accelerating-deep-learning-training-inference-system-level-optimizations
.. _a few small: https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi
