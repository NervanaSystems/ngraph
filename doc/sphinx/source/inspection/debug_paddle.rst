.. inspection/debug_paddle.rst: 

.. _debug_paddle:

Debug PaddlePaddle\*
====================

.. note:: In addition to the compile flags below, PaddlePaddle has its `own env vars`_. 


.. csv-table:: 
   :header: "Flag"
   :widths: 20
   :escape: ~

   ``check_nan_inf``
   ``fast_check_nan_inf``
   ``benchmark``
   ``eager_delete_scope``
   ``fraction_of_cpu_memory_to_use``
   ``initial_cpu_memory_in_mb``
   ``init_allocated_mem``
   ``paddle_num_threads``
   ``dist_threadpool_size``
   ``eager_delete_tensor_gb``
   ``fast_eager_deletion_mode``
   ``memory_fraction_of_eager_deletion``
   ``allocator_strategy``
   ``reader_queue_speed_test_mode``
   ``print_sub_graph_dir``
   ``pe_profile_fname``
   ``inner_op_parallelism``
   ``enable_parallel_graph``
   ``fuse_parameter_groups_size``
   ``multiple_of_cupti_buffer_size``
   ``fuse_parameter_memory_size``
   ``tracer_profile_fname``
   ``dygraph_debug``
   ``use_system_allocator``
   ``enable_unused_var_check``


.. _own env vars: https://github.com/PaddlePaddle/Paddle/blob/cdd46d7e022add8de56995e681fa807982b02124/python/paddle/fluid/__init__.py#L161-L227