.. inspection/debug_tf: 

.. _debug_tf:

Debug TensorFlow\*
==================

.. note:: These flags are all disabled by default

For profiling with TensorFlow\* and ``nbench``, see :ref:`nbench_tf`.

.. csv-table:: 
   :header: "Flag", "Description"
   :widths: 20, 35
   :escape: ~

   ``NGRAPH_ENABLE_SERIALIZE=1``,Generate nGraph-level serialized graphs
   ``NGRAPH_TF_VLOG_LEVEL=5``, Generate ngraph-tf logging info for different passes
   ``NGRAPH_TF_LOG_PLACEMENT=1``, Generate op placement log at stdout
   ``NGRAPH_TF_DUMP_CLUSTERS=1``, Dump Encapsulated TF Graphs formatted as ``NGRAPH_cluster_<cluster_num>``
   ``NGRAPH_TF_DUMP_GRAPHS=1``,"Dump TF graphs for different passes: precapture, capture, unmarked, marked, clustered, declustered, encapsulated"
   ``TF_CPP_MIN_VLOG_LEVEL=1``, Enable TF CPP logs
   ``NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1``, Dump graphs with final clusters assigned. Use this to view TF computation graph with colored nodes indicating clusters
   ``NGRAPH_TF_USE_LEGACY_EXECUTOR``, This flag will be obsolete soon.
