// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>

namespace ngraph
{
    //================================================================================================
    // NameableValue
    //     An Axis labels a dimension of a tensor. The op-graph uses
    //     the identity of Axis objects to pair and specify dimensions in
    //     symbolic expressions. This system has several advantages over
    //     using the length and position of the axis as in other frameworks:
    //
    //     1) Convenience. The dimensions of tensors, which may be nested
    //     deep in a computation graph, can be specified without having to
    //     calculate their lengths.
    //
    //     2) Safety. Axis labels are analogous to types in general-purpose
    //     programming languages, allowing objects to interact only when
    //     they are permitted to do so in advance. In symbolic computation,
    //     this prevents interference between axes that happen to have the
    //     same lengths but are logically distinct, e.g. if the number of
    //     training examples and the number of input features are both 50.
    //
    //     TODO: Please add to the list...
    //
    //     Arguments:
    //         length: The length of the axis.
    //         batch: Whether the axis is a batch axis.
    //         recurrent: Whether the axis is a recurrent axis.
    //================================================================================================
    class NameableValue
    {
    public:
        //!-----------------------------------------------------------------------------------
        //! NameableValue
        //!    An object that can be named.
        //!
        //!    Arguments:
        //!        graph_label_type: A label that should be used when drawing the graph.  Defaults to
        //!            the class name.
        //!        name (str): The name of the object.
        //!        **kwargs: Parameters for related classes.
        //!
        //!    Attributes:
        //!        graph_label_type: A label that should be used when drawing the graph.
        //!        id: Unique id for this object.
        //!-----------------------------------------------------------------------------------
        NameableValue(const std::string& name,
                      const std::string& graph_label_type = "",
                      const std::string& doc_string       = "");

        //!-----------------------------------------------------------------------------------
        //! graph_label
        //!    The label used for drawings of the graph.
        //!-----------------------------------------------------------------------------------
        const std::string& graph_label();

        //!-----------------------------------------------------------------------------------
        //! name
        //!    Sets the object name to a unique name based on name.
        //!
        //!    Arguments:
        //!        name: Prefix for the name
        //!-----------------------------------------------------------------------------------
        const std::string& name();

        //!-----------------------------------------------------------------------------------
        //! name
        //!-----------------------------------------------------------------------------------
        void name(const std::string& name);

        //!-----------------------------------------------------------------------------------
        //! short_name
        //!-----------------------------------------------------------------------------------
        const std::string& short_name();

        //!-----------------------------------------------------------------------------------
        //! named
        //!-----------------------------------------------------------------------------------
        NameableValue& named(const std::string& name);

        static size_t                               __counter;
        static std::map<std::string, NameableValue> __all_names;

        std::string m_name;
        std::string m_graph_label;
        std::string m_short_name;
        std::string m_doc_string;
    };

} // end namespace ngraph
