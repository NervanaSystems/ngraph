//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/function.hpp"

namespace ngraph
{
    /// \brief Creates a clone of a function, with the shapes of that function's parameters
    ///        specialized to some more specific element types and shapes.
    /// \param f The function to be cloned.
    /// \param parameter_element_types The new parameter element types to substitute.
    /// \param parameter_shapes The new parameter shapes to substitute.
    /// \return A clone of f, with the parameter element types and shapes specialized.
    /// \throws AssertionFailure if parameter_element_types or parameter_shapes is not valid
    ///         (see details).
    /// \throws NodeValidationError if node validation fails as the clone is being constructed.
    ///
    /// It is required that:
    ///    1. The length of parameter_element_types and parameter_shapes is the same as the number
    ///       of f's parameters.
    ///    2. Each shape in parameter_shapes is a refinement of the shape of the corresponding
    ///       parameter of f. Roughly speaking, a shape s1 is said to "refine" s2 if s1 can be
    ///       obtained from s2 by filling in s2's question marks. See PartialShape::refines for
    ///       more details.
    ///    3. For all i, either the element type of fp_i is dynamic, or fp_i is the same as
    ///       parameter_element_types[i]. (Here fp_i is the ith parameter of f.)
    std::shared_ptr<Function>
        specialize_shapes(std::shared_ptr<Function> f,
                          const std::vector<element::Type>& parameter_element_types,
                          const std::vector<PartialShape>& parameter_shapes);
}
