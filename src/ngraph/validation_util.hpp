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

#include <memory>
#include <tuple>
#include <typeindex>
#include <unordered_map>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        /// An op type specific validator implementation.
        /// A users op validator inherits from this specialized
        /// over the op type
        template <class OP_T>
        class OpValidator
        {
        public:
            using op_t = OP_T;
            virtual ~OpValidator() {}
            virtual void validate() = 0;
            void set(OP_T* op) { node = op; }
        protected:
            op_t* node = nullptr;
        };

        /// Non-templated base helper used for type
        /// erasure in the global validator map
        /// Method: validate - The virtual function implemented
        ///                    by each user for validation
        class OpValidationBase
        {
        public:
            virtual ~OpValidationBase() {}
            virtual void validate(ngraph::Node* node) = 0;
        };

        /// The validation helper which is templated over the
        /// user defined validator that inherits from OpValidator<OP_T>.
        /// Method: validate - Implemented here to cast and set the
        ///                    OpValidator<OP_T>::node and call the user's
        ///                    validator
        template <class OP_VALIDATOR>
        class OpValidationHelper : public OpValidationBase
        {
        public:
            using validator_t = OP_VALIDATOR;
            virtual void validate(ngraph::Node* node)
            {
                validator_t validator;
                validator.set(static_cast<typename validator_t::op_t*>(node));
                validator.validate();
            }
        };

        using OpValidatorMap =
            std::unordered_map<std::type_index, std::shared_ptr<OpValidationBase>>;
        /// Getter for the global validator map;
        /// The map utilizes std::type_index keys for each registered op type,
        /// and user defined validation functions for values.
        OpValidatorMap* get_validator_map();

        /// \brief Class to register the user defined op validator upon instantiation.
        ///        If an op would like to use the validator definition of a parent,
        ///        the optional PARENT_OP_VALIDATION_HELPER default parameter can be specified
        template <class OP_VALIDATION_HELPER,
                  class PARENT_OP_VALIDATION_HELPER = OP_VALIDATION_HELPER>
        class OpValidationHelperRegistration
        {
        public:
            OpValidationHelperRegistration()
            {
                get_validator_map()->operator[](
                    std::type_index(typeid(typename OP_VALIDATION_HELPER::validator_t::op_t))) =
                    std::shared_ptr<OpValidationBase>(new PARENT_OP_VALIDATION_HELPER());
            }
        };
    }
}

/// \brief Macro to register a user implementation of an op validator for a specific op type.
/// \param OpType The ngraph operator type
/// \param UserValidator The desired name of the user defined validation class
#define REGISTER_OP_VALIDATOR(OpType, UserValidator)                                               \
    class UserValidator : public OpValidator<OpType>                                               \
    {                                                                                              \
    public:                                                                                        \
        void validate();                                                                           \
    };                                                                                             \
    namespace                                                                                      \
    {                                                                                              \
        OpValidationHelperRegistration<OpValidationHelper<UserValidator>>                          \
            register_##UserValidator;                                                              \
    }

/// \brief Macro to reuse the user implementation of an op validator for a different op type.
/// \param OpType The ngraph operator type
/// \param ParentValidator The name of the user defined validation class to use in place of
///        a separately defined user validator for the given op type
/// \param UserValidator The desired name of the user defined validation class
#define INHERIT_OP_VALIDATOR(OpType, ParentValidator, UserValidator)                               \
    class UserValidator : public OpValidator<OpType>                                               \
    {                                                                                              \
    };                                                                                             \
    namespace                                                                                      \
    {                                                                                              \
        OpValidationHelperRegistration<OpValidationHelper<UserValidator>,                          \
                                       OpValidationHelper<ParentValidator>>                        \
            register_##UserValidator;                                                              \
    }

namespace ngraph
{
    Strides conv_default_strides(const Node* node,
                                 const PartialShape& data_batch_shape,
                                 const PartialShape& filters_shape);

    CoordinateDiff conv_default_padding(const Node* node,
                                        const PartialShape& data_batch_shape,
                                        const PartialShape& filters_shape);

    PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                       const PartialShape& data_shape,
                                                       const Strides& data_dilation,
                                                       const CoordinateDiff& data_padding_below,
                                                       const CoordinateDiff& data_padding_above,
                                                       const PartialShape& window_shape,
                                                       const Strides& window_strides,
                                                       const Strides& window_dilation,
                                                       bool is_window_all_in_padding_allowed,
                                                       bool ceil_mode = false);

    std::tuple<element::Type, PartialShape>
        infer_convolution_forward(const Node* node,
                                  element::Type et_batch,
                                  element::Type et_filters,
                                  const PartialShape& data_batch_shape,
                                  const Strides& data_dilation,
                                  const CoordinateDiff& data_padding_below,
                                  const CoordinateDiff& data_padding_above,
                                  const PartialShape& filters_shape,
                                  const Strides& filter_strides,
                                  const Strides& filter_dilation);

    PartialShape infer_batched_pooling_forward(const Node* node,
                                               const PartialShape& data_batch_shape,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const PartialShape& window_shape,
                                               const Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode = false);

    std::tuple<element::Type, PartialShape, PartialShape>
        infer_batch_norm_forward(const Node* node,
                                 element::Type input_element_type,
                                 element::Type gamma_element_type,
                                 element::Type beta_element_type,
                                 element::Type mean_element_type,
                                 element::Type variance_element_type,
                                 const PartialShape& input_shape,
                                 const PartialShape& gamma_shape,
                                 const PartialShape& beta_shape,
                                 const PartialShape& mean_shape,
                                 const PartialShape& variance_shape);

    std::tuple<element::Type, PartialShape, PartialShape>
        infer_batch_norm_forward(const Node* node,
                                 element::Type input_element_type,
                                 element::Type gamma_element_type,
                                 element::Type beta_element_type,
                                 const PartialShape& input_shape,
                                 const PartialShape& gamma_shape,
                                 const PartialShape& beta_shape);

    void infer_auto_padding(const Shape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below);
}
