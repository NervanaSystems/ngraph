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

#include <memory>

#include <Eigen/Dense>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        class TensorViewInfo;
        class CallFrame;

        namespace eigen
        {
            using DynamicStrides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
            using VectorStrides  = Eigen::Stride<Eigen::Dynamic, 1>;

            template <typename ET>
            using DynamicArray = Eigen::Array<typename ET::type, Eigen::Dynamic, Eigen::Dynamic>;

            template <typename ET>
            using EigenArrayBase = Eigen::Map<DynamicArray<ET>, 0, DynamicStrides>;

            template <typename ET>
            using DynamicMatrix = Eigen::Matrix<typename ET::type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            template <typename ET>
            using EigenMatrixBase = Eigen::Map<DynamicMatrix<ET>, 0, DynamicStrides>;

            template <typename ET>
            using DynamicVector = Eigen::Matrix<typename ET::type, Eigen::Dynamic, 1>;

            template <typename ET>
            using EigenVectorBase = Eigen::Map<DynamicVector<ET>, 0, VectorStrides>;

            namespace fmt
            {
                /// @brief vector format for Eigen wrappers.
                class V
                {
                public:
                    V(const TensorViewInfo& tensor_view_info)
                        : l0(tensor_view_info
                                 .get_layout<ngraph::descriptor::layout::DenseTensorViewLayout>()
                                 ->get_size())
                    {
                    }

                public:
                    size_t l0;
                    size_t l1{1};
                    size_t s0{1};
                    size_t s1{1};
                };

                class M
                {
                    M(const Shape& shape, const Strides& strides)
                        : l0(shape.at(0))
                        , l1(shape.at(1))
                        , s0(strides.at(0))
                        , s1(strides.at(1))
                    {
                    }

                    M(const std::shared_ptr<ngraph::descriptor::layout::DenseTensorViewLayout>&
                          layout)
                        : M(layout->get_shape(), layout->get_strides())
                    {
                    }

                public:
                    M(const TensorViewInfo& tensor_view_info)
                        : M(tensor_view_info
                                .get_layout<ngraph::descriptor::layout::DenseTensorViewLayout>())
                    {
                    }

                public:
                    size_t l0;
                    size_t l1;
                    size_t s0;
                    size_t s1;
                };
            }

            // ET element type
            // FMT array format (fmt::V for vector, etc.)
            // BASE select array/matrix
            template <typename ET, typename FMT, typename BASE, typename STRIDES = DynamicStrides>
            class EigenWrapper : public BASE
            {
                using base = BASE;

            public:
                EigenWrapper(typename ET::type* t, const FMT& fmt)
                    : base(t, fmt.l0, fmt.l1, STRIDES(fmt.s0, fmt.s1))
                {
                }

                EigenWrapper(
                    typename ET::type* t,
                    const std::shared_ptr<ngraph::descriptor::layout::DenseTensorViewLayout>&
                        layout)
                    : base(t, layout->get_size(), 1, DynamicStrides(1, 1))
                {
                }

                EigenWrapper(CallFrame& call_frame, const TensorViewInfo& tensor_view_info)
                    : EigenWrapper(
                          call_frame.get_tensor_view_data<ET>(tensor_view_info.get_index()),
                          FMT(tensor_view_info))
                {
                }

                template <typename U>
                EigenWrapper& operator=(const U& other)
                {
                    this->base::operator=(other);
                    return *this;
                }
            };

            template <typename ET, typename FMT = fmt::V>
            using EigenArray = EigenWrapper<ET, FMT, EigenArrayBase<ET>>;

            template <typename ET, typename FMT = fmt::M>
            using EigenMatrix = EigenWrapper<ET, FMT, EigenMatrixBase<ET>>;

            template <typename ET, typename FMT = fmt::V>
            using EigenVector = EigenWrapper<ET, FMT, EigenVectorBase<ET>, VectorStrides>;

            template <typename T, typename U>
            void set_map_array(T* t, size_t l0, size_t l1, size_t s0, size_t s1, const U& u)
            {
                Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
                           0,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
                    t, l0, l1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(s0, s1)) = u;
            }

            template <typename T, typename U>
            void set_map_array(T* t, size_t l0, size_t s0, const U& u)
            {
                Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>,
                           0,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
                    t, l0, 1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(s0, 1)) = u;
            }

            template <typename T>
            Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
                       0,
                       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                get_map_array(T* t, size_t l0, size_t l1, size_t s0, size_t s1)
            {
                return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
                                  0,
                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
                    t, l0, l1, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(s0, s1));
            }

            template <typename T>
            Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
                       0,
                       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                get_map_array(T* t, size_t l0, size_t s0)
            {
                return Eigen::
                    Map<Eigen::Array<T, Eigen::Dynamic, 1>, 0, Eigen::Stride<Eigen::Dynamic, 1>>(
                        t, l0, 1, Eigen::Stride<Eigen::Dynamic, 1>(s0, 1));
            }

            template <typename T, typename U>
            void set_map_array(std::shared_ptr<T>& t, const U& u)
            {
                auto& v = t->get_vector();
                set_map_array(&v[0], v.size(), 1, u);
            }

            template <typename T, typename U>
            void set_map_array(T* t, const U& u)
            {
                auto& v = t->get_vector();
                set_map_array(&v[0], v.size(), 1, u);
            }

            template <typename T, typename U>
            void set_map_matrix(std::shared_ptr<T>& t, const U& u)
            {
                auto& v = t->get_vector();
                Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>(
                    &v[0], v.size(), 1) = u;
            }

            template <typename T, typename U>
            void set_map_matrix(T* t, const U& u)
            {
                auto& v = t->get_vector();
                Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>(
                    &v[0], v.size(), 1) = u;
            }

            template <typename T, typename U>
            void set_map_array_2d(std::shared_ptr<T>& t, const U& u)
            {
                auto& v      = t->get_vector();
                auto& s      = t->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                Eigen::Map<Eigen::Array<typename T::value_type,
                                        Eigen::Dynamic,
                                        Eigen::Dynamic,
                                        Eigen::RowMajor>>(&v[0], s[0], ngraph::shape_size(s_rest)) =
                    u;
            }

            template <typename T, typename U>
            void set_map_array_2d(T* t, const U& u)
            {
                auto& v      = t->get_vector();
                auto& s      = t->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                Eigen::Map<Eigen::Array<typename T::value_type,
                                        Eigen::Dynamic,
                                        Eigen::Dynamic,
                                        Eigen::RowMajor>>(&v[0], s[0], ngraph::shape_size(s_rest)) =
                    u;
            }

            template <typename T, typename U>
            void set_map_matrix_2d(std::shared_ptr<T>& t, const U& u)
            {
                auto& v      = t->get_vector();
                auto& s      = t->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                Eigen::Map<Eigen::Matrix<typename T::value_type,
                                         Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest)) = u;
            }

            template <typename T, typename U>
            void set_map_matrix_2d(T* t, const U& u)
            {
                auto& v      = t->get_vector();
                auto& s      = t->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                Eigen::Map<Eigen::Matrix<typename T::value_type,
                                         Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest)) = u;
            }

            template <typename T>
            Eigen::Map<Eigen::Array<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic>>
                get_map_array(std::shared_ptr<T>& arg)
            {
                auto& v = arg->get_vector();
                return get_map_array(&v[0], v.size(), 1, 1, 1);
            }

            template <typename T>
            Eigen::Map<Eigen::Array<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic>,
                       0,
                       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                get_map_array(T* arg)
            {
                auto& v = arg->get_vector();
                return get_map_array(&v[0], v.size(), 1, 1, 1);
            }

            template <typename T>
            Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>
                get_map_matrix(std::shared_ptr<T>& arg)
            {
                auto& v = arg->get_vector();
                return Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>(
                    &v[0], v.size(), 1);
            }

            template <typename T>
            Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>
                get_map_matrix(T* arg)
            {
                auto& v = arg->get_vector();
                return Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>>(
                    &v[0], v.size(), 1);
            }

            template <typename T>
            Eigen::Map<
                Eigen::
                    Array<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                get_map_array_2d(std::shared_ptr<T>& arg)
            {
                auto& v      = arg->get_vector();
                auto& s      = arg->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                return Eigen::Map<Eigen::Array<typename T::value_type,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic,
                                               Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest));
            }

            template <typename T>
            Eigen::Map<
                Eigen::
                    Array<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                get_map_array_2d(T* arg)
            {
                auto& v      = arg->get_vector();
                auto& s      = arg->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                return Eigen::Map<Eigen::Array<typename T::value_type,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic,
                                               Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest));
            }

            template <typename T>
            Eigen::Map<
                Eigen::
                    Matrix<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                get_map_array_2d(std::shared_ptr<T>& arg)
            {
                auto& v      = arg->get_vector();
                auto& s      = arg->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                return Eigen::Map<Eigen::Matrix<typename T::value_type,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic,
                                                Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest));
            }

            template <typename T>
            Eigen::Map<
                Eigen::
                    Matrix<typename T::value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                get_map_matrix_2d(T* arg)
            {
                auto& v      = arg->get_vector();
                auto& s      = arg->get_shape();
                auto  s_rest = std::vector<size_t>(s.begin() + 1, s.end());
                return Eigen::Map<Eigen::Matrix<typename T::value_type,
                                                Eigen::Dynamic,
                                                Eigen::Dynamic,
                                                Eigen::RowMajor>>(
                    &v[0], s[0], ngraph::shape_size(s_rest));
            }
        }
    }
}
