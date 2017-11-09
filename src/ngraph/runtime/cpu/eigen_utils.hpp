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
#include "ngraph/runtime/cpu/call_frame.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        class TensorViewInfo;

        namespace cpu
        {
            class CallFrame;

            namespace eigen
            {
                using DynamicStrides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
                using VectorStrides = Eigen::Stride<Eigen::Dynamic, 1>;

                template <typename T>
                using DynamicArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

                template <typename T>
                using EigenArrayBase = Eigen::Map<DynamicArray<T>, 0, DynamicStrides>;

                template <typename T>
                using DynamicMatrix =
                    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

                template <typename T>
                using EigenMatrixBase = Eigen::Map<DynamicMatrix<T>, 0, DynamicStrides>;

                template <typename T>
                using DynamicVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

                template <typename T>
                using EigenVectorBase = Eigen::Map<DynamicVector<T>, 0, VectorStrides>;

                namespace fmt
                {
                    /// @brief vector format for Eigen wrappers.
                    class V
                    {
                    public:
                        V(const TensorViewInfo& tensor_view_info)
                            : l0(tensor_view_info
                                     .get_layout<
                                         ngraph::descriptor::layout::DenseTensorViewLayout>()
                                     ->get_size())
                        {
                        }

                        V(size_t s)
                            : l0(s)
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
                        M(const std::shared_ptr<ngraph::descriptor::layout::DenseTensorViewLayout>&
                              layout)
                            : M(layout->get_shape(), layout->get_strides())
                        {
                        }

                    public:
                        M(const Shape& shape, const Strides& strides)
                            : l0(shape.at(0))
                            , l1(shape.at(1))
                            , s0(strides.at(0))
                            , s1(strides.at(1))
                        {
                        }

                        M(const TensorViewInfo& tensor_view_info)
                            : M(tensor_view_info.get_layout<
                                  ngraph::descriptor::layout::DenseTensorViewLayout>())
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
                template <typename T,
                          typename FMT,
                          typename BASE,
                          typename STRIDES = DynamicStrides>
                class EigenWrapper : public BASE
                {
                    using base = BASE;

                public:
                    EigenWrapper(T* t, const FMT& fmt)
                        : base(t, fmt.l0, fmt.l1, STRIDES(fmt.s0, fmt.s1))
                    {
                    }

                    EigenWrapper(
                        T* t,
                        const std::shared_ptr<ngraph::descriptor::layout::DenseTensorViewLayout>&
                            layout)
                        : base(t, layout->get_size(), 1, DynamicStrides(1, 1))
                    {
                    }

                    template <typename U>
                    EigenWrapper& operator=(const U& other)
                    {
                        this->base::operator=(other);
                        return *this;
                    }
                };

                template <typename T, typename FMT = fmt::V>
                using EigenArray1d = EigenWrapper<T, FMT, EigenArrayBase<T>>;

                template <typename T, typename FMT = fmt::M>
                using EigenArray2d = EigenWrapper<T, FMT, EigenArrayBase<T>>;

                template <typename T, typename FMT = fmt::M>
                using EigenMatrix = EigenWrapper<T, FMT, EigenMatrixBase<T>>;

                template <typename T, typename FMT = fmt::V>
                using EigenVector = EigenWrapper<T, FMT, EigenVectorBase<T>, VectorStrides>;
            }
        }
    }
}
