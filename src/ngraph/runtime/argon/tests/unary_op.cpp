#include <memory>
#include "gtest/gtest.h"

#include "argon_api.h"

TEST(UnaryOpTest, ElementwiseNotOfTensor)
{
    // Step 1: Initialize argon and point batch start with id zero.
    ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
    arBegin(BLK_MINIBATCH, 0);

    // Step 2: Prepare three argon tensors.
    //         Important is to preserve their id.
    ArTensorId Aid, Bid;
    ArApiShape shape{128, 64};
    // Note: You can check api calls parameters in argon_api.h
    arCreateTensor(shape, 0, 1, 0, &Aid);
    arCreateTensor(shape, 0, 1, 0, &Bid);

    //***************condition 
    //*************** Not 1
    //*************************

    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, Aid, 1.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arUnary(AR_UNARY_EW_NOT, Aid, Bid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Bid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 0.0f)<< "(NOT of 1) condition  message"; 
    }



    //***************condition 
    //*************** Not 0
    //*************************

    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, Aid, 0.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arUnary(AR_UNARY_EW_NOT, Aid, Bid);

    // Step 5: Get results.
    size = shape.rows_ * shape.cols_;
    arCopyToArray(Bid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 1.0f)<< "(NOT of 0) condition  message"; 
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}