#include <memory>
#include "gtest/gtest.h"

#include "argon_api.h"

/*!
 * This test do element wise addition of two tensors in argon cpp api.
 * It can be starting point of learning how to use argon cpp api.
 */
TEST(SimpleEwTest, GIVENtwoTensorsWHENewAdditionTHENvalidResult)
{
    // Step 1: Initialize argon and point batch start with id zero.
    ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
    arBegin(BLK_MINIBATCH, 0);

    // Step 2: Prepare three argon tensors.
    //         Important is to preserve their id.
    ArTensorId Aid, Bid, Cid;
    ArApiShape shape{128, 64};
    // Note: You can check api calls parameters in argon_api.h
    arCreateTensor(shape, 0, 1, 0, &Aid);
    arCreateTensor(shape, 0, 1, 0, &Bid);
    arCreateTensor(shape, 0, 1, 0, &Cid);

    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, Aid, 1.0f);
    arSelf(AR_SELF_EW_FILL, Bid, 1.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_ADD, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 2.0f);
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}
