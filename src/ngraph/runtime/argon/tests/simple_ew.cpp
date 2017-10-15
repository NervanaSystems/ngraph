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

TEST(BinaryOpTest, DotProductOfTwoTensors)
{
    // Step 1: Initialize argon and point batch start with id zero.
    ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
    arBegin(BLK_MINIBATCH, 0);

    // Step 2: Prepare three argon tensors.
    //         Important is to preserve their id.
    ArTensorId A1id, B1id, C1id;
    ArApiShape shape{128, 128};
    // Note: You can check api calls parameters in argon_api.h
    arCreateTensor(shape, 0, 1, 0, &A1id);
    arCreateTensor(shape, 0, 1, 0, &B1id);
    arCreateTensor(shape, 0, 1, 0, &C1id);


    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, A1id, 1.0f);
    arSelf(AR_SELF_EW_FILL, B1id, 1.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_MAT_MUL, A1id, B1id, C1id);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(C1id, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 128.0f );
    }

    // Step 7: Delete tensors.
    arDeleteTensor(A1id);
    arDeleteTensor(B1id);
    arDeleteTensor(C1id);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}

TEST(BinaryOpTest, ElementwiseMultiplicationOfTwoTensors)
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
    arSelf(AR_SELF_EW_FILL, Bid, 5.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_MUL, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 5.0f);
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}


TEST(BinaryOpTest, ElementwiseSubtractionOfTwoTensors)
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
    arSelf(AR_SELF_EW_FILL, Bid, 5.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_SUB, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], (-4.0f) );
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}


TEST(BinaryOpTest, ElementwiseOrOperationTwoTensors)
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

    //***************condition 
    //***************1 OR 0 
    //*************************

    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, Aid, 1.0f);
    arSelf(AR_SELF_EW_FILL, Bid, 0.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_OR, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], (1.0f) )<< "OR (1 OR 0) condition  message"; 
    }

    //***************condition 
    //***************O OR 0 
    //*************************

    // Step 3: Fill A and B tensors with ones.
    arSelf(AR_SELF_EW_FILL, Aid, 0.0f);
    arSelf(AR_SELF_EW_FILL, Bid, 0.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_OR, Aid, Bid, Cid);

    // Step 5: Get results.
    size = shape.rows_ * shape.cols_;
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], (0.0f) )<< "OR (0 OR 0) condition  message"; 
    }


    // //***************condition 
    // //***************1 OR 1 
    // //*************************

    // // Step 3: Fill A and B tensors with ones.
    // arSelf(AR_SELF_EW_FILL, Aid, 1);
    // arSelf(AR_SELF_EW_FILL, Bid, 1);

    // // Step 4: Do element wise addition of A and B tensors.
    // arBinary(AR_BINARY_EW_AND, Aid, Bid, Cid);

    // // Step 5: Get results.
    // size = shape.rows_ * shape.cols_;
    // arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // // Step 6: Check if results are correct.
    // for (size_t i = 0; i < size; i++)
    // {
    //     EXPECT_EQ(array[i], (1) )<< "OR (1 OR 1) condition  message";
    // }

    
    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}



TEST(BinaryOpTest, ElementwiseDivideTwoTensors)
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
    arSelf(AR_SELF_EW_FILL, Aid, 30.0f);
    arSelf(AR_SELF_EW_FILL, Bid, 5.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_DIV, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 6.0f );
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}

TEST(BinaryOpTest, ElementwiseEXOROfTwoTensors)
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
    arSelf(AR_SELF_EW_FILL, Bid, 0.0f);

    // Step 4: Do element wise addition of A and B tensors.
    arBinary(AR_BINARY_EW_XOR, Aid, Bid, Cid);

    // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Cid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 1.0f );
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);
    arDeleteTensor(Cid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}





