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




// TEST(UnaryOpTest, ElementwiseOneHotEncoding)
// {
//     // Step 1: Initialize argon and point batch start with id zero.
//     ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
//     arBegin(BLK_MINIBATCH, 0);

//     // Step 2: Prepare three argon tensors.
//     //         Important is to preserve their id.
//     ArTensorId Aid, Bid;
//     ArApiShape shapeA{1, 5};
//     ArApiShape shapeB{5, 5};

//     // Note: You can check api calls parameters in argon_api.h
//     arCreateTensor(shapeA, 0, 1, 0, &Aid);
//     arCreateTensor(shapeB, 0, 1, 0, &Bid);
    

// 	std::vector<float> in_data  = { 1, 2, 3, 4, 5 };


//     // // Step 3: Fill A and B tensors with ones.
//     // arSelf(AR_SELF_EW_FILL, Aid, 1.0f);
// 	size_t arry_size = 5;
//     arCopyFromArray(Aid, in_data.data(), arry_size * sizeof(float));

//     // // Step 4: Do element wise addition of A and B tensors.
//     arUnary(AR_UNARY_EW_ONEHOT, Aid, Bid);

//     // // Step 5: Get results.
//     size_t size = shapeA.rows_ * shapeA.cols_;
//     std::unique_ptr<float[]> array(new float[size]);
//     arCopyToArray(Aid, array.get(), size * sizeof(float), true);

//     // // Step 6: Check if results are correct.
//     for (size_t i = 0; i < size; i++)
//     {
//         EXPECT_EQ(array[i], (int(i)+1) )<< "one hot encoding condition  message"; 
//     }

//     // Step 7: Delete tensors.
//     arDeleteTensor(Aid);
//     arDeleteTensor(Bid);

//     // Step 8: Point batch end and finalize argon.
//     arEnd(BLK_MINIBATCH, 0);
//     arFinalize();
// }


// TEST(UnaryOpTest, ElementwiseSquare)
// {
//     // Step 1: Initialize argon and point batch start with id zero.
//     ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
//     arBegin(BLK_MINIBATCH, 0);

//     // Step 2: Prepare three argon tensors.
//     //         Important is to preserve their id.
//     ArTensorId Aid, Bid;
//     ArApiShape shape{128, 128};
   

//     // Note: You can check api calls parameters in argon_api.h
//     arCreateTensor(shape, 0, 1, 0, &Aid);
//     arCreateTensor(shape, 0, 1, 0, &Bid);
    

//     // // Step 3: Fill A and B tensors with ones.
//     arSelf(AR_SELF_EW_FILL, Aid, 5.0f);
  
//     // // Step 4: Do element wise addition of A and B tensors.
//     arUnary(AR_UNARY_EW_SQUARE, Aid, Bid);

//     // // Step 5: Get results.
//     size_t size = shape.rows_ * shape.cols_;
//     std::unique_ptr<float[]> array(new float[size]);
//     arCopyToArray(Aid, array.get(), size * sizeof(float), true);

//     // // Step 6: Check if results are correct.
//     for (size_t i = 0; i < size; i++)
//     {
//         EXPECT_EQ(array[i], 25 )<< "Square condition  message"; 
//     }

//     // Step 7: Delete tensors.
//     arDeleteTensor(Aid);
//     arDeleteTensor(Bid);

//     // Step 8: Point batch end and finalize argon.
//     arEnd(BLK_MINIBATCH, 0);
//     arFinalize();
// }



TEST(TensorCopy, CopyArrayOrVectorToTensorAndBackToArray)
{
    // Step 1: Initialize argon and point batch start with id zero.
    ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
    arBegin(BLK_MINIBATCH, 0);

    // Step 2: Prepare three argon tensors.
    //         Important is to preserve their id.
    ArTensorId Aid;
    ArApiShape shapeA{5,1};

    // Note: You can check api calls parameters in argon_api.h
    arCreateTensor(shapeA, 0, 1, 0, &Aid);

    // create a vector or array
	std::vector<float> in_data  = { 1, 2, 3, 4, 5 };

	//copy vector to tensor
	size_t arry_size = 5;
    arCopyFromArray(Aid, in_data.data(), arry_size * sizeof(float));


    // // Step 5: Get results.
    size_t size = shapeA.rows_ * shapeA.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Aid, array.get(), size * sizeof(float), true);

    // // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], (int(i)+1) )<< "(Copy a single dimention array) condition  message"; 
    }

    // Step 7: Delete tensors.
    arDeleteTensor(Aid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}



TEST(UnaryOpTest, ElementwiseReLU)
{
    // Step 1: Initialize argon and point batch start with id zero.
    ASSERT_EQ(arInitialize(NRV_ARGON_API_VERSION_MAJOR, NRV_ARGON_API_VERSION_MINOR), 0);
    arBegin(BLK_MINIBATCH, 0);

    // Step 2: Prepare three argon tensors.
    //         Important is to preserve their id.
    ArTensorId Aid, Bid;
    
    ArApiShape shape{5, 1};

    // Note: You can check api calls parameters in argon_api.h
    arCreateTensor(shape, 0, 1, 0, &Aid);
    arCreateTensor(shape, 0, 1, 0, &Bid);
    
    //***************condition 
    //*************** RelU value > 0 
    //*************************

    // fill the tensor with positive values 
    std::vector<float> in_data  = { 0.5, 1.5, 2.5, 3.5, 4.5 };
    size_t arry_size = 5;
    arCopyFromArray(Aid, in_data.data(), arry_size * sizeof(float));
	
    // // Step 4: Do element wise addition of A and B tensors.
    arUnary(AR_UNARY_EW_RELU, Aid, Bid);

    // // Step 5: Get results.
    size_t size = shape.rows_ * shape.cols_;
    std::unique_ptr<float[]> array(new float[size]);
    arCopyToArray(Aid, array.get(), size * sizeof(float), true);

    // // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], (int(i)+0.5) )<< "(rRelU value > 0 ) condition  message"; 
    }

    //***************condition 
    //*************** RelU value <= 0 
    //*************************

    in_data  = { -1, -2, -3, -4, 0 };

	//copy vector to tensor
	arry_size = 5;
    arCopyFromArray(Aid, in_data.data(), arry_size * sizeof(float));

    // // Step 4: Do element wise addition of A and B tensors.
    arUnary(AR_UNARY_EW_RELU, Aid, Bid);

    // Step 5: Get results.
    size = shape.rows_ * shape.cols_;
    arCopyToArray(Bid, array.get(), size * sizeof(float), true);

    // Step 6: Check if results are correct.
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(array[i], 0.0f)<< "(rRelU value <= 0 ) condition  message"; 
    }



    // Step 7: Delete tensors.
    arDeleteTensor(Aid);
    arDeleteTensor(Bid);

    // Step 8: Point batch end and finalize argon.
    arEnd(BLK_MINIBATCH, 0);
    arFinalize();
}


