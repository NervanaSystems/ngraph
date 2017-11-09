#include <gtest/gtest.h>

#include <cuda.h>
#include <cudnn.h>

TEST(cudnn, simple)
{
    auto cudnn_version = cudnnGetVersion();
    EXPECT_FLOAT_EQ(cudnn_version, CUDNN_VERSION);
}
