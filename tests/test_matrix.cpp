#include <gtest/gtest.h>

#include "matrix.h"

TEST(MatrixTest, InitializesToZero)
{
    MiniNeuron::Matrix matrix(3, 4);

    for (float value : matrix.data)
    {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(MatrixTest, CreatesCorrectDimensions)
{
    MiniNeuron::Matrix matrix(3, 4);

    EXPECT_EQ(matrix.rows, 3);
    EXPECT_EQ(matrix.cols, 4);
    EXPECT_EQ(matrix.data.size(), 12);
}

TEST(MatrixTest, CanWriteAndReadElements)
{
    MiniNeuron::Matrix matrix(3, 4);

    matrix(0, 0) = 1.0f;
    matrix(1, 2) = 42.0f;
    matrix(2, 3) = -5.5f;

    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(matrix(1, 2), 42.0f);
    EXPECT_FLOAT_EQ(matrix(2, 3), -5.5f);
}

TEST(MatrixTest, StoresElementsInCorrectPositions)
{
    MiniNeuron::Matrix matrix(2, 3);

    matrix(0, 0) = 1.0f;
    matrix(0, 1) = 2.0f;
    matrix(0, 2) = 3.0f;

    matrix(1, 0) = 4.0f;
    matrix(1, 1) = 5.0f;
    matrix(1, 2) = 6.0f;

    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(matrix(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(matrix(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(matrix(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(matrix(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(matrix(1, 2), 6.0f);
}

TEST(MatrixTest, CanAccessRow)
{
    MiniNeuron::Matrix matrix(3, 4);

    matrix(1, 0) = 10.0f;
    matrix(1, 1) = 20.0f;
    matrix(1, 2) = 30.0f;
    matrix(1, 3) = 40.0f;

    float* row = matrix.row(1);

    EXPECT_FLOAT_EQ(row[0], 10.0f);
    EXPECT_FLOAT_EQ(row[1], 20.0f);
    EXPECT_FLOAT_EQ(row[2], 30.0f);
    EXPECT_FLOAT_EQ(row[3], 40.0f);
}

TEST(MatrixTest, CanModifyThroughRow)
{
    MiniNeuron::Matrix matrix(2, 3);

    float* row = matrix.row(1);

    row[0] = 10.0f;
    row[1] = 20.0f;
    row[2] = 30.0f;

    EXPECT_FLOAT_EQ(matrix(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(matrix(1, 1), 20.0f);
    EXPECT_FLOAT_EQ(matrix(1, 2), 30.0f);
}

TEST(MatrixTest, SupportsConstAccess)
{
    MiniNeuron::Matrix matrix(2, 2);

    matrix(0, 0) = 42.0f;

    const MiniNeuron::Matrix& constMatrix = matrix;

    EXPECT_FLOAT_EQ(constMatrix(0, 0), 42.0f);
}

TEST(MatrixTest, SupportsConstRowAccess)
{
    MiniNeuron::Matrix matrix(2, 2);

    matrix(1, 0) = 42.0f;

    const MiniNeuron::Matrix& constMatrix = matrix;

    const float* row = constMatrix.row(1);

    EXPECT_FLOAT_EQ(row[0], 42.0f);
}

TEST(MatrixTest, ClearSetsAllValuesToZero)
{
    MiniNeuron::Matrix matrix(2, 3);

    matrix(0, 0) = 1.0f;
    matrix(0, 1) = 2.0f;
    matrix(0, 2) = 3.0f;
    matrix(1, 0) = 4.0f;
    matrix(1, 1) = 5.0f;
    matrix(1, 2) = 6.0f;

    matrix.clear();

    for (float value : matrix.data)
    {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(MatrixTest, DefaultConstructorCreatesEmptyMatrix)
{
    MiniNeuron::Matrix matrix;

    EXPECT_EQ(matrix.rows, 0);
    EXPECT_EQ(matrix.cols, 0);
    EXPECT_TRUE(matrix.data.empty());
}