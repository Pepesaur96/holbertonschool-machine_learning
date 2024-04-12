## What is a vector?

A vector is a one-dimensional array of numbers. It can represent various data or quantities, such as physical vectors (with magnitude and direction) in physics, or simply an array of features in machine learning. In programming, a vector often refers to a single row or column of data in an array.

## What is a matrix?

A matrix is a two-dimensional array of numbers arranged in rows and columns. It is a fundamental structure in linear algebra used to represent and manipulate data, perform transformations, solve linear systems, and more. In machine learning, matrices often represent datasets, where rows are samples and columns are features.

## What is a transpose?

The transpose of a matrix or vector is a new matrix or vector obtained by swapping the rows with columns. For a matrix, if A is an original matrix, A^T is the transpose. This operation changes its shape; for instance, if A is of shape m _ n, A^T will be of shape n _ m.

## What is the shape of a matrix?

The shape of a matrix refers to the dimensions of the matrix, given as a pair of numbers: (rows, columns). For example, a matrix with 3 rows and 2 columns has a shape of (3, 2).

## What is an axis?

In the context of multidimensional arrays, an axis refers to a specific dimension of the array. For a 2D array (matrix), axis 0 typically refers to the rows and axis 1 to the columns. In higher dimensions, each additional axis corresponds to a new dimension.

## What is a slice?

Slicing is the operation of extracting a portion of a dataset. For a vector, slicing could mean taking a subset of its elements. For a matrix, slicing could involve extracting a subset of rows, columns, or both, resulting in a smaller matrix or sub-array.

## How do you slice a vector/matrix?

Slicing a vector or matrix involves specifying the range of indices for the elements you want to extract. In Python, using NumPy or similar libraries, you can slice using the colon (:) operator. For example, vector[start:end] or matrix[start_row:end_row, start_column:end_column].

## What are element-wise operations?

Element-wise operations are operations applied to each element of an array individually. For example, adding two matrices of the same shape results in a new matrix where each element is the sum of the corresponding elements in the original matrices.

## How do you concatenate vectors/matrices?

Concatenation is the process of joining two or more vectors/matrices end-to-end along a specified axis. In NumPy, numpy.concatenate can concatenate arrays along a given axis, while numpy.vstack and numpy.hstack are convenient for vertical and horizontal stacking, respectively.

## What is the dot product?

The dot product is a scalar value obtained from two equal-length vectors. It's calculated by multiplying corresponding elements and summing the results. For vectors a and b, the dot product is Equivalent to the sum of a[i] \* b[i] for all i.

## What is matrix multiplication?

Matrix multiplication is a way to combine two matrices, where the element in the resulting matrix at row i, column j is the dot product of the ith row of the first matrix and the jth column of the second matrix. The operation requires the number of columns in the first matrix to equal the number of rows in the second.

## What is Numpy?

NumPy is a foundational library for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

## What is parallelization and why is it important?

Parallelization involves distributing a task across multiple processing elements to execute simultaneously, thereby reducing overall execution time. It's crucial for improving the performance of computational tasks, especially in data science and machine learning, where dealing with large datasets and complex algorithms is common.

## What is broadcasting?

Broadcasting is a technique used in libraries like NumPy to perform element-wise operations on arrays of different shapes. It allows for the automatic expansion of the smaller array along the larger array's dimensions, enabling efficient and concise operations without explicit replication of data.
