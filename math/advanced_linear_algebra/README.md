## Determinant
Definition:
The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important properties about the matrix, such as whether it is invertible, the volume scaling factor of linear transformations represented by the matrix, and whether the matrix is singular or non-singular.

Calculation:
- For a 2 x 2 matrix:

        det [a, b, c, d] = ad - bc
- For a 3 x 3 matrix:

        det [a, b, c, d, e, f, g, h, i] = a(ei - fh) - b(di - fg) + c(dh - eg)
- For larger matrices, determinants are calculated using expansion by minors or row reduction methods.
# Minor
## Definition:
A minor of an element of a matrix is the determinant of the submatrix formed by deleting the row and column of that element.

Calculation:

For matrix A:

𝐴 = [𝑎, 𝑏, 𝑐, 𝑑, 𝑒, 𝑓, 𝑔, ℎ, 𝑖]


