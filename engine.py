class utils:
    def format(num, precision=8):
        res = str(round(num, precision))
        if "." in res:
            res = res.rstrip("0").rstrip(".")
        if res == "-0":
            res = "0"
        return res

    def solve_system_inverse(mat):
        if not isinstance(mat, matrix):
            raise TypeError("can only solve augmented n*n matrix systems")

        if mat.cols != mat.rows + 1:
            raise ValueError(
                "to solve a system of equations you must provide an augmented n*n matrix"
            )

        squared = mat.square()
        determinant = squared.det()

        if determinant != 0:
            return squared.inv(precalc_det=determinant) * mat[:-1]
        else:
            print("system is either dependent or inconsistent")

    def solve_system_gaussian(mat):
        if not isinstance(mat, matrix):
            raise TypeError("can only solve augmented n*n matrix systems")

        if mat.cols != mat.rows + 1:
            raise ValueError(
                "to solve a system of equations you must provide an augmented n*n matrix"
            )

        reduced = utils.rref(mat)
        missing = 0
        for row in reduced:
            if utils._leading_zeroes(row) == mat.cols:
                missing += 1

        if missing:
            print(f"missing {missing} independent equations to solve for a point")
            print(f"got\n{reduced}\nmatrix")
        else:
            return reduced[:-1]

    def _leading_zeroes(mat):
        return next(
            (index for index, value in enumerate(mat) if value not in [0, [0]]),
            len(mat),
        )

    def ref(mat, get_swaps=False, scale=True):
        if not isinstance(mat, matrix):
            raise TypeError("can only get ref of matrices")

        if mat.cols == 1:
            return matrix([1] + [0] * (mat.rows - 1), mat.rows, 1, safe=True)

        swaps = 0

        def sort_mat(mat):
            nonlocal swaps

            temp = [utils._leading_zeroes(arr) for arr in mat]
            finished = False
            while not finished:
                finished = True
                for i in range(mat.rows - 1):
                    if temp[i] > temp[i + 1]:
                        temp[i], temp[i + 1] = temp[i + 1], temp[i]
                        mat[i], mat[i + 1] = mat[i + 1], mat[i]
                        swaps += 1
                        finished = False

            return mat

        mat = mat.copy()
        for key_row in range(mat.rows - 1):
            mat = sort_mat(mat)

            zeroes = utils._leading_zeroes(mat[key_row])

            if zeroes == mat.cols:
                break

            for other_row in range(key_row + 1, mat.rows):
                value = mat[key_row:zeroes]
                if value:
                    mat[other_row] -= (mat[other_row:zeroes] / value) * mat[key_row]
                else:
                    break

        if scale:
            for row in range(mat.rows):
                zeroes = utils._leading_zeroes(mat[row])
                if zeroes == mat.cols:
                    break
                value = mat[row:zeroes]
                if value:
                    mat[row] /= value

        mat = sort_mat(mat)
        if get_swaps:
            return mat, swaps
        return mat

    def rref(mat):
        if not isinstance(mat, matrix):
            raise TypeError("can only get rref of matrices")

        mat = utils.ref(mat)

        zeroes = [utils._leading_zeroes(row) for row in mat]
        try:
            starting = zeroes.index(mat.cols)

        except ValueError:
            starting = mat.rows - 1

        for key_row in range(starting, 0, -1):
            zero = zeroes[key_row]
            if zero == mat.cols:
                continue
            value = mat[key_row:zero]

            for row in range(key_row - 1, -1, -1):
                mat[row] -= (mat[row:zero] / value) * mat[key_row]

        return mat

    def identity(size):
        return matrix(
            [[int(col == row) for col in range(size)] for row in range(size)], safe=True
        )

    def dot(m1, m2):
        if not (isinstance(m1, matrix) and isinstance(m2, matrix)):
            raise TypeError("can only dot two matrices")

        if m1.cols != 1 or m2.cols != 1:
            raise ValueError("can only dot two vector-like matrices (1 column each)")

        if m1.rows != m2.rows:
            raise ValueError("can only dot two vectors-like matrices of the same shape")

        return sum(a[0] * b[0] for a, b in zip(m1, m2))

    def cross(m1, m2):
        if not (isinstance(m1, matrix) and isinstance(m2, matrix)):
            raise TypeError("can only cross two matrices")

        if not (m1.shape() == [3, 1] and m2.shape() == [3, 1]):
            raise ValueError("can only cross two 3d vector-like matrices")

        return matrix(
            [
                m1[1:0] * m2[2:0] - m1[2:0] * m2[1:0],
                m1[2:0] * m2[0:0] - m1[0:0] * m2[2:0],
                m1[0:0] * m2[1:0] - m1[1:0] * m2[0:0],
            ],
            3,
            1,
            safe=True,
        )


class matrix:
    def __init__(self, *args, safe=False):
        if len(args) == 1:
            if not safe:
                if not isinstance(args[0], list):
                    raise TypeError(
                        f"template should be a list, got {type(args[0]).__name__}"
                    )

                if not (
                    all(isinstance(item, list) for item in args[0])
                    or all(type(item) in (int, float) for item in args[0])
                ):
                    raise TypeError(
                        "template must either by a 2d list, or an array of numbers"
                    )

            self._init_2d_array(args[0], safe)

        elif len(args) == 3:
            if not safe:
                if not isinstance(args[0], list):
                    raise TypeError(
                        f"template should be a list, got {type(args[0]).__name__}"
                    )

                if not type(args[1]) in (int, float):
                    raise TypeError(
                        f"expected number for row count, got {type(args[1]).__name__}"
                    )

                if not type(args[2]) in (int, float):
                    raise TypeError(
                        f"expected number for column count, got {type(args[2]).__name__}"
                    )

            self._init_array_row_col(*args, safe)

    def _init_2d_array(self, array, safe):
        self.rows = len(array)
        self.cols = len(array[0])

        if not safe:
            if any(len(row) != self.cols for row in array):
                raise ValueError("matrix rows must be of equal length")

            if any(type(item) not in (int, float) for row in array for item in row):
                raise TypeError("matrix elements must all be numbers")

        self.values = [item for row in array for item in row]

    def _init_array_row_col(self, array, rows, cols, safe):
        if not safe:
            if len(array) != rows * cols:
                raise ValueError(
                    f"incorrect amount of entries for {rows} rows and {cols} columns. got {len(array)}, but expected {rows * cols}"
                )

            if any(type(item) not in (int, float) for item in array):
                raise TypeError("matrix elements must all be numbers")

        self.values = array
        self.rows = rows
        self.cols = cols

    def __str__(self):
        return repr(self)

    def __repr__(self):
        formatted_nums = [utils.format(item) for item in self.values]
        max_number_length = max(len(item) for item in formatted_nums)
        res = "["

        i = 1
        for item in formatted_nums:
            res += item.rjust(max_number_length) + " "
            if i == self.cols:
                i = 1
                res += "\n "
            else:
                i += 1

        return res[:-3] + "]"

    def __iter__(self):
        for row in range(self.rows):
            yield self.values[row * self.cols : (row + 1) * self.cols]

    def __len__(self):
        return self.rows

    def __add__(self, other):
        if not isinstance(other, matrix):
            return NotImplemented

        if not (self.rows == other.rows and self.cols == other.cols):
            raise ValueError("matrices must be of the same shape to add")

        return matrix(
            [self.values[i] + other.values[i] for i in range(self.rows * self.cols)],
            self.rows,
            self.cols,
            safe=True,
        )

    def __sub__(self, other):
        if not isinstance(other, matrix):
            return NotImplemented

        if not (self.rows == other.rows and self.cols == other.cols):
            raise ValueError("matrices must be of the same shape to subtract")

        return matrix(
            [self.values[i] - other.values[i] for i in range(self.rows * self.cols)],
            self.rows,
            self.cols,
            safe=True,
        )

    def __mul__(self, other):
        if type(other) in (int, float):
            return matrix(
                [item * other for item in self.values],
                self.rows,
                self.cols,
                safe=True,
            )

        elif isinstance(other, matrix):
            if self.cols != other.rows:
                raise ValueError(
                    "dimension mismatch, matrix multiplication requres m*n by n*p matrices"
                )

            new_rows = self.rows
            new_cols = other.cols
            res = []

            for row in range(new_rows):
                for col in range(new_cols):
                    res.append(
                        sum(self[row:k] * other[k:col] for k in range(self.cols))
                    )

            return matrix(res, new_rows, new_cols, safe=True)

        return NotImplemented

    def __rmul__(self, other):
        if type(other) in (int, float):
            return self * other

        return NotImplemented

    def __matmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other) in (int, float):
            return matrix(
                [item / other for item in self.values],
                self.rows,
                self.cols,
                safe=True,
            )

        return NotImplemented

    def __floordiv__(self, other):
        if type(other) in (int, float):
            return matrix(
                [item // other for item in self.values],
                self.rows,
                self.cols,
                safe=True,
            )

        return NotImplemented

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        if other < 0:
            raise ValueError("matrix exponentiation only supports positive integers")

        if other == 0:
            if self.rows == self.cols:
                return utils.identity(self.rows)

            raise ValueError("there exists no zero power for non square matrices")

        res = self.copy()
        for i in range(other - 1):
            res *= self
        return res

    def __getitem__(self, key):
        """
        matrix[int] to get row
        matrix[:int] to get column
        matrix[int:] to get row
        matrix[int:int] to get element
        """

        if isinstance(key, int):
            if key < 0:
                key = self.rows + key
            if not 0 <= key < self.rows:
                raise IndexError(f"only {self.rows} rows, while got {key} as index")
            return matrix(
                self.values[self.cols * key : self.cols * (key + 1)],
                self.cols,
                1,
                safe=True,
            )

        if isinstance(key, slice):
            stop = key.stop
            start = key.start
            has_start = start is not None
            has_end = stop is not None

            if has_start:
                if not -self.rows <= start < self.rows:
                    raise IndexError(
                        f"only {self.rows} rows, while got {start} as index"
                    )
                if start < 0:
                    start = self.rows + start

            if has_end:
                if not -self.cols <= stop < self.cols:
                    raise IndexError(
                        f"only {self.cols} cols, while got {stop} as index"
                    )
                if stop < 0:
                    stop = self.cols + stop

            if has_start and has_end:
                return self.values[self.cols * start + stop]

            elif has_start:
                return self[start]

            elif has_end:
                return matrix(
                    [self.values[row * self.cols + stop] for row in range(self.rows)],
                    self.rows,
                    1,
                    safe=True,
                )

            else:
                raise IndexError("matrix slice-indexing requires a start and/or end")

        return IndexError(
            f"matrix indexes can only be integers or slices, got {type(key).__name__}"
        )

    def __setitem__(self, key, value):
        if isinstance(value, matrix):
            if value.cols != 1:
                raise ValueError(
                    "matrix assignment via matrices requires column-vector like structure"
                )
            if value.rows == 1:
                value = int(value)
            else:
                value = [item[0] for item in list(value)]

        if isinstance(key, int):
            if not isinstance(value, list):
                raise TypeError(
                    f"matrix row assignment must be via list/matrix, got {type(value).__name__}"
                )
            if len(value) != self.cols:
                raise ValueError(
                    f"matrix row assignment requires a row-length list, got {len(value)} when expected {self.cols}"
                )
            if key >= self.rows:
                raise ValueError(
                    f"matrix row assignment out of bounds, got {key} with a maximum of {self.rows - 1}"
                )
            for index, item in enumerate(value):
                if not type(item) in (int, float):
                    raise TypeError(
                        f"matrix assignment requires numbers, got {type(item).__name__}"
                    )
                self.values[key * self.cols + index] = item

        if isinstance(key, slice):
            stop = key.stop
            start = key.start

            has_start = start is not None
            has_end = stop is not None

            if has_start:
                if not -self.rows <= start < self.rows:
                    raise IndexError(
                        f"only {self.rows} rows, while got {start} as index"
                    )
                if start < 0:
                    start = self.rows + start

            if has_end:
                if not -self.cols <= stop < self.cols:
                    raise IndexError(
                        f"only {self.cols} cols, while got {stop} as index"
                    )
                if stop < 0:
                    stop = self.cols + stop

            if has_start and has_end:
                if not type(value) in (int, float):
                    raise TypeError(
                        f"matrix item assignment only supports numbers, got {type(value).__name__}"
                    )

                self.values[self.cols * start + stop] = value

            elif has_start:
                self[start] = value

            elif has_end:
                for index, item in enumerate(value):
                    if not type(item) in (int, float):
                        raise TypeError(
                            f"matrix assignment requires numbers, got {type(item).__name__}"
                        )
                    self.values[index * self.cols + stop] = item

            else:
                raise IndexError("matrix slice-indexing requires a start and/or end")

        return IndexError(
            f"matrix indexes can only be integers or slices, got {type(key).__name__}"
        )

    def __eq__(self, other):
        if not isinstance(other, matrix):
            return False

        if self.shape != other.shape:
            return False

        for item1, item2 in zip(self, other):
            if item1 != item2:
                return False

        return True

    def det_naive(self):
        def det(values, dimensions):
            if dimensions == 2:
                return values[3] * values[0] - values[1] * values[2]

            items = dimensions * dimensions
            res = 0
            sign = 1

            for i in range(dimensions):
                submatrix = [
                    values[j] for j in range(dimensions, items) if j % dimensions != i
                ]

                res += sign * det(submatrix, dimensions - 1) * values[i]
                sign = -sign

            return res

        if self.rows != self.cols:
            raise ValueError(
                f"only square matrices have determinants, this matrix is {self.rows}x{self.cols}"
            )

        if self.rows == 1:
            return self.values[0]

        return det(self.values, self.rows)

    def det(self):
        if self.rows != self.cols:
            raise ValueError(
                f"only square matrices have determinants, this matrix is {self.rows}x{self.cols}"
            )

        if self.cols == 2:
            return self.values[0] * self.values[3] - self.values[1] * self.values[2]

        if self.cols == 1:
            return self.values[0]

        reduced, swaps = utils.ref(self, get_swaps=True, scale=False)
        res = 1
        for i in range(self.rows):
            res *= reduced[i:i]
        if swaps % 2:
            res *= -1
        return res

    def minor(self, row, column):
        if row < 0:
            row = self.rows + row
        if column < 0:
            column = self.cols + column

        if not 0 <= row < self.rows:
            raise ValueError("cannot take minor out of matrix bounds")

        if not 0 <= column < self.cols:
            raise ValueError("cannot take minor out of matrix bounds")

        res = matrix(
            [
                self.values[self.cols * i + j]
                for i in range(self.rows)
                for j in range(self.cols)
                if i != row and j != column
            ],
            self.rows - 1,
            self.cols - 1,
            safe=True,
        )

        return res.det()

    def cof(self, row, column):
        return self.minor(row, column) * ((-1) ** ((row + column) % 2))

    def cof_matrix(self):
        return matrix(
            [
                self.cof(row, col)
                for row in range(self.rows)
                for col in range(self.cols)
            ],
            self.rows,
            self.cols,
        )

    def transpose(self):
        return matrix(
            [
                self.values[row * self.rows + col]
                for col in range(self.cols)
                for row in range(self.rows)
            ],
            self.cols,
            self.rows,
            safe=True,
        )

    def T(self):
        return self.transpose()

    def adj(self):
        return self.cof_matrix().T()

    def inv(self, precalc_det=None):
        if precalc_det is not None:
            determinant = precalc_det
        else:
            determinant = self.det()
        if determinant == 0:
            raise ZeroDivisionError("cannot take inverse of matrix with 0 determinant")

        return self.adj() / determinant

    def square(self):
        dimension = min(self.rows, self.cols)
        return matrix(
            [
                self.values[row * self.cols + col]
                for row in range(dimension)
                for col in range(dimension)
            ],
            dimension,
            dimension,
            safe=True,
        )

    def shape(self):
        return [self.rows, self.cols]

    def copy(self):
        return matrix(self.values.copy(), self.rows, self.cols, safe=True)

    def trace(self):
        if self.rows != self.cols:
            raise ValueError("can only find trace of square matrices")

        return sum(self[i:i] for i in range(self.rows))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()


def main():
    pass


if __name__ == "__main__":
    main()
