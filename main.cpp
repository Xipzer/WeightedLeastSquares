#include <iostream>
#include <vector>

using namespace std;

typedef vector<float> row;
typedef vector<row> matrix;

void cofactor(const matrix& x, matrix& cofactor, int ri, int ci, int dimension);
float determinant(const matrix& x, int dimension);
matrix adjugate(const matrix& x);
matrix identity(const matrix& x);
matrix inverse(const matrix& x);
matrix transpose(const matrix& x);
matrix dot(const matrix& x, const matrix& y);
matrix ordinaryLeastSquares(const matrix& x, const matrix& y);
matrix weightedLeastSquares(const matrix& w, const matrix& x, const matrix& y);

void printMatrix(const matrix& x);
void visualiseDeterminant();
void visualiseCofactor();
void visualiseIdentity();
void visualiseInverse();
void visualiseTranspose();
void visualiseDot();
void visualiseOrdinaryLeastSquares();
void visualiseWeightedLeastSquares();

int main()
{
    visualiseDeterminant();
    cout << "\n";
    visualiseCofactor();
    cout << "\n";
    visualiseIdentity();
    cout << "\n";
    visualiseInverse();
    cout << "\n";
    visualiseTranspose();
    cout << "\n";
    visualiseDot();
    cout << "\n";
    visualiseOrdinaryLeastSquares();
    cout << "\n";
    visualiseWeightedLeastSquares();

    return 0;
}

/**
 * Computes the identity matrix for a given n * n matrix.
 *
 * @param x the subject matrix.
 * @return the identity matrix.
 */
matrix identity(const matrix& x)
{
    int rows = int(x.size());
    int columns = int(x[0].size());

    matrix identity;

    for (int i = 0; i < rows; i++)
    {
        row temp;

        for (int j = 0; j < columns; j++)
            (j == i) ? temp.push_back(1) : temp.push_back(0);

        identity.push_back(temp);
    }

    return identity;
}

/**
 * Computes the cofactor of the given matrix.
 *
 * @param x the subject matrix.
 * @param cofactor the cofactor matrix.
 * @param ri the row index.
 * @param ci the column index.
 * @param dimension the current dimension of the subject matrix.
 */
void cofactor(const matrix& x, matrix& cofactor, int ri, int ci, int dimension)
{
    int r = 0, c = 0;

    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            if (i != ri && j != ci)
            {
                cofactor[r][c++] = x[i][j];

                if (c == dimension - 1)
                {
                    c = 0;
                    r++;
                }
            }
}

/**
 * Computes the determinant of a given matrix.
 *
 * @param x the subject matrix.
 * @param dimension the current dimension of the subject matrix.
 * @return the determinant.
 */
float determinant(const matrix& x, int dimension)
{
    if (dimension == 1)
        return x[0][0];

    float result = 0;
    float sign = 1;

    matrix c(x.size(), vector<float>(x.size()));

    for (int i = 0; i < dimension; i++)
    {
        cofactor(x, c, 0, i, dimension);
        result += sign * x[0][i] * determinant(c, dimension - 1);

        sign = -sign;
    }

    return result;
}

/**
 * Compute the adjugate matrix of a given matrix.
 *
 * @param x the subject matrix.
 * @return the adjugate matrix.
 */
matrix adjugate(const matrix& x)
{
    int dimension = int(x.size());

    matrix a(dimension, vector<float>(dimension));

    if (dimension == 1)
        return a;

    float sign;

    matrix c(dimension, vector<float>(dimension));

    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
        {
            sign = (i + j) % 2 == 0 ? 1 : -1;

            cofactor(x, c, i, j, dimension);
            a[j][i] = sign * determinant(c, dimension - 1);
        }

    return a;
}

/**
 * Computes the inverse of a given matrix.
 *
 * @param x the matrix to be inverted.
 * @return the inverted matrix.
 */
matrix inverse(const matrix& x)
{
    int dimension = int(x.size());

    matrix inverse(x.size(), vector<float>(dimension));

    float d = determinant(x, dimension);

    if (d == 0)
        return inverse;

    matrix a = adjugate(x);

    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            inverse[i][j] = a[i][j] / d;

    return inverse;
}

/**
 * Computes the transpose of a given matrix.
 *
 * @param x the matrix to be transposed.
 * @return the transposed matrix.
 */
matrix transpose(const matrix& x)
{
    int rows = int(x.size());
    int columns = int(x[0].size());

    matrix t(columns, vector<float>(rows));

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            t[j][i] = x[i][j];

    return t;
}

/**
 * Computes the multiplication of given n * m and m * p matrices.
 *
 * @param x the first input matrix.
 * @param y the second input matrix.
 * @return the resultant matrix.
 */
matrix dot(const matrix& x, const matrix& y)
{
    int xRows = int(x.size());
    int xColumns = int(x[0].size());
    int yColumns = int(y[0].size());

    matrix d(xRows, vector<float>(yColumns));

    for (int i = 0; i < xRows; i++)
        for (int j = 0; j < yColumns; j++)
            for (int k = 0; k < xColumns; k++)
                d[i][j] += x[i][k] * y[k][j];

    return d;
}

/**
 * Computes the weights for the line that best fits the weighted features and labels.
 *
 * @param x the features matrix.
 * @param y the labels matrix.
 * @return the resultant weights.
 */
matrix ordinaryLeastSquares(const matrix& x, const matrix& y)
{
    matrix xt = transpose(x);

    return dot(inverse(dot(xt, x)), dot(xt, y));
}

/**
 * Computes the weights for the line that best fits the weighted features and labels.
 *
 * @param w the weights matrix.
 * @param x the features matrix.
 * @param y the labels matrix.
 * @return the resultant weights.
 */
matrix weightedLeastSquares(const matrix& w, const matrix& x, const matrix& y)
{
    matrix xt = transpose(x);

    return dot(inverse(dot(dot(xt, w), x)), dot(dot(xt, w), y));
}

void printMatrix(const matrix& x)
{
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < x[0].size(); j++)
            cout << " " << x[i][j];
        cout << "\n";
    }
}

void visualiseDeterminant()
{
    matrix x;

    row first = { 0, 1, 2 };
    row second = { 1, 2, 1 };
    row third = { 2, 1, 0 };

    x.push_back(first);
    x.push_back(second);
    x.push_back(third);

    cout << "Original Matrix:" << "\n";
    printMatrix(x);

    float d = determinant(x, int(x.size()));

    cout << "Determinant: " << d << "\n";
}

void visualiseCofactor()
{
    matrix x;

    row first = { 0, 1, 2 };
    row second = { 1, 2, 1 };
    row third = { 2, 1, 0 };

    x.push_back(first);
    x.push_back(second);
    x.push_back(third);

    cout << "Original Matrix:" << "\n";
    printMatrix(x);

    matrix c(x.size(), vector<float>(x.size()));

    cofactor(x, c, 0, 0, x.size());

    cout << "Cofactor Matrix (For x11):" << "\n";
    printMatrix(c);
}

void visualiseIdentity()
{
    matrix x;

    cout << "Original Matrix:" << "\n";
    for (int i = 0; i < 3; i++)
    {
        row temp;

        for (int j = 0; j < 3; j++)
        {
            cout << " " << j;
            temp.push_back(j);
        }

        cout << "\n";
        x.push_back(temp);
    }

    x = identity(x);

    cout << "Identity Matrix:" << "\n";
    printMatrix(x);
}

void visualiseInverse()
{
    matrix x;

    row first = { 0, 1, 2 };
    row second = { 1, 2, 1 };
    row third = { 2, 1, 0 };

    x.push_back(first);
    x.push_back(second);
    x.push_back(third);

    cout << "Original Matrix:" << "\n";
    printMatrix(x);

    x = inverse(x);

    cout << "Inverse Matrix:" << "\n";
    printMatrix(x);
}

void visualiseTranspose()
{
    matrix x;

    cout << "Original Matrix:" << "\n";
    for (int i = 0; i < 3; i++)
    {
        row temp;

        for (int j = 0; j < 3; j++)
        {
            cout << " " << j;
            temp.push_back(j);
        }

        cout << "\n";
        x.push_back(temp);
    }

    x = transpose(x);

    cout << "Transpose Matrix:" << "\n";
    printMatrix(x);
}

void visualiseDot()
{
    matrix x;
    matrix y;

    cout << "Original Matrix (1):" << "\n";
    for (int i = 0; i < 3; i++)
    {
        row temp;

        for (int j = 0; j < 3; j++)
        {
            cout << " " << j;
            temp.push_back(j);
        }

        cout << "\n";
        x.push_back(temp);
    }

    cout << "Original Matrix (2):" << "\n";
    for (int i = 0; i < 3; i++)
    {
        row temp;

        for (int j = 2; j >= 0; j--)
        {
            cout << " " << j;
            temp.push_back(j);
        }

        cout << "\n";
        y.push_back(temp);
    }

    matrix lol;
    lol = dot(x, y);

    cout << "Dot Matrix:" << "\n";
    printMatrix(lol);
}

void visualiseOrdinaryLeastSquares()
{
    matrix x = {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}};

    cout << "Original Matrix (x):" << "\n";
    printMatrix(x);

    matrix y = {{2}, {1}, {2}, {1}, {1}, {1}, {1}, {1}};

    cout << "Original Matrix (y):" << "\n";
    printMatrix(y);

    matrix weights = ordinaryLeastSquares(x, y);

    cout << "Ordinary Least Squares:" << "\n";
    printMatrix(weights);
}

void visualiseWeightedLeastSquares()
{
    matrix x = {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}};

    cout << "Original Matrix (x):" << "\n";
    printMatrix(x);

    matrix y = {{2}, {1}, {2}, {1}, {1}, {1}, {1}, {1}};

    cout << "Original Matrix (y):" << "\n";
    printMatrix(y);

    matrix w = {{4}, {66.5}, {56}, {14.5}, {85.5}, {36.5}, {34}, {88}};

    cout << "Original Matrix (w):" << "\n";
    printMatrix(w);

    matrix weights = weightedLeastSquares(w, x, y);

    cout << "Weighted Least Squares:" << "\n";
    printMatrix(weights);
}


