#include <iostream>
#include <cmath>

void multiply_mat(float *x, float *y, float *z, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            z[i * N + j] = 0.0f;
            for (int m = 0; m < N; m++)
            {
                z[i * N + j] += (x[i * N + m] * y[m * N + j]);
            }
        }
    }
}

int main()
{
    int N = 4; // Square matrices of shape N x N (here N = 32)
    size_t size = N * N * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);
    float *z = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            x[i * N + j] = 2.0f;
            y[i * N + j] = 1.0f;
            z[i * N + j] = 0.0f;
        }
    }

    multiply_mat(x, y, z, N);

    float max_err = 0.0f;

    // Check if all values in result are 2 * N
    float target = 2 * N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            max_err = std::fmax(max_err, std::fabs(target - z[i * N + j]));
        }
    }

    std::cout << "max error = " << max_err << std::endl;

    free(x);
    free(y);
    free(z);

    return 0;
}