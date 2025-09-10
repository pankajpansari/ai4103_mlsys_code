#include <iostream>
#include <cmath>

void add_arr(float *x, float *y, int N)
{
    for (int i = 0; i < N; i++)
        y[i] = y[i] + x[i];
}

int main()
{
    int N = 1 << 20; // About 1M elements
    size_t size = N * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add_arr(x, y, N);

    float max_err = 0.0f;
    // Check if all values in sum are 3
    for (int i = 0; i < N; i++)
    {
        max_err = std::fmax(max_err, std::fabs(3.0 - y[i]));
    }

    std::cout << "max error = " << max_err << std::endl;

    free(x);
    free(y);

    return 0;
}