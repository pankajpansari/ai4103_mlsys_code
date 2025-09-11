#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <sndfile.h>
#include <algorithm>
#include <cassert>

#include "blur.cuh"

using std::cout;
using std::cerr;
using std::endl;

const float PI = 3.14159265358979;

float gaussian(float x, float mean, float std) {
    return (1 / (std * sqrt(2 * PI)))
        * exp(-1.0 / 2.0 * pow((x - mean) / std, 2));
}

/*
 * Reads in audio data, and convolves each channel with the specified 
 * filtering function h[n], producing output data. 
 * 
 * Uses both CPU and GPU implementations, and compares the results.
 */

int large_gauss_test(int argc, char **argv) {

    /* Form Gaussian blur vector */
    float mean = 0.0;
    float std = 5.0;

    int GAUSSIAN_SIDE_WIDTH = 10;
    int GAUSSIAN_SIZE = 2 * GAUSSIAN_SIDE_WIDTH + 1;

    // Space for both sides of the gaussian blur vector, plus the middle,
    // gives this size requirement
    float *blur_v = (float *) malloc(sizeof (float) * GAUSSIAN_SIZE );

    // Fill it from the middle out
    for (int i = -GAUSSIAN_SIDE_WIDTH; i <= GAUSSIAN_SIDE_WIDTH; i++)
        blur_v[ GAUSSIAN_SIDE_WIDTH + i ] = gaussian(i, mean, std);

    // Normalize to avoid clipping and/or hearing loss (brackets for scoping)
    {
        float total = 0.0;
        for (int i = 0; i < GAUSSIAN_SIZE; i++)
            total += blur_v[i];
        for (int i = 0; i < GAUSSIAN_SIZE; i++)
            blur_v[i] /= total;

        cout << "Normalized by factor of: " << total << endl;
    }


    for (int i = 0; i < GAUSSIAN_SIZE; i++)
        cout << "gaussian[" << i << "] = " << blur_v[i] << endl;


    SNDFILE *in_file, *out_file;
    SF_INFO in_file_info, out_file_info;

    int amt_read;

    // Open input audio file
    in_file = sf_open(argv[3], SFM_READ, &in_file_info);

    // Read audio
    float *all_channel_input =
        new float[in_file_info.frames * in_file_info.channels];
    amt_read =
        sf_read_float(in_file, all_channel_input,
            in_file_info.frames * in_file_info.channels);
    assert(amt_read == in_file_info.frames * in_file_info.channels);

    // Prepare output storage
    float *all_channel_output =
        new float[in_file_info.frames * in_file_info.channels];

    int n_channels = in_file_info.channels;
    int n_frames = in_file_info.frames;


    // Per-channel input data
    float *input_data = (float *) malloc(sizeof (float) * n_frames);

    // Output data storage for GPU implementation (will write to this from GPU)
    float *output_data = (float *) malloc(n_frames * sizeof (float));

    // Output data storage for CPU implementation
    float *output_data_host = (float *) malloc(n_frames * sizeof (float));

    // Iterate through each audio channel (e.g. 2 iterations for stereo files)
    for (int ch = 0; ch < n_channels; ch++) {
      // Load this channel's data
      for (int i = 0; i < n_frames; i++)
          input_data[i] = all_channel_input[(i * n_channels) + ch];
       
        // CPU Blurring
        cout << "CPU blurring..." << endl;

        memset(output_data_host, 0, n_frames * sizeof (float));

        // CPU Convolution
        {
            // edge case: clamp to left
            for (int i = 0; i < GAUSSIAN_SIZE; i++) {
                for (int j = 0; j <= i; j++)
                    output_data_host[i] += input_data[i - j] * blur_v[j]; 
            }
            // typical case
            for (int i = GAUSSIAN_SIZE; i < n_frames; i++) {
                for (int j = 0; j < GAUSSIAN_SIZE; j++)
                    output_data_host[i] += input_data[i - j] * blur_v[j]; 
            }
        }

        // GPU blurring
        cout << "GPU blurring..." << endl;

        // Cap the number of blocks
        const unsigned int local_size = atoi(argv[1]);
        const unsigned int max_blocks = atoi(argv[2]);
        const unsigned int blocks = std::min(max_blocks,
            (unsigned int) ceil(n_frames / (float) local_size));

        cuda_call_blur_kernel(blocks, local_size, input_data, blur_v, 
            output_data, n_frames, GAUSSIAN_SIZE);


        // Compare results
        bool success = true;
        for (int i = 0; i < n_frames; i++) {
            if (fabs(output_data_host[i] - output_data[i]) < 1e-6) {
            #if 0
                cout << "Correct output at index " << i << ": " << output_data_host[i] << ", " 
                    << output_data[i] << endl;
            #endif
            }
            else {
                success = false;
                cerr << "Incorrect output at index " << i << ": " <<
                    output_data_host[i] << ", "  << output_data[i] << endl;
            }
        }

        if (success)
            cout << endl << "Successful output" << endl;

        // Write output audio data to multichannel array
        for (int i = 0; i < n_frames; i++){
            all_channel_output[i * n_channels + ch] = output_data[i];
        }


    }

    // Free memory on host
    free(input_data);
    free(output_data);
    free(output_data_host);


    // Write audio output to file
    out_file_info = in_file_info;
    out_file = sf_open(argv[4], SFM_WRITE, &out_file_info);
    if (!out_file) {
        cerr << "Cannot open output file, exiting\n";
        exit(EXIT_FAILURE);
    }

    sf_write_float(out_file, all_channel_output, amt_read); 
    sf_close(in_file);
    sf_close(out_file);

    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
    return large_gauss_test(argc, argv);
}


