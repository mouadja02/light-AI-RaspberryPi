#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------------------------
// Include your generated CNN weights file
// --------------------------------------------------------------------------
#include "taha_cnn_weights.h"

// --------------------------------------------------------------------------
// Include your BMP structures and preprocess functions
// (Paste your existing BMP headers and preprocess here or in a separate .h)
// --------------------------------------------------------------------------
#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;      // "BM"
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BMPFileHeader;

typedef struct {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BMPInfoHeader;
#pragma pack(pop)

#define THRESHOLD 20  // or any threshold you want

// ----------------------------
// Provided helper prototypes
// ----------------------------
void save_bmp(const char *output_path, uint8_t *data, int width, int height);
void vertical_flip(uint8_t* data, int width, int height);
int  preprocess(const char *filename, uint8_t out[28 * 28]);

// --------------------------------------------------------------------------
// Helper function: clamp
// --------------------------------------------------------------------------
static inline float clamp(float x, float lower, float upper) {
    return (x < lower) ? lower : (x > upper ? upper : x);
}

// --------------------------------------------------------------------------
// Convolution (single batch) with stride=1, possibly padding, single-channel or multi-channel
// input_shape = [in_channels, in_h, in_w]
// kernel_shape = [out_channels, in_channels, kernel_h, kernel_w]
// pad = layer.padding
//
// Output shape is [out_channels, out_h, out_w] where
//   out_h = in_h (with pad=1, stride=1) 
//   out_w = in_w (with pad=1, stride=1)
// --------------------------------------------------------------------------
float* conv2d(
    const float* input, 
    int in_channels, int in_h, int in_w,
    const ConvLayerDef* layer,
    int* out_channels, int* out_h, int* out_w
) {
    // Extract parameters
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int pad      = layer->padding; 
    int outC     = layer->out_channels;
    int inC      = layer->in_channels; 
    const float* W = layer->weight;
    const float* B = layer->bias;

    // Compute output dimensions (stride=1)
    *out_channels = outC;
    *out_h = in_h;  // With pad=1 and stride=1, output height = in_h
    *out_w = in_w;  // Similarly, output width = in_w

    // Allocate output
    float* output = (float*)calloc(outC * (*out_h) * (*out_w), sizeof(float));

    // For each output channel oc
    for (int oc = 0; oc < outC; oc++) {
        float bias_val = B[oc];

        // For each pixel in the output
        for (int oh = 0; oh < *out_h; oh++) {
            for (int ow = 0; ow < *out_w; ow++) {
                float sum = 0.0f;

                // For each input channel ic
                for (int ic = 0; ic < inC; ic++) {
                    // For each kernel element
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int in_h_idx = oh - pad + kh;
                            int in_w_idx = ow - pad + kw;

                            if (in_h_idx >= 0 && in_h_idx < in_h &&
                                in_w_idx >= 0 && in_w_idx < in_w) {
                                // input index
                                float in_val = input[ic * in_h * in_w + 
                                                     in_h_idx * in_w + 
                                                     in_w_idx];
                                // weight index
                                float w_val = W[ oc*(inC*kernel_h*kernel_w)
                                                 + ic*(kernel_h*kernel_w)
                                                 + kh*kernel_w
                                                 + kw ];
                                sum += in_val * w_val;
                            }
                        }
                    }
                }
                sum += bias_val;

                // output index
                output[oc * (*out_h) * (*out_w) + (oh * (*out_w)) + ow] = sum;
            }
        }
    }

    return output;
}

// --------------------------------------------------------------------------
// ReLU activation in-place
// --------------------------------------------------------------------------
void relu_inplace(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// --------------------------------------------------------------------------
// Max Pool 2x2, stride=2 (no padding), single batch
// input shape: [channels, in_h, in_w]
// output shape: [channels, in_h/2, in_w/2]
// --------------------------------------------------------------------------
float* maxpool2d(
    const float* input, 
    int channels, int in_h, int in_w,
    int kernel, int stride,
    int* out_h, int* out_w
) {
    *out_h = in_h / stride;
    *out_w = in_w / stride;
    
    float* output = (float*)calloc(channels * (*out_h) * (*out_w), sizeof(float));

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < *out_h; oh++) {
            for (int ow = 0; ow < *out_w; ow++) {
                float max_val = -1e30f;
                int in_h_start = oh * stride;
                int in_w_start = ow * stride;
                // For a 2x2 kernel
                for (int kh = 0; kh < kernel; kh++) {
                    for (int kw = 0; kw < kernel; kw++) {
                        int ih = in_h_start + kh;
                        int iw = in_w_start + kw;
                        float val = input[c * in_h * in_w + ih * in_w + iw];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[c * (*out_h) * (*out_w) + oh * (*out_w) + ow] = max_val;
            }
        }
    }

    return output;
}

// --------------------------------------------------------------------------
// Fully connected layer
// input: [in_features]
// weight: [out_features, in_features]
// bias: [out_features]
// output: [out_features]
// --------------------------------------------------------------------------
float* fully_connected(
    const float* input, 
    int in_features,
    const FCLayerDef* layer
) {
    int out_features = layer->out_features;
    const float* W   = layer->weight;
    const float* B   = layer->bias;

    float* output = (float*)calloc(out_features, sizeof(float));

    for (int of = 0; of < out_features; of++) {
        float sum = 0.0f;
        for (int inf = 0; inf < in_features; inf++) {
            sum += input[inf] * W[of * in_features + inf];
        }
        sum += B[of];
        output[of] = sum;
    }

    // Activation if needed
    if (layer->activation_type == ACT_RELU) {
        for (int of = 0; of < out_features; of++) {
            if (output[of] < 0.0f) {
                output[of] = 0.0f;
            }
        }
    }

    return output;
}

// --------------------------------------------------------------------------
// Forward pass
// Architecture we replicate (from your Python):
//   x -> Conv1 -> ReLU -> MaxPool(2x2) -> Conv2 -> ReLU -> MaxPool(2x2)
//     -> Flatten -> FC1 -> ReLU -> FC2 -> [ArgMax for class prediction]
// --------------------------------------------------------------------------
int forward_pass(const float* input_1x28x28) {
    // 0) Input shape is [1, 28, 28] but we treat it as [in_channels=1, in_h=28, in_w=28]
    int inC = 1, inH = 28, inW = 28;

    // 1) Conv1
    int outC, outH, outW;
    float* conv1_out = conv2d(input_1x28x28, inC, inH, inW, &CNN_CONV_LAYERS[0], &outC, &outH, &outW);
    // ReLU
    relu_inplace(conv1_out, outC * outH * outW);
    // MaxPool(2x2)
    int mp1_outH, mp1_outW;
    float* mp1_out = maxpool2d(conv1_out, outC, outH, outW, 2, 2, &mp1_outH, &mp1_outW);
    free(conv1_out);

    // 2) Conv2
    int conv2_outC, conv2_outH, conv2_outW;
    float* conv2_out = conv2d(mp1_out, outC, mp1_outH, mp1_outW, &CNN_CONV_LAYERS[1],
                              &conv2_outC, &conv2_outH, &conv2_outW);
    // ReLU
    relu_inplace(conv2_out, conv2_outC * conv2_outH * conv2_outW);
    // MaxPool(2x2)
    int mp2_outH, mp2_outW;
    float* mp2_out = maxpool2d(conv2_out, conv2_outC, conv2_outH, conv2_outW, 2, 2, &mp2_outH, &mp2_outW);
    free(mp1_out);
    free(conv2_out);

    // 3) Flatten => shape = [conv2_outC * mp2_outH * mp2_outW]
    int flatten_size = conv2_outC * mp2_outH * mp2_outW;
    float* flatten = (float*)calloc(flatten_size, sizeof(float));
    // Copy
    for (int i = 0; i < flatten_size; i++) {
        flatten[i] = mp2_out[i];
    }
    free(mp2_out);

    // 4) FC1
    float* fc1_out = fully_connected(flatten, flatten_size, &CNN_FC_LAYERS[0]);
    free(flatten);

    // 5) FC2
    float* fc2_out = fully_connected(fc1_out, CNN_FC_LAYERS[0].out_features, &CNN_FC_LAYERS[1]);
    free(fc1_out);

    // 6) ArgMax
    int out_features = CNN_FC_LAYERS[1].out_features;  // should be 10
    int predicted_class = 0;
    float max_val = fc2_out[0];
    for (int i = 1; i < out_features; i++) {
        if (fc2_out[i] > max_val) {
            max_val = fc2_out[i];
            predicted_class = i;
        }
    }
    free(fc2_out);

    return predicted_class;
}


// --------------------------------------------------------------------------
// Your provided helper function implementations (same as in your snippet).
// Copy/Paste them below or place them in a separate .h/.c as you prefer.
// For brevity, included inline here:
// --------------------------------------------------------------------------

void save_bmp(const char *output_path, uint8_t *data, int width, int height) {
    FILE *output_file = fopen(output_path, "wb");
    if (!output_file) {
        printf("Error: Could not open output file %s\n", output_path);
        return;
    }

    // BMP headers
    BMPFileHeader bmp_file_header = {0};
    BMPInfoHeader bmp_info_header = {0};

    int row_padded = (width + 3) & ~3;
    int image_size = row_padded * height;

    // Populate BMP headers
    bmp_file_header.bfType = 0x4D42; // 'BM'
    bmp_file_header.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
    bmp_file_header.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    bmp_info_header.biSize = sizeof(BMPInfoHeader);
    bmp_info_header.biWidth = width;
    bmp_info_header.biHeight = -height; // Negative height for top-down
    bmp_info_header.biPlanes = 1;
    bmp_info_header.biBitCount = 8; // Grayscale
    bmp_info_header.biCompression = 0;
    bmp_info_header.biSizeImage = image_size;

    // Write headers
    fwrite(&bmp_file_header, sizeof(BMPFileHeader), 1, output_file);
    fwrite(&bmp_info_header, sizeof(BMPInfoHeader), 1, output_file);

    // Write grayscale palette
    for (int i = 0; i < 256; i++) {
        uint8_t grayscale_entry[4] = {i, i, i, 0};
        fwrite(grayscale_entry, sizeof(uint8_t), 4, output_file);
    }

    // Write pixel data with padding
    uint8_t padding[3] = {0}; // Padding to align rows
    for (int y = 0; y < height; y++) {
        fwrite(&data[y * width], sizeof(uint8_t), width, output_file);
        fwrite(padding, sizeof(uint8_t), row_padded - width, output_file);
    }

    fclose(output_file);
    printf("Processed image saved as BMP to %s\n", output_path);
}

void vertical_flip(uint8_t* data, int width, int height) {
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width; x++) {
            int top_idx = y * width + x;
            int bottom_idx = (height - 1 - y) * width + x;
            uint8_t temp = data[top_idx];
            data[top_idx] = data[bottom_idx];
            data[bottom_idx] = temp;
        }
    }
}

int preprocess(const char *filename, uint8_t out[28 * 28]) {
    FILE *input_file = fopen(filename, "rb");
    if (!input_file) {
        perror("Error opening BMP file");
        return -1;
    }

    BMPFileHeader bmp_file_header;
    BMPInfoHeader bmp_info_header;
    fread(&bmp_file_header, sizeof(BMPFileHeader), 1, input_file);
    fread(&bmp_info_header, sizeof(BMPInfoHeader), 1, input_file);

    // Validate BMP format
    if (bmp_file_header.bfType != 0x4D42 || bmp_info_header.biBitCount != 24) {
        fprintf(stderr, "Error: Unsupported BMP format (must be 24-bit BMP)\n");
        fclose(input_file);
        return -2;
    }

    int width = bmp_info_header.biWidth;
    int height = abs(bmp_info_header.biHeight);
    int row_padded = (width * 3 + 3) & ~3;

    // Read BMP pixel data
    uint8_t *bmp_data = (uint8_t *)malloc(height * row_padded);
    if (!bmp_data) {
        fclose(input_file);
        return -3;
    }
    fseek(input_file, bmp_file_header.bfOffBits, SEEK_SET);
    fread(bmp_data, 1, height * row_padded, input_file);
    fclose(input_file);

    // Convert to grayscale and apply threshold
    uint8_t *grayscale = (uint8_t *)malloc(width * height);
    if (!grayscale) {
        free(bmp_data);
        return -4;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * row_padded + x * 3;
            uint8_t b = bmp_data[idx + 0];
            uint8_t g = bmp_data[idx + 1];
            uint8_t r = bmp_data[idx + 2];
            uint8_t gray = (uint8_t)(0.3f*r + 0.59f*g + 0.11f*b);
            grayscale[y * width + x] = (gray > THRESHOLD) ? 255 : 0;
        }
    }
    free(bmp_data);

    // Find bounding box of white pixels
    int x_min = width, y_min = height, x_max = 0, y_max = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grayscale[y * width + x] == 255) {
                if (x < x_min) x_min = x;
                if (y < y_min) y_min = y;
                if (x > x_max) x_max = x;
                if (y > y_max) y_max = y;
            }
        }
    }

    // Handle case of no white pixels
    if (x_max < x_min || y_max < y_min) {
        // No foreground found, just set output to zero
        memset(out, 0, 28*28);
        free(grayscale);
        return 0;
    }

    // Crop the image
    int cropped_width  = x_max - x_min + 1;
    int cropped_height = y_max - y_min + 1;
    uint8_t *cropped = (uint8_t *)malloc(cropped_width * cropped_height);
    if (!cropped) {
        free(grayscale);
        return -5;
    }
    for (int y = 0; y < cropped_height; y++) {
        for (int x = 0; x < cropped_width; x++) {
            cropped[y * cropped_width + x] = 
                grayscale[(y_min + y) * width + (x_min + x)];
        }
    }
    free(grayscale);

    // Resize to 28x28 (simple nearest-neighbor)
    int resized_width  = 28; 
    int resized_height = 28;
    uint8_t *resized = (uint8_t *)malloc(resized_width * resized_height);
    if (!resized) {
        free(cropped);
        return -6;
    }
    for (int y = 0; y < resized_height; y++) {
        for (int x = 0; x < resized_width; x++) {
            int src_x = x * cropped_width  / resized_width;
            int src_y = y * cropped_height / resized_height;
            resized[y * resized_width + x] = cropped[src_y * cropped_width + src_x];
        }
    }
    free(cropped);

    // Copy to out
    memcpy(out, resized, resized_width * resized_height);
    free(resized);

    // Flip vertically if your coordinate system needs it
    vertical_flip(out, 28, 28);

    // Optionally save the intermediate result (for debugging)
    save_bmp("output.bmp", out, 28, 28);

    return 0;
}


// --------------------------------------------------------------------------
// Main
// Usage: ./cnn_inference input.bmp
// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input.bmp>\n", argv[0]);
        return 1;
    }
    
    const char* input_bmp_path = argv[1];
    
    // 1. Preprocess the input image => 28x28 grayscale in [0..255]
    uint8_t img_28x28[28*28];
    if (preprocess(input_bmp_path, img_28x28) != 0) {
        printf("Error in preprocessing.\n");
        return -1;
    }

    // 2. Convert to float in range [0.0,1.0], then apply Normalize((0.5),(0.5)) => (x - 0.5)/0.5
    float input_data[28*28];
    for (int i = 0; i < 28*28; i++) {
        float x = img_28x28[i] / 255.0f;  // scale to [0,1]
        // Now normalization: (x - 0.5)/0.5 = 2x - 1
        float x_norm = (x - 0.5f) / 0.5f; 
        input_data[i] = x_norm;
    }

    // 3. Forward pass
    int prediction = forward_pass(input_data);

    // 4. Print the result
    printf("Predicted digit: %d\n", prediction);

    return 0;
}