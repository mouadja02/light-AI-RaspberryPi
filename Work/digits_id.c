#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------------------------
// Common BMP Structures
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

// --------------------------------------------------------------------------
// Constants & Macros
// --------------------------------------------------------------------------
#define THRESHOLD 20     // Binarization threshold
#define IMAGE_SIZE 28    // 28x28 final image

// --------------------------------------------------------------------------
// BMP / Image I/O Helpers
// --------------------------------------------------------------------------
static inline float clamp(float x, float lower, float upper) {
    return (x < lower) ? lower : (x > upper ? upper : x);
}

void save_bmp(const char *output_path, uint8_t *data, int width, int height) {
    FILE *output_file = fopen(output_path, "wb");
    if (!output_file) {
        printf("Error: Could not open output file %s\n", output_path);
        return;
    }

    BMPFileHeader bmp_file_header = {0};
    BMPInfoHeader bmp_info_header = {0};

    int row_padded = (width + 3) & ~3;
    int image_size = row_padded * height;

    bmp_file_header.bfType = 0x4D42; // 'BM'
    bmp_file_header.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
    bmp_file_header.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    bmp_info_header.biSize = sizeof(BMPInfoHeader);
    bmp_info_header.biWidth = width;
    bmp_info_header.biHeight = - height; // Negative => top-down
    bmp_info_header.biPlanes = 1;
    bmp_info_header.biBitCount = 8; // Grayscale
    bmp_info_header.biCompression = 0;
    bmp_info_header.biSizeImage = image_size;

    // Write BMP headers
    fwrite(&bmp_file_header, sizeof(BMPFileHeader), 1, output_file);
    fwrite(&bmp_info_header, sizeof(BMPInfoHeader), 1, output_file);

    // Write grayscale palette
    for (int i = 0; i < 256; i++) {
        uint8_t grayscale_entry[4] = {i, i, i, 0};
        fwrite(grayscale_entry, sizeof(uint8_t), 4, output_file);
    }

    // Write pixel data
    uint8_t padding[3] = {0};
    for (int y = 0; y < height; y++) {
        fwrite(&data[y * width], sizeof(uint8_t), width, output_file);
        fwrite(padding, sizeof(uint8_t), row_padded - width, output_file);
    }

    fclose(output_file);
    printf("Processed image saved as BMP to %s\n", output_path);
}

void vertical_flip(uint8_t *data, int width, int height) {
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width; x++) {
            int top_idx    = y * width + x;
            int bottom_idx = (height - 1 - y) * width + x;
            uint8_t temp   = data[top_idx];
            data[top_idx]  = data[bottom_idx];
            data[bottom_idx] = temp;
        }
    }
}

// --------------------------------------------------------------------------
// Preprocess a 24-bit BMP => 28×28 binarized grayscale
// --------------------------------------------------------------------------
int preprocess(const char *filename, uint8_t out[IMAGE_SIZE * IMAGE_SIZE]) {
    FILE *input_file = fopen(filename, "rb");
    if (!input_file) {
        perror("Error opening BMP file");
        return -1;
    }

    BMPFileHeader bmp_file_header;
    BMPInfoHeader bmp_info_header;
    fread(&bmp_file_header, sizeof(BMPFileHeader), 1, input_file);
    fread(&bmp_info_header, sizeof(BMPInfoHeader), 1, input_file);

    // Check BMP format
    if (bmp_file_header.bfType != 0x4D42 || bmp_info_header.biBitCount != 24) {
        fprintf(stderr, "Error: Unsupported BMP format (must be 24-bit)\n");
        fclose(input_file);
        return -2;
    }

    int width  = (int)bmp_info_header.biWidth;
    int height = (bmp_info_header.biHeight > 0) ? bmp_info_header.biHeight : -bmp_info_header.biHeight;
    int row_padded = (width * 3 + 3) & ~3;

    // Read raw pixel data
    uint8_t *bmp_data = (uint8_t*)malloc(height * row_padded);
    if (!bmp_data) {
        fclose(input_file);
        return -3;
    }
    fseek(input_file, bmp_file_header.bfOffBits, SEEK_SET);
    fread(bmp_data, 1, height * row_padded, input_file);
    fclose(input_file);

    // Convert to grayscale + threshold
    uint8_t *grayscale = (uint8_t*)malloc(width * height);
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
            if (gray > THRESHOLD) grayscale[y*width + x] = 255;
            else                  grayscale[y*width + x] = 0;
        }
    }
    free(bmp_data);

    // Bounding box of white pixels
    int x_min = width, x_max = 0;
    int y_min = height, y_max = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grayscale[y*width + x] == 255) {
                if (x < x_min) x_min = x;
                if (y < y_min) y_min = y;
                if (x > x_max) x_max = x;
                if (y > y_max) y_max = y;
            }
        }
    }

    // If no white pixels found
    if (x_max < x_min || y_max < y_min) {
        memset(out, 0, IMAGE_SIZE*IMAGE_SIZE);
        free(grayscale);
        return 0;
    }

    // Crop
    int cropped_width  = x_max - x_min + 1;
    int cropped_height = y_max - y_min + 1;
    uint8_t *cropped = (uint8_t*)malloc(cropped_width*cropped_height);
    if (!cropped) {
        free(grayscale);
        return -5;
    }
    for (int yy = 0; yy < cropped_height; yy++) {
        for (int xx = 0; xx < cropped_width; xx++) {
            cropped[yy*cropped_width + xx] =
                grayscale[(y_min+yy)*width + (x_min+xx)];
        }
    }
    free(grayscale);

    // Resize to 28x28 (nearest neighbor)
    uint8_t *resized = (uint8_t*)malloc(IMAGE_SIZE*IMAGE_SIZE);
    if (!resized) {
        free(cropped);
        return -6;
    }
    for (int yy = 0; yy < IMAGE_SIZE; yy++) {
        for (int xx = 0; xx < IMAGE_SIZE; xx++) {
            int src_x = xx * cropped_width  / IMAGE_SIZE;
            int src_y = yy * cropped_height / IMAGE_SIZE;
            resized[yy*IMAGE_SIZE + xx] = cropped[src_y*cropped_width + src_x];
        }
    }
    free(cropped);

    memcpy(out, resized, IMAGE_SIZE*IMAGE_SIZE);
    free(resized);

    // Flip vertically if desired
    vertical_flip(out, IMAGE_SIZE, IMAGE_SIZE);

    // Optional save for debugging
    save_bmp("output.bmp", out, IMAGE_SIZE, IMAGE_SIZE);

    return 0;
}

// --------------------------------------------------------------------------
// MLP CODE (compiled if USE_MLP defined)
// --------------------------------------------------------------------------
#ifdef USE_MLP
#include "mlp_weights.h"  

static float* fully_connected_mlp(
    const float* input,
    int in_features,
    const float* weight,
    const float* bias,
    int out_features
) {
    float* output = (float*)calloc(out_features, sizeof(float));
    for (int of = 0; of < out_features; of++) {
        float sum = 0.0f;
        for (int inf = 0; inf < in_features; inf++) {
            sum += input[inf] * weight[of*in_features + inf];
        }
        sum += bias[of];
        output[of] = sum;
    }
    return output;
}

static void relu_inplace_mlp(float* data, int length) {
    for (int i = 0; i < length; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// MLP forward pass
static int mlp_forward_pass(const uint8_t* image_28x28) {
    // Flatten + normalize => [-1..1]
    float input_data[IMAGE_SIZE*IMAGE_SIZE];
    for(int i = 0; i < IMAGE_SIZE*IMAGE_SIZE; i++) {
        float x = (float)image_28x28[i] / 255.0f; 
        input_data[i] = x * 2.0f - 1.0f;
    }

    float* current_activations = (float*)calloc(IMAGE_SIZE*IMAGE_SIZE, sizeof(float));
    memcpy(current_activations, input_data, sizeof(float)*IMAGE_SIZE*IMAGE_SIZE);
    int current_size = IMAGE_SIZE*IMAGE_SIZE; // 784

    for (int layer_idx = 0; layer_idx < MLP_NUM_LAYERS; layer_idx++) {
        FCLayerDef layer = MLP_LAYERS[layer_idx];
        float* out = fully_connected_mlp(
            current_activations,
            layer.in_features,
            layer.weight,
            layer.bias,
            layer.out_features
        );
        free(current_activations);
        current_activations = out;
        current_size = layer.out_features;

        if (layer.activation_type == ACT_RELU) {
            relu_inplace_mlp(current_activations, current_size);
        }
    }

    // Argmax
    int predicted_class = 0;
    float max_val = current_activations[0];
    for (int i = 1; i < current_size; i++) {
        if (current_activations[i] > max_val) {
            max_val = current_activations[i];
            predicted_class = i;
        }
    }
    free(current_activations);
    return predicted_class;
}

#endif // USE_MLP

// --------------------------------------------------------------------------
// CNN CODE (compiled if USE_CNN defined)
// --------------------------------------------------------------------------
#ifdef USE_CNN
#include "cnn_weights.h" 

static float* conv2d_cnn(
    const float* input, 
    int in_channels, int in_h, int in_w,
    const ConvLayerDef* layer,
    int* out_channels, int* out_h, int* out_w
) {
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int outC     = layer->out_channels;
    int inC      = layer->in_channels;
    int pad      = layer->padding;
    const float* W = layer->weight;
    const float* B = layer->bias;

    // For stride=1, out dims = in dims if pad=1
    *out_channels = outC;
    *out_h = in_h;
    *out_w = in_w;

    float* output = (float*)calloc(outC * (*out_h) * (*out_w), sizeof(float));
    for (int oc = 0; oc < outC; oc++) {
        float bias_val = B[oc];
        for (int oh = 0; oh < *out_h; oh++) {
            for (int ow = 0; ow < *out_w; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh - pad + kh;
                            int iw = ow - pad + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                float in_val = input[ic*in_h*in_w + ih*in_w + iw];
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
                output[oc*(*out_h)*(*out_w) + oh*(*out_w) + ow] = sum;
            }
        }
    }
    return output;
}

static void relu_inplace_cnn(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

static float* maxpool2d_cnn(
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
                for (int kh = 0; kh < kernel; kh++) {
                    for (int kw = 0; kw < kernel; kw++) {
                        int ih = in_h_start + kh;
                        int iw = in_w_start + kw;
                        float val = input[c*in_h*in_w + ih*in_w + iw];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[c*(*out_h)*(*out_w) + oh*(*out_w) + ow] = max_val;
            }
        }
    }
    return output;
}

static float* fully_connected_cnn(
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
            sum += input[inf] * W[of*in_features + inf];
        }
        sum += B[of];
        output[of] = sum;
    }

    // Activation?
    if (layer->activation_type == ACT_RELU) {
        for (int i = 0; i < out_features; i++) {
            if (output[i] < 0.0f) output[i] = 0.0f;
        }
    }
    return output;
}

// CNN forward
static int cnn_forward_pass(const uint8_t* img_28x28) {
    // Normalize [0..255] -> [-1..1]
    float input_data[28*28];
    for (int i = 0; i < 28*28; i++) {
        float x = img_28x28[i] / 255.0f;
        input_data[i] = x * 2.0f - 1.0f;
    }

    // shape: [1,28,28]
    int inC = 1, inH = 28, inW = 28;

    // Conv1
    int outC, outH, outW;
    float* conv1_out = conv2d_cnn(input_data, inC, inH, inW, &CNN_CONV_LAYERS[0], &outC, &outH, &outW);
    relu_inplace_cnn(conv1_out, outC*outH*outW);
    float* mp1_out = maxpool2d_cnn(conv1_out, outC, outH, outW, 2, 2, &outH, &outW);
    free(conv1_out);

    // Conv2
    float* conv2_out = conv2d_cnn(mp1_out, outC, outH, outW, &CNN_CONV_LAYERS[1], &outC, &outH, &outW);
    relu_inplace_cnn(conv2_out, outC*outH*outW);
    float* mp2_out = maxpool2d_cnn(conv2_out, outC, outH, outW, 2, 2, &outH, &outW);
    free(mp1_out);
    free(conv2_out);

    // Flatten
    int flatten_size = outC * outH * outW;
    float* flatten = (float*)calloc(flatten_size, sizeof(float));
    for (int i = 0; i < flatten_size; i++) {
        flatten[i] = mp2_out[i];
    }
    free(mp2_out);

    // FC1
    float* fc1_out = fully_connected_cnn(flatten, flatten_size, &CNN_FC_LAYERS[0]);
    free(flatten);

    // FC2
    float* fc2_out = fully_connected_cnn(fc1_out, CNN_FC_LAYERS[0].out_features, &CNN_FC_LAYERS[1]);
    free(fc1_out);

    // Argmax
    int out_features = CNN_FC_LAYERS[1].out_features;
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
#endif // USE_CNN

// --------------------------------------------------------------------------
// Main
//   compile with -D USE_MLP or -D USE_CNN
// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input.bmp>\n", argv[0]);
        return 1;
    }

    // 1) Preprocess -> 28x28 grayscale
    uint8_t img_28x28[IMAGE_SIZE * IMAGE_SIZE];
    if (preprocess(argv[1], img_28x28) != 0) {
        printf("Error in preprocessing!\n");
        return -1;
    }

#if defined(USE_MLP)
    printf("Running MLP inference...\n");
    int prediction = mlp_forward_pass(img_28x28);

#elif defined(USE_CNN)
    printf("Running CNN inference...\n");
    int prediction = cnn_forward_pass(img_28x28);

#else
    printf("Error: No architecture selected!\n");
    printf("Compile with -D USE_MLP or -D USE_CNN.\n");
    return -2;
#endif

    // 3) Print result
    printf("Predicted digit: %d\n", prediction);
    return 0;
}
