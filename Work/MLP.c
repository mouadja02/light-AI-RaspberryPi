#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------------------------
// Include your generated MLP weights file
//   This file must be in the same directory or adjust the include path.
//   It should define MLP_NUM_LAYERS, FCLayerDef, MLP_LAYERS[], etc.
// --------------------------------------------------------------------------
#include "mlp_weights.h"

// --------------------------------------------------------------------------
// BMP Structures (same as your CNN code snippet)
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
// Constants
// --------------------------------------------------------------------------
#define THRESHOLD 20  // threshold for binarization (adjust if needed)
#define IMAGE_SIZE 28 // 28x28 final

// --------------------------------------------------------------------------
// Helper: clamp
// --------------------------------------------------------------------------
static inline float clamp(float x, float lower, float upper) {
    return (x < lower) ? lower : (x > upper ? upper : x);
}

// --------------------------------------------------------------------------
// Functions for BMP saving/flipping (for debugging)
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
    bmp_info_header.biHeight = -height; // Negative => top-down
    bmp_info_header.biPlanes = 1;
    bmp_info_header.biBitCount = 8; // Grayscale
    bmp_info_header.biCompression = 0;
    bmp_info_header.biSizeImage = image_size;

    // Write headers
    fwrite(&bmp_file_header, sizeof(BMPFileHeader), 1, output_file);
    fwrite(&bmp_info_header, sizeof(BMPInfoHeader), 1, output_file);

    // Write a grayscale palette (256 entries)
    for (int i = 0; i < 256; i++) {
        uint8_t grayscale_entry[4] = {i, i, i, 0};
        fwrite(grayscale_entry, sizeof(uint8_t), 4, output_file);
    }

    // Write pixel data with padding
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
// Preprocessing Function:
//   1) Read a 24-bit BMP from disk
//   2) Convert to grayscale & binarize
//   3) Find bounding box of white pixels
//   4) Crop & resize to 28x28
//   5) Flip vertically (if needed)
//   6) Save optional "output.bmp" for debugging
//   7) Store final 28×28 in out[] (range 0..255)
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

    // Validate BMP format (expecting 24-bit uncompressed)
    if (bmp_file_header.bfType != 0x4D42 || bmp_info_header.biBitCount != 24) {
        fprintf(stderr, "Error: Unsupported BMP format (must be 24-bit BMP)\n");
        fclose(input_file);
        return -2;
    }

    int width  = bmp_info_header.biWidth;
    int height = abs(bmp_info_header.biHeight);
    int row_padded = (width * 3 + 3) & ~3;

    // Read pixel data
    uint8_t *bmp_data = (uint8_t *)malloc(height * row_padded);
    if (!bmp_data) {
        fclose(input_file);
        return -3;
    }
    fseek(input_file, bmp_file_header.bfOffBits, SEEK_SET);
    fread(bmp_data, 1, height * row_padded, input_file);
    fclose(input_file);

    // Convert to grayscale & threshold
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
            // Simple grayscale
            uint8_t gray = (uint8_t)(0.3f * r + 0.59f * g + 0.11f * b);
            // Binarize
            if (gray > THRESHOLD)
                grayscale[y * width + x] = 255;
            else
                grayscale[y * width + x] = 0;
        }
    }
    free(bmp_data);

    // Find bounding box of white pixels
    int x_min = width, x_max = 0;
    int y_min = height, y_max = 0;
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

    // If no white pixels found, fill out with 0
    if (x_max < x_min || y_max < y_min) {
        memset(out, 0, IMAGE_SIZE * IMAGE_SIZE);
        free(grayscale);
        return 0;
    }

    int cropped_width  = x_max - x_min + 1;
    int cropped_height = y_max - y_min + 1;
    uint8_t *cropped = (uint8_t *)malloc(cropped_width * cropped_height);
    if (!cropped) {
        free(grayscale);
        return -5;
    }

    // Copy the bounding box
    for (int yy = 0; yy < cropped_height; yy++) {
        for (int xx = 0; xx < cropped_width; xx++) {
            cropped[yy * cropped_width + xx] =
                grayscale[(y_min + yy) * width + (x_min + xx)];
        }
    }
    free(grayscale);

    // Resize to 28x28 via nearest neighbor
    uint8_t *resized = (uint8_t *)malloc(IMAGE_SIZE * IMAGE_SIZE);
    if (!resized) {
        free(cropped);
        return -6;
    }
    for (int yy = 0; yy < IMAGE_SIZE; yy++) {
        for (int xx = 0; xx < IMAGE_SIZE; xx++) {
            int src_x = xx * cropped_width  / IMAGE_SIZE;
            int src_y = yy * cropped_height / IMAGE_SIZE;
            resized[yy * IMAGE_SIZE + xx] = cropped[src_y * cropped_width + src_x];
        }
    }
    free(cropped);

    // Copy to output & vertically flip if needed
    memcpy(out, resized, IMAGE_SIZE * IMAGE_SIZE);
    free(resized);

    vertical_flip(out, IMAGE_SIZE, IMAGE_SIZE);

    // Optional: save the processed image as "output.bmp"
    save_bmp("output.bmp", out, IMAGE_SIZE, IMAGE_SIZE);

    return 0;
}

// --------------------------------------------------------------------------
// Fully Connected Forward
//   input  shape: in_features
//   output shape: out_features
//   weight shape: [out_features, in_features]
//   bias shape:   [out_features]
// --------------------------------------------------------------------------
float* fully_connected(
    const float* input,
    int in_features,
    const float* weight,
    const float* bias,
    int out_features
) {
    float* output = (float*)calloc(out_features, sizeof(float));
    for (int of = 0; of < out_features; of++) {
        // Dot product
        float sum = 0.0f;
        for (int inf = 0; inf < in_features; inf++) {
            sum += input[inf] * weight[of * in_features + inf];
        }
        sum += bias[of];
        output[of] = sum;
    }
    return output;
}

// --------------------------------------------------------------------------
// ReLU Activation In-Place
// --------------------------------------------------------------------------
void relu_inplace(float* data, int length) {
    for (int i = 0; i < length; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// --------------------------------------------------------------------------
// MLP Forward Pass
//   1) Flatten (28×28) => 784
//   2) For each layer in MLP_LAYERS:
//        out = W x in + b
//        apply ReLU if activation_type == ACT_RELU
//   3) final out => ArgMax => predicted class
// --------------------------------------------------------------------------
int mlp_forward_pass(const uint8_t* image_28x28) {
    // 1) Convert to float, apply transforms.Normalize((0.5,),(0.5,))
    //    => x_norm = (x/255.0 - 0.5)/0.5 = (x/255.0)*2 - 1
    float input[IMAGE_SIZE*IMAGE_SIZE];
    for(int i = 0; i < IMAGE_SIZE*IMAGE_SIZE; i++){
        float x = (float)image_28x28[i] / 255.0f; // [0..1]
        float x_norm = x * 2.0f - 1.0f;           // [0..1]->[-1..1]
        input[i] = x_norm;
    }

    // We'll maintain a pointer to our current buffer of activations
    // Start is "input" (784 floats)
    float* current_activations = (float*)calloc(IMAGE_SIZE*IMAGE_SIZE, sizeof(float));
    memcpy(current_activations, input, sizeof(float)*IMAGE_SIZE*IMAGE_SIZE);
    int current_size = IMAGE_SIZE*IMAGE_SIZE;

    // 2) For each layer in MLP
    //    use fully_connected -> possibly relu
    for (int layer_idx = 0; layer_idx < MLP_NUM_LAYERS; layer_idx++) {
        // get layer definition
        FCLayerDef layer_def = MLP_LAYERS[layer_idx];

        float* out = fully_connected(
            current_activations,
            layer_def.in_features,
            layer_def.weight,
            layer_def.bias,
            layer_def.out_features
        );

        free(current_activations);
        current_activations = out;
        current_size        = layer_def.out_features;

        // Activation if needed
        if (layer_def.activation_type == ACT_RELU) {
            relu_inplace(current_activations, current_size);
        }
    }

    // 3) ArgMax of last layer
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

// --------------------------------------------------------------------------
// Main
// Usage: ./mlp_inference input.bmp
// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input.bmp>\n", argv[0]);
        return 1;
    }

    const char* input_bmp_path = argv[1];

    // Preprocess the input image => 28x28
    uint8_t img_28x28[IMAGE_SIZE * IMAGE_SIZE];
    if (preprocess(input_bmp_path, img_28x28) != 0) {
        printf("Error in preprocessing.\n");
        return -1;
    }

    // Forward pass through MLP
    int prediction = mlp_forward_pass(img_28x28);

    // Print the result
    printf("Predicted digit: %d\n", prediction);

    return 0;
}
