#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define THRESHOLD 128

#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BMPFileHeader;

typedef struct {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BMPInfoHeader;
#pragma pack(pop)

// Function to save BMP file
void save_bmp(const char* output_path, uint8_t* data, int width, int height) {
    FILE* output_file = fopen(output_path, "wb");
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

// Function to flip an image vertically
void vertical_flip(uint8_t* data, int width, int height) {
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width; x++) {
            int top_idx = y * width + x;
            int bottom_idx = (height - 1 - y) * width + x;

            // Swap pixels
            uint8_t temp = data[top_idx];
            data[top_idx] = data[bottom_idx];
            data[bottom_idx] = temp;
        }
    }
}

void process_bmp(const char* input_path, const char* output_path) {
    // Open the BMP file
    FILE* input_file = fopen(input_path, "rb");
    if (!input_file) {
        printf("Error: Could not open input file %s\n", input_path);
        return;
    }

    // Read the BMP headers
    BMPFileHeader bmp_file_header;
    BMPInfoHeader bmp_info_header;
    fread(&bmp_file_header, sizeof(BMPFileHeader), 1, input_file);
    fread(&bmp_info_header, sizeof(BMPInfoHeader), 1, input_file);

    // Validate BMP format
    if (bmp_file_header.bfType != 0x4D42 || bmp_info_header.biBitCount != 24) {
        printf("Error: Unsupported BMP format (must be 24-bit BMP)\n");
        fclose(input_file);
        return;
    }

    int width = bmp_info_header.biWidth;
    int height = abs(bmp_info_header.biHeight);
    int row_padded = (width * 3 + 3) & ~3;

    // Allocate memory for the BMP pixel data
    uint8_t* bmp_data = (uint8_t*)malloc(height * row_padded);
    fseek(input_file, bmp_file_header.bfOffBits, SEEK_SET);
    fread(bmp_data, 1, height * row_padded, input_file);
    fclose(input_file);

    // Convert BMP to grayscale and threshold
    uint8_t* grayscale = (uint8_t*)malloc(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * row_padded + x * 3;
            uint8_t r = bmp_data[idx + 2];
            uint8_t g = bmp_data[idx + 1];
            uint8_t b = bmp_data[idx];
            uint8_t gray = (uint8_t)(0.3 * r + 0.59 * g + 0.11 * b);
            grayscale[y * width + x] = (gray > THRESHOLD) ? 255 : 0;
        }
    }
    free(bmp_data);

    // Find bounding box
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

    // Crop the image
    int cropped_width = x_max - x_min + 1;
    int cropped_height = y_max - y_min + 1;
    uint8_t* cropped = (uint8_t*)malloc(cropped_width * cropped_height);
    for (int y = 0; y < cropped_height; y++) {
        for (int x = 0; x < cropped_width; x++) {
            cropped[y * cropped_width + x] = grayscale[(y_min + y) * width + (x_min + x)];
        }
    }
    free(grayscale);

    // Resize to 28x28
    int resized_width = 28, resized_height = 28;
    uint8_t* resized = (uint8_t*)malloc(resized_width * resized_height);
    for (int y = 0; y < resized_height; y++) {
        for (int x = 0; x < resized_width; x++) {
            int src_x = x * cropped_width / resized_width;
            int src_y = y * cropped_height / resized_height;
            resized[y * resized_width + x] = cropped[src_y * cropped_width + src_x];
        }
    }
    free(cropped);

    // Apply vertical flip
    vertical_flip(resized, resized_width, resized_height);

    // Save the resized and flipped image as BMP
    save_bmp(output_path, resized, resized_width, resized_height);

    free(resized);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_bmp_path> <output_bmp_path>\n", argv[0]);
        return -1;
    }

    process_bmp(argv[1], argv[2]);
    return 0;
}
