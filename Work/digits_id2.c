#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/*
   Inclure le fichier qui contient tous vos poids et biais, par ex. "weights.h"
   généré via le script Python d'export. Il doit définir :
   - static const float FC1_WEIGHT[128][784];
   - static const float FC1_BIAS[128];
   - static const float FC2_WEIGHT[64][128];
   - static const float FC2_BIAS[64];
   - static const float FC3_WEIGHT[10][64];
   - static const float FC3_BIAS[10];
*/
#include "weights.h"

// --------------------------------------------------------------------
// Fonctions utilitaires du MLP
// --------------------------------------------------------------------
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// --------------------------------------------------------------------
// Structures d'en‐tête BMP basiques (24 bits, non compressé)
// --------------------------------------------------------------------
#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;       // 0x4D42 = "BM"
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BMPFileHeader;

typedef struct {
    uint32_t biSize;
    int32_t  biWidth;      // doit valoir 32
    int32_t  biHeight;     // doit valoir 32
    uint16_t biPlanes;
    uint16_t biBitCount;   // doit valoir 24
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BMPInfoHeader;
#pragma pack(pop)

/*
   read_bmp_32x32_then_resize_28x28 :
   1) Lit un BMP 32×32, 24 bits/pixel non compressé.
   2) Convertit chaque pixel en niveau de gris.
   3) Redimensionne de 32×32 vers 28×28 par nearest neighbor.
   4) Normalise en [-1..+1] (identique à la Normalize((0.5,),(0.5,))).

   out[784] contiendra finalement 28×28 = 784 valeurs normalisées.
*/
int read_bmp_32x32_then_resize_28x28(const char *filename, float out[28*28])
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        perror("Echec d'ouverture du fichier BMP");
        return -1;
    }

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    fread(&fileHeader, sizeof(fileHeader), 1, fp);
    fread(&infoHeader, sizeof(infoHeader), 1, fp);

    // Vérifications minimales
    if(fileHeader.bfType != 0x4D42) {
        fprintf(stderr, "Erreur: format BMP invalide\n");
        fclose(fp);
        return -2;
    }
    if(infoHeader.biWidth != 32 || infoHeader.biHeight != 32 || infoHeader.biBitCount != 24) {
        fprintf(stderr, "Erreur: ce code requiert un BMP 24 bits 32×32\n");
        fclose(fp);
        return -3;
    }

    // Se positionner sur la zone de données
    fseek(fp, fileHeader.bfOffBits, SEEK_SET);

    // Stockage temporaire du gris en 32×32 (sans normalisation), pour ensuite redimensionner
    static float gray32[32*32];

    // Chaque ligne est alignée sur un multiple de 4 octets
    int rowSizeBytes = (infoHeader.biWidth * 3 + 3) & ~3; // 32*3=96, déjà multiple de 4 => rowSizeBytes=96
    for(int y = 31; y >= 0; y--) {
        for(int x = 0; x < 32; x++) {
            unsigned char bgr[3];
            fread(bgr, 1, 3, fp);
            // Conversion en gris (pondérée)
            float gray = 0.299f*(float)bgr[2] + 0.587f*(float)bgr[1] + 0.114f*(float)bgr[0];
            gray32[y*32 + x] = gray; // en [0..255]
        }
        // Sauter éventuellement le padding (si rowSizeBytes > 32*3)
        int padding = rowSizeBytes - (32*3);
        if(padding > 0) {
            fseek(fp, padding, SEEK_CUR);
        }
    }
    fclose(fp);

    // --------------------------------------------------------
    // Redimensionnement nearest neighbor 32×32 -> 28×28
    // --------------------------------------------------------
    // out[y*28 + x] = gray32[ round( y * (32/28) ) * 32 + round( x*(32/28) ) ]
    // On va plutôt prendre floor / cast (int) comme nearest neighbor simple
    for(int y = 0; y < 28; y++) {
        for(int x = 0; x < 28; x++) {
            float srcY = (float)y * 32.0f / 28.0f; // ex. y=0 => 0
            float srcX = (float)x * 32.0f / 28.0f; // ex. x=27 => ~30.85
            int sy = (int)(srcY + 0.5f); // on peut arrondir
            int sx = (int)(srcX + 0.5f);
            if(sy < 0) sy = 0; if(sy > 31) sy = 31;
            if(sx < 0) sx = 0; if(sx > 31) sx = 31;

            float g = gray32[sy*32 + sx];
            // Normalisation => [-1..+1]
            float norm_val = (g/255.0f - 0.5f)/0.5f;
            out[y*28 + x] = norm_val;
        }
    }

    return 0;
}

int main(int argc, char *argv[])
{
    if(argc < 2) {
        fprintf(stderr, "Usage : %s <bmp_32x32_24bpp>\n", argv[0]);
        return 1;
    }

    // 1) Lire et redimensionner l'image dans input[784]
    float input[28*28];
    if(read_bmp_32x32_then_resize_28x28(argv[1], input) != 0) {
        fprintf(stderr, "Erreur de lecture/format BMP\n");
        return 2;
    }

    // 2) Couche fc1 + ReLU
    float layer1[128];
    for(int i=0; i<128; i++) {
        float sum = 0.0f;
        for(int j=0; j<784; j++) {
            sum += FC1_WEIGHT[i][j] * input[j];
        }
        sum += FC1_BIAS[i];
        layer1[i] = relu(sum);
    }

    // 3) Couche fc2 + ReLU
    float layer2[64];
    for(int i=0; i<64; i++) {
        float sum = 0.0f;
        for(int j=0; j<128; j++) {
            sum += FC2_WEIGHT[i][j] * layer1[j];
        }
        sum += FC2_BIAS[i];
        layer2[i] = relu(sum);
    }

    // 4) Couche fc3 (logits) et argmax
    float logits[10];
    for(int i=0; i<10; i++) {
        float sum = 0.0f;
        for(int j=0; j<64; j++) {
            sum += FC3_WEIGHT[i][j] * layer2[j];
        }
        sum += FC3_BIAS[i];
        logits[i] = sum;
    }

    int predicted_class = 0;
    float max_val = logits[0];
    for(int i=1; i<10; i++) {
        if(logits[i] > max_val) {
            max_val = logits[i];
            predicted_class = i;
        }
    }

    printf("Classe prédite : %d\n", predicted_class);
    return 0;
}
