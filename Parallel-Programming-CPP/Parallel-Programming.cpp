#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <stdint.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;
const double PI = 3.14159265358979323846;

// Негатив
void negative(const char* filename) {
    int width, height, channels;
    uint8_t* image = stbi_load(filename, &width, &height, &channels, 0);

    if (channels == 3) {
        for (int i = 0; i < width * height * 3; i++) {
            image[i] = 255 - image[i];
        }
        stbi_write_png("output.png", width, height, channels, image, width * channels);
        stbi_image_free(image);
    }
    else {
        // Мы используем uint8_t потому что это 8-ми битный беззнаковый целочисленный тип данных, который принимает значения от 0 до 255
        vector<uint8_t>output(width * height * 3);
        // Преобразование выражения из 4х канального в 3х канальный
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[(y * width + x) * 3 + 0] = image[(y * width + x) * 4 + 0];
                output[(y * width + x) * 3 + 1] = image[(y * width + x) * 4 + 1];
                output[(y * width + x) * 3 + 2] = image[(y * width + x) * 4 + 2];
            }
        }
        for (int i = 0; i < width * height * 3; i++) {
            output[i] = 255 - output[i];
        }
        stbi_write_png("output.png", width, height, 3, output.data(), width * 3);
        stbi_image_free(image);
    }
}
void negativeOpenMP(const char* filename) {
    int width, height, channels;
    uint8_t* image = stbi_load(filename, &width, &height, &channels, 0);

    if (channels == 3) {
#pragma omp parallel for
        for (int i = 0; i < width * height * 3; i++) {
            image[i] = 255 - image[i];
        }
        stbi_write_png("output1.png", width, height, channels, image, width * channels);
        stbi_image_free(image);
    }
    else {
        // Мы используем uint8_t потому что это 8-ми битный беззнаковый целочисленный тип данных, который принимает значения от 0 до 255
        vector<uint8_t>output(width * height * 3);
        // Преобразование выражения из 4х канального в 3х канальный
#pragma omp parallel for
        for (int y = 0; y < height; y++) {
#pragma omp parallel for
            for (int x = 0; x < width; x++) {
                output[(y * width + x) * 3 + 0] = image[(y * width + x) * 4 + 0];
                output[(y * width + x) * 3 + 1] = image[(y * width + x) * 4 + 1];
                output[(y * width + x) * 3 + 2] = image[(y * width + x) * 4 + 2];
            }
        }
#pragma omp parallel for
        for (int i = 0; i < width * height * 3; i++) {
            output[i] = 255 - output[i];
        }
        stbi_write_png("output1.png", width, height, 3, output.data(), width * 3);
        stbi_image_free(image);
    }
}
void negativeVectorization(const char* filename) {
    int width, height, channels;
    uint8_t* image = stbi_load(filename, &width, &height, &channels, 0);

    if (channels == 3) {
        // Переменная, содеражащая вектор, элементы которого равны 255
        __m128i image_copy = _mm_set1_epi8(255);
        for (int i = 0; i < width * height * 3; i += 16) {
            // Загрузка 32 байт данных из массива image в массив data
            __m128i data = _mm_load_si128((__m128i*)(image + i));
            // Вычитание из массива image_copy вектора data
            __m128i neg_data = _mm_sub_epi8(image_copy, data);
            // Сохранение результата
            _mm_store_si128((__m128i*)(image + i), neg_data);
        }
        stbi_write_png("output2.png", width, height, channels, image, width * channels);
        stbi_image_free(image);
    }
    else {
        // Мы используем uint8_t потому что это 8-ми битный беззнаковый целочисленный тип данных, который принимает значения от 0 до 255
        vector<uint8_t>output(width * height * 3);
        // Преобразование выражения из 4х канального в 3х канальный
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[(y * width + x) * 3 + 0] = image[(y * width + x) * 4 + 0];
                output[(y * width + x) * 3 + 1] = image[(y * width + x) * 4 + 1];
                output[(y * width + x) * 3 + 2] = image[(y * width + x) * 4 + 2];
            }
        }
        __m128i image_copy = _mm_set1_epi8(255);
        for (int i = 0; i < width * height * 3; i += 16) {
            __m128i data = _mm_load_si128((__m128i*)(output.data() + i));
            __m128i neg_data = _mm_sub_epi8(image_copy, data);
            _mm_store_si128((__m128i*)(output.data() + i), neg_data);
        }
        stbi_write_png("output2.png", width, height, 3, output.data(), width * 3);
        stbi_image_free(image);
    }
}

// Медианный фильтр
void medianFilter(unsigned char* image, int width, int height, int channels, int kernel_size)
{
    vector<unsigned char> values(kernel_size * kernel_size * channels);
    int radius = kernel_size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int count = 0;
                // Собираем значения пикселей из окна фильтра
                for (int j = -radius; j <= radius; j++) {
                    for (int i = -radius; i <= radius; i++) {
                        int xx = max(0, min(x + i, width - 1));
                        int yy = max(0, min(y + j, height - 1));
                        values[count++] = image[(yy * width + xx) * channels + c];
                    }
                }
                // Сортируем значения и выбираем медианное
                sort(values.begin(), values.end());
                image[(y * width + x) * channels + c] = values[kernel_size * kernel_size / 2];
            }
        }
    }
}
void median(const char* filename) {
    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);

    medianFilter(image, width, height, channels, 7);

    stbi_write_png("output.png", width, height, channels, image, width * channels);
    stbi_image_free(image);
}
void medianFilterOpenMP(unsigned char* image, int width, int height, int channels, int kernel_size)
{
    vector<unsigned char> values(kernel_size * kernel_size * channels);
    int radius = kernel_size / 2;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int count = 0;
                // Собираем значения пикселей из окна фильтра
#pragma omp parallel for
                for (int j = -radius; j <= radius; j++) {
                    for (int i = -radius; i <= radius; i++) {

                        int xx = max(0, min(x + i, width - 1));
                        int yy = max(0, min(y + j, height - 1));
                        values[count++] = image[(yy * width + xx) * channels + c];
                    }
                }
                // Сортируем значения и выбираем медианное (быстрая сортировка)
                sort(values.begin(), values.end());
                image[(y * width + x) * channels + c] = values[kernel_size * kernel_size / 2];
            }
        }
    }
}
void medianOpenMP(const char* filename) {
    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);

    medianFilterOpenMP(image, width, height, channels, 7);

    stbi_write_png("output1.png", width, height, channels, image, width * channels);
    stbi_image_free(image);
}
void medianFilterVectorization(unsigned char* image, int width, int height, int channels, int kernel_size)
{
    vector<unsigned char> values(kernel_size * kernel_size * channels);
    int radius = kernel_size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int count = 0;
                // Собираем значения пикселей из окна фильтра
                for (int j = -radius; j <= radius; j++) {
                    for (int i = -radius; i <= radius; i++) {
                        int xx = max(0, min(x + i, width - 1));
                        int yy = max(0, min(y + j, height - 1));
                        values[count++] = image[(yy * width + xx) * channels + c];
                    }
                }
                // Сортируем значения и выбираем медианное
                sort(values.begin(), values.end());
                image[(y * width + x) * channels + c] = values[kernel_size * kernel_size / 2];
            }
        }
    }
}
void medianVectorization(const char* filename) {
    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);

    medianFilterVectorization(image, width, height, channels, 7);

    stbi_write_png("output2.png", width, height, channels, image, width * channels);
    stbi_image_free(image);
}

int main()
{
    const char* filename = "300x300.png";

    /*for (int i = 0; i < 10; i++) {
        auto start = chrono::steady_clock::now();
        negativeVectorization(filename);
        auto end = chrono::steady_clock::now();
        cout << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 << endl;
    }*/


    for (int i = 0; i < 10; i++) {
        auto start = chrono::steady_clock::now();
        medianVectorization(filename);
        auto end = chrono::steady_clock::now();
        cout << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 << endl;
    }

    return 0;
}