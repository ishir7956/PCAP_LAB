#include <stdio.h>
#include <string.h>
#include <cuda.h>

const int MaxThreadsPerBlock = 256;

__global__ void countWordKernel(const char* sentence, const char* word, int* count, int sentenceLength, int wordLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wordCount = 0;

    for (int i = tid; i < sentenceLength - wordLength + 1; i += blockDim.x * gridDim.x) {
        int j;
        bool isMatch = true;

        for (j = 0; j < wordLength; ++j) {
            if (sentence[i + j] != word[j]) {
                isMatch = false;
                break;
            }
        }

        if (isMatch) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const char* sentence = "hello my name is hello and I live in hello";
    const char* wordToCount = "hello";

    const char* d_sentence;
    const char* d_word;
    int* d_count;
    int sentenceLength = strlen(sentence);
    int wordLength = strlen(wordToCount);
    int count = 0;

    cudaMalloc((void**)&d_sentence, sentenceLength * sizeof(char));
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy((void*)d_sentence, sentence, sentenceLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_word, wordToCount, wordLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    int numThreadsPerBlock = MaxThreadsPerBlock;
    int numBlocks = (sentenceLength + numThreadsPerBlock - 1) / numThreadsPerBlock;
    countWordKernel<<<numBlocks, numThreadsPerBlock>>>(d_sentence, d_word, d_count, sentenceLength, wordLength);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree((void*)d_sentence);
    cudaFree((void*)d_word);
    cudaFree(d_count);

    printf("The word '%s' appears %d times in the sentence.\n", wordToCount, count);

    return 0;
}
