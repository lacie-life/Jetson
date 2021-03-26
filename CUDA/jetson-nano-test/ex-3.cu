#include <stdio.h>
#include <cuda.h>
__device__ __managed__ int ret[1000];
__global__ void AplusB(int a, int b){
    //printf("hello\n");
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main(){
    int *ret;

    AplusB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i++){
        printf("%d: A+B = %d \n", i , ret[i]);
    }

    // cudaFree(ret);
    return 0;
}