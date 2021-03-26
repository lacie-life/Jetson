#include <stdio.h>
#include <cuda.h>

__global__ void AplusB(int *ret, int a, int b){
    //printf("hello\n");
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main() {
    int *ret;

    cudaMalloc( (void**)&ret, 1000*sizeof(int));

    AplusB<<<1, 1000>>>(ret, 10, 100);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("add start exceuting! \n");
        printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
        printf("\n");
    }

    int *host_ret = (int *)malloc(1000*sizeof(int));

    cudaMemcpy(host_ret, ret, 1000*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++){
        printf("%d: A+B = %d \n", i, host_ret[i]);
    }

    free(host_ret);
    cudaFree(ret);
    return 0;
}

