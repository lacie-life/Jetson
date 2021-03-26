#include <stdio.h>
#include <cuda.h>

__global__ void AplusB(int *ret, int a, int b){
    //printf("hello\n");
    ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void set_paras(int *ret){
    ret[threadIdx.x] = ret[threadIdx.x] - 50;
}
int main(){
    int *ret;

    cudaMallocManaged(&ret, 1000*sizeof(int));

    AplusB<<<1, 1000>>>(ret, 10, 100);
    set_paras<<<1,1000>>>(ret);
    cudaDeviceSynchronize();
    

    for(int i = 0; i < 10; i++){
        printf("%d: A+B = %d \n", i , ret[i]);
    }

    cudaFree(ret);
    return 0;
}