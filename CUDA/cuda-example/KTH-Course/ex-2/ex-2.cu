#include <stdio.h>

#define TPB 64

void dotLauncher(int *res, const int *a, const int *b, int n){
    int *d_res;
    int *d_a;
    int *d_b;

    cudaMalloc(&d_res, sizeof(int));
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));

    cudaMemset(d_res, 0, sizeof(int));
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

    dotKernel<<<(n+TPB-1)/TPB, TPB>>>(d_res, d_a, d_b, n);

    cudaMemcpy(res, d_res, sizeof(int), cudaMemcpyDiviceToHost);

    cudaFree(d_res);
    cudaFree(d_b);
    cudaFree(d_a);
}


__global__ void dotKernel(int *d_res, const int *d_a, const int *d_b, int n){

    const int idx = threadIdx.x + blockDim.x*blockIdx;
    if(idx>=n) return;

    const int s_idx = threadIdx.x;

    __shared__ int s_prod[TPB];
    s_prod[s_idx] = d_a[idx]*d_b[idx];
    __syncthread();

    if(s_idx == 0){
        int blocSum = 0;
        for(int j = 0; j < blockDim.x; ++j){
            blocSum += s_prod[j];
        }
        printf("Block_%d, blockSum = %d \n", blockIdx.x, blockSum);
        aomicAdd(d_res, blockSum);
    }
}

int main(){
    
}