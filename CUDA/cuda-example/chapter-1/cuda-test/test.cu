#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#define N  5

__global__ void add(int *a, int *b, int *c)
{
        printf("after copy ....\n");
        for (int i = 0; i<N; i++)
        {
                printf("%d + %d ? \n", a[i], b[i]);
        }
        int tid = blockIdx.x;
        if (tid<N){
            c[tid] = a[tid] + b[tid];
            printf("Result: %d + %d = %d \n",a[tid], b[tid], c[tid]);
        }  
}

int main(void)
{
        int a[N],b[N],c[N];
        int *dev_a, *dev_b, *dev_c;

        cudaError_t rc = cudaMalloc((void **) &dev_a, N*sizeof(int));

        if (rc != cudaSuccess)
            printf("Could not allocate memory: %d", rc);

        cudaMalloc((void**)&dev_b, N * sizeof(int));
        cudaMalloc((void**)&dev_c, N * sizeof(int));

        for (int i = 0; i<N; i++)
        {
                a[i] = i;
                b[i] = i;
                printf("%d + %d ? \n", a[i], b[i]);
        }

        cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        
        add<<<N,1>>>(dev_a, dev_b, dev_c);
        cudaThreadSynchronize();
        
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("add start exceuting! \n");
            printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
            printf("\n");
        }
        
        cudaError_t err2 = cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
        if(err2 != cudaSuccess){
            printf("The error is %s", cudaGetErrorString(err2));
            printf("\n");
        }

        for(int i =0; i<N; i++)
            printf("%d + %d = %d\n",a[i],b[i],c[i]);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return EXIT_SUCCESS;
}