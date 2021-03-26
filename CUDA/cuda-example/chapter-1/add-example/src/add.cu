#include <iostream>
#include <cuda.h>
#include "../../../common/book.h"
#include <stdio.h>

__global__ void add( int a, int b, int *c ) {
    printf("Hello from GPU !!! \n");
    *c = a + b;
}

int main( void ) {
    int c;
    int *dev_c;

    cudaMalloc( (void**)&dev_c, sizeof(int));

    add<<<1,1>>>( 2, 7, dev_c );

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("add start exceuting! \n");
        printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
        printf("\n");
    }

    cudaMemcpy( &c,
                dev_c,
                sizeof(int),
                cudaMemcpyDeviceToHost );
    
    printf( "2 + 7 = %d\n", c );

    cudaFree( dev_c );

    return 0;
}