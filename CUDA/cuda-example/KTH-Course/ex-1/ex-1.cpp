#include <math.h>
#include <stdio.h>

#define N 64

float scale(int i, int n){
    return ((float)i)/(n -1);
}

float distnace(float x1, float x2){
    return sqrt((x2-x1)*(x2-x1));
}

int main(){
    float out[N];

    const float ref = 0.5;

    for(int i = 0; i < N; i++){
        float x = scale(i, N);
        out[i] = distnace(x, ref);
    }

    for (int i = 0; i < N; i++){
        printf("distance %d: %f \n", i, out[i]);
    }
    return 0;
}