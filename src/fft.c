#include "fftw3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4
#define REAL 0
#define IMAG 1
#define PI 3.1415926535898

int main (void)
{
    fftw_complex *in, *out;
    fftw_plan p;
    double constants[N] = {10, 2.1, 4.7, 1.3};
    double f;
    int i, j;

    /* Allocate memory for the arrays */
    in = fftw_malloc(sizeof(fftw_complex) * N);
    out = fftw_malloc(sizeof(fftw_complex) * N);

    if ((in == NULL) || (out == NULL)) {
 printf ("Error: insufficient available memory\n");
    }
    else {
 /* Create the FFTW execution plan */
 p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
 /* Initialize the input data */
 for (i = 0; i < N; i++) { /* All sampling points */
     in[i][REAL] = constants[0];
     in[i][IMAG] = 0;
     for (j = 1; j < N; j++) { /* All frequencies */
  in[i][REAL] += constants[j] * cos(j * i * 2 * PI / (double)N);
  in[i][IMAG] += constants[j] * sin(j * i * 2 * PI / (double)N);
     }
 }

 /* Execute plan */
 fftw_execute(p);

 /* Destroy plan */
 fftw_destroy_plan(p);

 /* Display results */
 printf ("Constants[] = {");
 for (i = 0; i < N; i++)
     printf("%lf%s", constants[i], (i == N-1) ? "}\n" : ", ");

 printf ("Input[][REAL] = {");
 for (i = 0; i < N; i++)
     printf("%lf%s", in[i][REAL], (i == N-1) ? "}\n" : ", ");

 printf ("Output[][REAL] = {");
 for (i = 0; i < N; i++)
     printf("%lf%s", out[i][REAL], (i == N-1) ? "}\n" : ", ");

 /* Scale output */
 f = 1.0/sqrt((double)N);
 for (i = 0; i < N; i++)
     out[i][REAL] *= f;

 /* Display final results */
 printf ("Scaled[][REAL] = {");
 for (i = 0; i < N; i++)
     printf("%lf%s", out[i][REAL], (i == N-1) ? "}\n" : ", ");
    }

    /* Free allocated memory */
    if (in != NULL) fftw_free(in);
    if (out != NULL) fftw_free(out);

    return 0;
}
