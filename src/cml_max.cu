#include <time.h>
#include <sys/time.h>
//#include <array>>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <cblas.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define L 140
#define L_power 19600


__global__ void reduction_kernel(float *d_in,float *d_out,int N){
	extern __shared__ float sp[];
	const int tid = threadIdx.x;
	float sum=0.0;
	for (size_t i = blockIdx.x*blockDim.x + tid;
		i < N ;
		i += blockDim.x*gridDim.x){
		sum += d_in[i];
	}
	sp[tid] = sum;
	__syncthreads();
	
	int floorpow = blockDim.x;
	if (floorpow & (floorpow -1)){
		while (floorpow & (floorpow -1)){
			floorpow &= floorpow -1;
		}
		if (tid >= floorpow){
			sp[tid-floorpow]+= sp[tid];
		}
		__syncthreads();
	}
	
	for (int activethreads = floorpow >>1;
		activethreads;
		activethreads >>=1){
		if (tid < activethreads){
			sp[tid]+= sp[tid+activethreads];
		}
		__syncthreads();
	}
	if (tid==0){
		d_out[blockIdx.x]=sp[0];
		}
	}
__global__ void reduction2_kernel(float *d_in,float *d_out,int N){
	extern __shared__ float sp[];
	const int tid = threadIdx.x;
	float maxvalue=d_in[blockIdx.x*blockDim.x+tid];
	for (size_t i = blockIdx.x*blockDim.x + tid;
		i < N ;
		i += blockDim.x*gridDim.x){
		maxvalue=fmaxf(maxvalue,d_in[i]);
	}
	sp[tid] = maxvalue;
	__syncthreads();
	
	int floorpow = blockDim.x;
	if (floorpow & (floorpow -1)){
		while (floorpow & (floorpow -1)){
			floorpow &= floorpow -1;
		}
		if (tid >= floorpow){
			sp[tid-floorpow]=fmaxf(sp[tid-floorpow],sp[tid]);
		}
		__syncthreads();
	}
	
	for (int activethreads = floorpow >>1;
		activethreads;
		activethreads >>=1){
		if (tid < activethreads){
//			sp[tid]+= sp[tid+activethreads];
			sp[tid]=fmaxf(sp[tid],sp[tid+activethreads]);
		}
		__syncthreads();
	}
	if (tid==0){
		d_out[blockIdx.x]=sp[0];
		}
	}
__global__ void reduction3_kernel(float *d_in,int *d_in_index,float *d_out,int *d_out_index,int N){
	extern __shared__ float sp[];
	extern __shared__ int ip[];
	const int tid = threadIdx.x;
	float maxvalue=d_in[blockIdx.x*blockDim.x+tid];
	int maxindex=d_in_index[blockIdx.x*blockDim.x+tid];
	for (size_t i = blockIdx.x*blockDim.x + tid;
		i < N ;
		i += blockDim.x*gridDim.x){
		maxvalue=fmaxf(maxvalue,d_in[i]);
		if (maxvalue==d_in[i]){
		//	maxvalue=d_in[i];
			maxindex=d_in_index[i];
			}
	}
	sp[tid] = maxvalue;
	ip[tid] = maxindex;
	__syncthreads();
	
	int floorpow = blockDim.x;
	if (floorpow & (floorpow -1)){
		while (floorpow & (floorpow -1)){
			floorpow &= floorpow -1;
		}
		if (tid >= floorpow){
			sp[tid-floorpow]=fmaxf(sp[tid-floorpow],sp[tid]);
			
			if (sp[tid-floorpow]==sp[tid]){
//				sp[tid-floorpow]=sp[tid];
				ip[tid-floorpow]=ip[tid];
				}
		__syncthreads();
		}
		__syncthreads();
	}
	
	for (int activethreads = floorpow >>1;
		activethreads;
		activethreads >>=1){
		if (tid < activethreads){
//			sp[tid]+= sp[tid+activethreads];
			sp[tid]=fmaxf(sp[tid],sp[tid+activethreads]);
			if (sp[tid]==sp[tid+activethreads]){
//				sp[tid]=sp[tid+activethreads];
				ip[tid]=ip[tid+activethreads];
				}
		}
		__syncthreads();
	}
	if (tid==0){
		d_out[blockIdx.x]=sp[0];
		d_out_index[blockIdx.x]=ip[0];
		}
	}
__global__ void my_reduction_kernel1(float *d_in,float *d_out,int *d_out_index,int N){
	const int globalid=(blockDim.x*blockIdx.x+threadIdx.x);
	float local_max=d_in[globalid];
	int local_max_index=globalid;
	for (int i=1;i<N;i++){
		local_max=fmaxf(local_max,d_in[globalid+i*N]);
		if (local_max==d_in[globalid+i*N]){
			local_max_index=globalid+i*N;
			}
		}
	d_out[threadIdx.x]=local_max;
	d_out_index[threadIdx.x]=local_max_index;
	}
__global__ void my_reduction_kernel2(float *d_in,int *index_in,float *d_out,int *d_out_index,int N){
	float local_max=d_in[0];
	int max_index=index_in[0];
	for (int i=1;i<N;i++){
		local_max=fmaxf(local_max,d_in[i]);
		if (local_max==d_in[i]){
			max_index=index_in[i];
			}
		}
	*d_out=local_max;
	*d_out_index=max_index;
}
__global__ void simple_kel(float *d_in,float *d_out,int *d_out_index,int N){
	float local_max=d_in[0];
	int max_index=0;
	int NP=N*N;
	for (int i=1;i<NP;i++){
		local_max=fmaxf(local_max,d_in[i]);
		if (local_max==d_in[i]){
			max_index=i;
			}
		}
	*d_out=local_max;
	*d_out_index=max_index;
}

__global__ void com_kernel(float *d_in,float *d_p,int *d_pi,float *d_out,int *d_out_index,int N){
    my_reduction_kernel1<<<1,L>>>(d_in,d_p,d_pi,N);
    my_reduction_kernel2<<<1,1>>>(d_p,d_pi,d_out,d_out_index,L);
}
	
		
int main(){
	float A[L_power];
	int index[L_power];
	for (int i=0;i<L_power;i++){
		A[i]=float(i);
		index[i]=i;
		}
	A[299]=40300.0;
	float h_sum;
	int  h_max;
	float *d_A;
	int *d_index;
	float *d_sum;
	float *d_p;
	int *d_pi;
	int *d_max;
	cudaMalloc((void **)&d_A,sizeof(float)*L_power);
	cudaMalloc((void **)&d_index,sizeof(int)*L_power);
	cudaMalloc((void **)&d_sum,sizeof(float));
	cudaMalloc((void **)&d_max,sizeof(int));
	cudaMalloc((void **)&d_p,sizeof(float)*L);
	cudaMalloc((void **)&d_pi,sizeof(int)*L);
	
	cudaMemcpy(d_A,A,sizeof(float)*L_power,cudaMemcpyHostToDevice);
	cudaMemcpy(d_index,index,sizeof(int)*L_power,cudaMemcpyHostToDevice);

//	reduction3_kernel<<<70,70,70*(sizeof(float)+sizeof(int))>>>(d_A,d_index,d_p,d_pi,L_power);
//	reduction3_kernel<<<1,70,70*(sizeof(float)+sizeof(int))>>>(d_p,d_pi,d_sum,d_max,70);
//	reduction2_kernel<<<70,70,70*(sizeof(float)+sizeof(int))>>>(d_A,d_p,L_power);
//	reduction2_kernel<<<1,70,70*(sizeof(float)+sizeof(int))>>>(d_p,d_sum,70);
	
	my_reduction_kernel1<<<1,L>>>(d_A,d_p,d_pi,L);
	my_reduction_kernel2<<<1,1>>>(d_p,d_pi,d_sum,d_max,L);
	simple_kel<<<1,1>>>(d_A,d_sum,d_max,L);
	com_kernel<<<1,1>>>(d_A,d_p,d_pi,d_sum,d_max,L);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_sum,d_sum,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_max,d_max,sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_sum);
	cudaFree(d_index);
	cudaFree(d_max);
	cudaFree(d_pi);
	cudaFree(d_A);
	cudaFree(d_p);
	
	printf("sum is %f\n",h_sum);
	printf("index is %d\n",h_max);
	return 0;
}	
