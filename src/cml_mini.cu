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
/*
 *this program is used to test cuda-function
 * 1.can cublas in kernel read data correctly?
 * 2.can cublas in kernel compute correctly?
 * 3.the data returened by kernel is correctly?
 *
 *
 *
 *
 */


/*
__global__ void mykernel2(float *d_data,float *max_value,cublasHandle_t handle){
	cublasStatus_t status;
	const float alpha=1.0;
	const float beta=0.0;

}
*/
float MYSUM(int Num,const float *p){
    float re=0.0f;
    int i;
    for(i=0;i<Num;i++){
        re=re+p[i];
    }
    return re;
}

__global__ void simplekernel(float *d_process,float *d_max_value){
    float maxvalue;
    int i;
    maxvalue=d_process[0];
    for (i=0;i<L_power;i++){
        if (maxvalue<d_process[i]){
            maxvalue=d_process[i];
        }
    }
    *d_max_value=maxvalue;
}

__global__ void testlambda(float *d_process){
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	d_process[id]=d_process[id]*1.0;	
}

__global__ void flambda(float *d_process,float *d_mean,float *d_sum,float *d_stdv,int image_a,int image_b){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    int i=blockIdx.x;
    int j=threadIdx.x;
    d_process[id]=(d_process[id]+L*d_mean[image_a*L+i]*d_mean[image_b*L+j]-d_sum[image_a*L+i]*d_mean[image_b*L+j]-d_mean[image_a*L+i]*d_sum[image_b*L+j])/(L*d_stdv[image_a*L+i]*d_stdv[image_b*L+j]);
}

__global__ void reduction_kernel(float *d_process,float *d_max_value,int *d_max_index,int N){
	extern __shared__ float mPartials[];
	extern __shared__ int iPartials[];
	const int tid = threadIdx.x;
	const int gid = blockDim.x*blockDim.x + tid;
	float max_value = d_process[gid];
	int max_index = gid;
	for (size_t i = gid;i<N;i +=blockDim.x*gridDim.x){
		max_value=fmaxf(max_value,d_process[i]);
		if (max_value==d_process[i]){
			max_index=i;
			}
		}
	mPartials[tid]=max_value;
	iPartials[tid]=max_index;
	__syncthreads();	
	for (int activeThreads = blockDim.x>>1;
		 activeThreads ;
		 activeThreads >>= 1){
		if ( tid < activeThreads){
			mPartials[tid]=fmaxf(mPartials[tid],mPartials[tid+activeThreads]);
			if (mPartials[tid]==mPartials[tid+activeThreads]){
				iPartials[tid]=iPartials[tid+activeThreads];
				}
			}
			__syncthreads();
		}
	if (tid==0){
		d_max_value[blockIdx.x] = mPartials[0];
		d_max_index[blockIdx.x] = iPartials[0];
		}
}

__global__ void reduction6_kernel(float *d_in,int *d_in_index,float *d_out_max_value,int *d_out_max_index,int N){
		extern __shared__ float mPartials[];
		extern __shared__ int iPartials[];
		const int tid = threadIdx.x;
		const int gid = blockIdx.x*blockDim.x + tid;
		float max_value = d_in[gid];
		int max_index = gid;
		for (size_t i = gid; i < N ; i += blockDim.x*gridDim.x ){
			max_value = fmaxf(max_value,d_in[i]);
			if (max_value==d_in[i]){
//				max_index=i;
				max_index=d_in_index[i];
				}
			}
		mPartials[tid] = max_value;
		iPartials[tid] = max_index;
		__syncthreads();

		int floorPow2 = blockDim.x;
		if ( floorPow2 & (floorPow2 -1 ) ){
			while (floorPow2 & (floorPow2 -1)){
				floorPow2 &= floorPow2 -1;
				}
			if (tid >= floorPow2){
				mPartials[tid-floorPow2] = fmaxf(mPartials[tid-floorPow2],mPartials[tid]);
				if (mPartials[tid-floorPow2]==mPartials[tid]){
					iPartials[tid-floorPow2]=iPartials[tid];
					}
				}
			__syncthreads();
		}
		
		for (int activeThreads = floorPow2>>1;
			activeThreads;
			activeThreads >>= 1){
			if (tid < activeThreads){
				mPartials[tid] = fmaxf(mPartials[tid],mPartials[tid+activeThreads]);
				if (mPartials[tid]==mPartials[tid+activeThreads]){
					iPartials[tid]=iPartials[tid+activeThreads];
				}
			}
			__syncthreads();
		}
		if (tid == 0){
			d_out_max_value[blockIdx.x] = mPartials[0];
			d_out_max_index[blockIdx.x] = iPartials[0];
			}
	}


__global__ void mykernel(float *d_data,float *d_result,cublasHandle_t handle){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    cublasHandle_t handles;
    cublasStatus_t status;
    status=cublasCreate(&handles);
    const float alpha=1.0;
    const float beta=0.0;
    const int si=140;
    float *C=(float*)malloc(sizeof(float)*si*si);
//    float *C;
//    cudaMalloc((void **)&C,sizeof(float)*si*si); 
    status=cublasSgemm(handles,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,d_data,L,d_data,L,&beta,C,L);
    for (int i=0;i<L_power;i++){
		d_result[i]=C[i];
	}
    cublasDestroy(handles);
    free(C);
//    cudaFree(C);
}
/*
__global__ void mykernel_3(float *d_data,float *d_max_value){
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha=1.0;
	const float beta=0.0;
	const int si=140;
	int i=0;
	float maxvalue;
	float *C=(float*)malloc(sizeof(float)*si*si);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,d_data,L,d_data,L,&beta,C,L);
	for (i=0;i<L_power;i++){
		if (maxvalue	


}
*/

//lambdakernel<<<140,140,0,stream[i]>>>(&d_result[L_power*i],&
/*
__global__ void lambdakernel(float *d_process,){
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	int i=blockIdx.x;
	int j=threadIdx.x;
	d_process[id]=d_process[id]+1;
}
*/

/*
__global__  void nocublas_kernel(float *d_data,float *d_result){
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i,j,k,m;
	for (i=0;i<L;i++){
		for (j=0;j<L;j++){
			d_result[id*L_power+i*L+j]=0.0;
			for (k=0;k<L;k++){
				for (m=0;m<L;m++){
					d_result[id*L_power+i*L+j]+=d_data[L_power*id+i*L+k]*d_data[L_power*id+m*L+j];
					}
				}
			}
		}

}
*/

int main(int argc,char *argv[]){
    int oc;
    FILE *fdata;
    char *datafilename;
    while((oc = getopt(argc, argv,"f:")) !=-1){
        switch(oc)
        {
        case 'f':
            datafilename=optarg;
            break;
        }
    }
//    datafilename="~/..";
    fdata=fopen(datafilename,"rb");
    float *matrix;
    const float alpha=1.0;
    const float beta=0.0;
    int N=50;
    matrix = new float [L_power*N];
    fseek(fdata,0,SEEK_SET);
    fread(matrix,sizeof(float),N*L_power,fdata);

    float *result;
    result = new float [L_power*N];

    float *d_data;
    float *d_result;
//    float *d_process[N];
    float max_value[N];
    int index[N];
    float *d_max_value;
    int *d_index;
    cudaMalloc((void **) &d_data,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_result,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_max_value,sizeof(float)*N);
    cudaMalloc((void **) &d_index,sizeof(int)*N);

    cudaMemcpy(d_data,matrix,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
	
    float *d_maxvalue_partial;
    int *d_index_partial;

    int *std_index;
    std_index = new int [L_power];
    for (int i=0;i<L_power;i++){
	std_index[i]=i;
	}
    int *d_std_index;
    cudaMalloc((void **) &d_std_index,sizeof(int)*L_power);
    cudaMemcpy(d_std_index,std_index,sizeof(int)*L_power,cudaMemcpyHostToDevice);
    

    float **total_nccq[N];
    for (int i=0;i<N;i++){
	total_nccq[i]= new float* [L];
	}
    for (int i=0;i<N;i++){
	for (int j=0;j<L;j++){
		total_nccq[i][j] = new float[4];
		}
	}
    float *my_mean;
    float *my_sum;
    float *my_dot;
    float *my_stdv;
    my_mean = new float [N*L];
    my_sum = new float [N*L];
    my_dot = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
	for (int j=0;j<L;j++){
	int postion=i*L_power+j*L;
//	total_nccq[i][j][0] = MYSUM(L,&matrix[postion]);
	my_sum[i*L+j] = MYSUM(L,&matrix[postion]);
//	total_nccq[i][j][1] = total_nccq[i][j][0] / L;
	my_mean[i*L+j] = my_sum[i*L+j] / L;
//	total_nccq[i][j][2] = cblas_sdot( L, &matrix[postion], 1, &matrix[postion], 1);
	my_dot[i*L+j] = cblas_sdot(L , &matrix[postion], 1,  &matrix[postion],1);
//	total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + L*total_nccq[i][j][1]*total_nccq[i][j][1]- 2*total_nccq[i][j][0]*total_nccq[i][j][1])/L);
	my_stdv[i*L+j] = sqrt((my_dot[i*L+j]+L*my_mean[i*L+j]*my_mean[i*L+j]-2*my_mean[i*L+j]*my_sum[i*L+j])/L);
		}
	}
     float *d_mean;
     float *d_sum;
     float *d_stdv;
     cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
     cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
     cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

     cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
     cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
     cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);
	
//     float *d_buffer;
//     cudaMalloc((void **)&d_buffer,sizeof(float)*N*L_power);


//    mykernel<<<1,10>>>(d_data,d_result);
    //    nocublas_kernel<<<1,10>>>(d_data,d_result);
//    cudaDeviceSynchronize();
    //cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);

    cudaStream_t stream[N];
    cublasHandle_t handle[N];
    for(int i=0;i<N;i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
    }
    for(int i=0;i<N;i++){
        cublasSetStream(handle[i],stream[i]);
    }
    int numofBlocks=10;
    cudaMalloc((void **)&d_maxvalue_partial,sizeof(float)*N*numofBlocks);
    cudaMalloc((void **)&d_index_partial,sizeof(int)*N*numofBlocks);
    for(int i=0;i<N;i++){
        cublasSgemm(handle[i],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[L_power*i],L,&d_data[L_power*i],L,&beta,&d_result[L_power*i],L);
	//mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i],handle[i]);
//        testlambda<<<L,L,0,stream[i]>>>(&d_result[L_power*i]);
        flambda<<<L,L,0,stream[i]>>>(&d_result[L_power*i],d_mean,d_sum,d_stdv,i,i);
	reduction6_kernel<<<10,32,32*(sizeof(float)+sizeof(int)),stream[i]>>>(&d_result[L_power*i],d_std_index,&d_maxvalue_partial[i*10],&d_index_partial[i*10],L_power);
	reduction6_kernel<<<1,32,32*(sizeof(float)+sizeof(int)),stream[i]>>>(&d_maxvalue_partial[i*10],&d_index_partial[i*10],&d_max_value[i],&d_index[i],10);
//        simplekernel<<<1,1,0,stream[i]>>>(&d_result[L_power*i],&d_max_value[i]);
    //mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i]);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);
    cudaMemcpy(max_value,d_max_value,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(index,d_index,sizeof(int)*N,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i=0;i<N;i++){
	cublasDestroy(handle[i]);
	cudaStreamDestroy(stream[i]);
	}
    float *Host_result;
    float Host_max[N];
    int Host_index[N];
    Host_result = new float [N*L_power];
    for (int i=0;i<N;i++){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,L,L,L,1,&matrix[i*L_power],L,&matrix[i*L_power],L,0,&Host_result[i*L_power],L);
    }
    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            for (int k=0;k<L;k++){
                Host_result[i*L_power+j*L+k]=(Host_result[i*L_power+j*L+k]+L*my_mean[i*L+j]*my_mean[i*L+k]-my_sum[i*L+j]*my_mean[i*L+k]-my_sum[i*L+k]*my_mean[i*L+j])/(L*my_stdv[i*L+j]*my_stdv[i*L+k]);
            }
        }
    }
    for (int i=0;i<N;i++){
        Host_max[i]=Host_result[i*L_power];
	Host_index[i]=0;
        for (int j=0;j<L_power;j++){
                if (Host_max[i]<Host_result[i*L_power+j]){
                    Host_max[i]=Host_result[i*L_power+j];
		    Host_index[i]=j;
                }
        }
    }
    
    for (int i=0;i<N;i++){
        float diff=0.0f;
        for (int j=0;j<L_power;j++){
            diff+=(Host_result[i*L_power+j]-result[i*L_power+j])*(Host_result[i*L_power+j]-result[i*L_power+j]);
        }
        printf("diff %d\t%f\n",i,sqrt(diff/(L_power*L_power)));
    }
    printf("DIFF %f\t%f\n",Host_result[55],result[55]);    
    for (int i=0;i<N;i++){
        printf("%d\t%f\t%f\t%d\t%d\n",i,max_value[i],Host_max[i],index[i],Host_index[i]);
    }

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_max_value);
    cudaFree(d_mean);
    cudaFree(d_sum);
    cudaFree(d_stdv);
    cudaFree(d_std_index);
    cudaFree(d_maxvalue_partial);
    cudaFree(d_index_partial);
    delete[] my_mean;
    delete[] my_sum;
    delete[] my_dot;
    delete[] my_stdv; 
    delete[] matrix;
    delete[] result;
    delete[] Host_result;
    delete[] std_index;
    fclose(fdata);
    return 1;

}
