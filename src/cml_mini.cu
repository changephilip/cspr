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
    float *d_max_value;
    cudaMalloc((void **) &d_data,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_result,sizeof(float)*N*L_power);
    cudaMalloc((void **) &d_max_value,sizeof(float)*N);

    cudaMemcpy(d_data,matrix,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);

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
    for(int i=0;i<N;i++){
        cublasSgemm(handle[i],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[L_power*i],L,&d_data[L_power*i],L,&beta,&d_result[L_power*i],L);
	//mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i],handle[i]);
//        testlambda<<<L,L,0,stream[i]>>>(&d_result[L_power*i]);
        flambda<<<L,L,0,stream[i]>>>(&d_result[L_power*i],d_mean,d_sum,d_stdv,i,i);
        simplekernel<<<1,1,0,stream[i]>>>(&d_result[L_power*i],&d_max_value[i]);
    //mykernel<<<1,1,0,stream[i]>>>(&d_data[L_power*i],&d_result[L_power*i]);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result,d_result,sizeof(float)*N*L_power,cudaMemcpyDeviceToHost);
    cudaMemcpy(max_value,d_max_value,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i=0;i<N;i++){
	cublasDestroy(handle[i]);
	cudaStreamDestroy(stream[i]);
	}
    float *Host_result;
    float Host_max[N];
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
        for (int j=0;j<L_power;j++){
                if (Host_max[i]<Host_result[i*L_power+j]){
                    Host_max[i]=Host_result[i*L_power+j];
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
        printf("%d\t%f\t%f\n",i,max_value[i],Host_max[i]);
    }

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_max_value);
    cudaFree(d_mean);
    cudaFree(d_sum);
    cudaFree(d_stdv);
    delete[] my_mean;
    delete[] my_sum;
    delete[] my_dot;
    delete[] my_stdv; 
    delete[] matrix;
    delete[] result;
    delete[] Host_result;
    fclose(fdata);
    return 1;

}
