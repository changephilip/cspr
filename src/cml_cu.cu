#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
//#include "cml_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define L 140
#define L_power 19600
typedef struct{
	int x;
	int y;
	float value;
}cml_retstruc;
__global__ void parent_ncc_kernel(float *d_data,int *d_ctr,int *d_ctr_id1,int *d_ctr_id2,float *d_sum,float *d_mean,float *d_stdv,int N,cml_retstruc *S){
     //获取局部id
     //设置cublas环境，启动cublas_sgemm
     //设置局部变量C3,接受sgemmm的结果，估计为160K,调用子内核时，不能使用local memory,必须把C3分配在global memory
     //调整方案，不使用子内核调用，直接部署代码
    int globalThreadID=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    int imagea=d_ctr_id1[globalThreadID];
    int imageb=d_ctr_id2[globalThreadID];
//    int L_power=L*L;
    int i,j;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha=1.0;
    float beta=0.0;
    float C3[L_power];
    //cudaMalloc((void**)&C3,L*L*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,&d_data[L_power*imageb],N,&d_data[L_power*imagea],N,&beta,C3,N);
    cublasDestroy(handle);

    //help矩阵排列，直接在主机端排列好？还是kernel调用？
    for (i=0;i<L;i++){
        //image_1*L+i
        for (j=0;j<L;j++){
            //image_2*L+j
//            C[i*L+j]=(C[i*L+j]+L*help[image_1*3].y*help[image_2*3].y-xy-yx)/(N*z*z);
            C3[i*L+j]=(C3[i*L+j]+L*d_mean[imagea*L+i]*d_mean[imageb*L+j]-d_sum[imagea*L+i]*d_mean[imageb*L+j]-d_mean[imagea*L+i]*d_sum[imageb*L+j])/(N*d_stdv[imagea*L+i]*d_stdv[imageb*L+j]);
        }
    }

     //分配flambda网格，让C3就地接受结果
//     flambda();
     //C3作为参数，获得最大返回值参数，获得alpha_ij编号，并获得L中的序号。
    float max_value=C3[0];
    int max_index_i=0;
    int max_index_j=0;
    for (i=0;i<L;i++){
        for (j=0;j<L;j++){
        if (C3[i*L+j]>max_value){
            max_value=C3[i*L+j];
            max_index_i=i;
            max_index_j=j;
            }
        }
    }
    S[globalThreadID].value=max_value;
    S[globalThreadID].x=max_index_i;
    S[globalThreadID].y=max_index_j;

}

extern "C" void wrapper_kernel(float *data,int N,int cml_size,float ***help,cml_retstruc *S){
    //wrapper_kernel前应该完成，数据打包成一个长数组
    //设置控制矩阵
    //读取数据接口，数值矩阵，，返回值矩阵，设置cuda环境，启动kernel
    //返回值矩阵，包括cml_matrix的value和坐标
    int control_size=N*(N-1)/2;
    int i,j;

//    int BLOCK_SIZE;//理论上没有上限
 //   int THREAD_PER_BLOCK;//(<512,根据显卡设备的cuda参数定)
    //配置控制矩阵，alpha_ij序数控制,ctr为alphaij序数
//    ctr = (int *)malloc(control_size);
    int *ctr;
    int *ctr_id1;
    int *ctr_id2;
    ctr= new int [control_size];
    ctr_id1 = new int [control_size];
    ctr_id2 = new int [control_size];
    for (i=0;i<control_size;i++){
        ctr[i]=i;
    }
    for (i=0;i<control_size;i++){
        for (j=i+1;j<control_size;j++){
//an error here ,the id is not eq i;should be modified later.
            ctr_id1[i] = i;
            ctr_id2[i] = j;
        }
    }
    //配置辅助矩阵help.拆分成三个数组，每个数组为N×L
    float *sum;
    float *mean;
    float *stdv;
    sum = new float [N*L];
    mean = new float [N*L];
    stdv = new float [N*L];
    for (i=0;i<N;i++){
        for (j=0;j<L;j++){
            sum[i*L+j]=help[i][j][0];
            mean[i*L+j]=help[i][j][1];
            stdv[i*L+j]=help[i][j][3];
        }
    }
    int *d_ctr;
    int *d_ctr_id1;
    int *d_ctr_id2;
    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
    cml_retstruc *d_S;
    cudaMalloc((void **) &d_sum,sizeof(float)*N*L);
    cudaMalloc((void **) &d_mean,sizeof(float)*N*L);
    cudaMalloc((void **) &d_stdv,sizeof(float)*N*L);
    cudaMalloc((void **) &d_data,sizeof(float)*N*L*L);
    cudaMalloc((void **) &d_ctr,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id1,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id2,sizeof(int)*control_size);

    cudaMalloc((void **) &d_S,sizeof(cml_retstruc)*control_size);

    cudaMemcpy(d_sum,sum,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,mean,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,stdv,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr,ctr,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id1,ctr_id1,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,data,N*L*L*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimGrid(control_size/500,1,1);
    dim3 dimBlock(500,1,1);
    parent_ncc_kernel<<<dimGrid,dimBlock>>>(d_data,d_ctr,d_ctr_id1,d_ctr_id2,d_sum,d_mean,d_stdv,N,d_S);
    cudaMemcpy(S,d_S,sizeof(cml_retstruc)*control_size,cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_ctr);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    cudaFree(d_S);
    delete[] sum;
    delete[] mean;
    delete[] stdv;
    delete[] ctr;
    delete[] ctr_id1;
    delete[] ctr_id2;
    //使用一个简单的kernel,不使用child kernel调用。
    //flambda需要的辅助矩阵，设置为线性格式

    //分配cuda内存，把数据矩阵、辅助矩阵存入

    //返回值矩阵，线性，分配内存

    //设置网格、线程参数，启动parent_ncc_kernel



}
