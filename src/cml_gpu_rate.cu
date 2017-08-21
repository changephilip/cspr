//#include "cml_nocv.h"
//#include "cml_cuda.h"
//#include "cml_cuda.cu"
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
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <cblas.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

#define L 128 
#define L_power 16384

using namespace std;
typedef struct{
    int x;
    int y;
}cmlncv_tuple;
struct voted
{
    int index;
    int value;
};
typedef struct{
    int x;
    int y;
    float value;
} cml_retstruc;
bool comp(const voted &a,const voted &b)
{
    return a.value<b.value;
}

float MYSUM(int Num,const float *p){
    float re=0.0f;
    int i;
    for(i=0;i<Num;i++){
        re=re+p[i];
    }
    return re;
}

float cvoting(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,float cons2){
    double a,b,c;
//    double two_pi=6.28318530;
    float angleij;
    double cons=180.0/M_PI;
    a = cos((cmlkj-cmlki)*cons2);
    b = cos((cmljk-cmlji)*cons2);
    c = cos((cmlik-cmlij)*cons2);
//    if ((1-c*c)<0 or (1-b*b)<0){
//        printf("error in cvoting\n");
//        exit(EXIT_FAILURE);
//    }
    float t;
    if ((1+2*a*b*c)>a*a+b*b+c*c) {
        t=(a-b*c)/(sqrt(1-b*b)*sqrt(1-c*c));
        if (t<1 and t>-1){
        angleij = acos(t)*cons;
        }
        else if(t>=1){
         angleij=0.0;
        }
        else if (t<=-1){
         angleij=180.0;
        }
    }
    else {angleij=-10.0;}
    return angleij;
}

cmlncv_tuple NCC_value(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
    //        value[i][j] = BNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
		value[i][j]=1.0f;
//            printf("\n000003");
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_value0(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
//            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
		value[i][j]=1.0f;
//            printf("\n000003");
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_Q(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
    float Qci[after_dft_size][4];
    float Qcj[after_dft_size][4];
//#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
//        Qci[i][0] = cblas_sasum( after_dft_size, &Ci[i*after_dft_size], 1);//sum
        Qci[i][0] = MYSUM(after_dft_size,&Ci[i*after_dft_size]);
        Qci[i][1] = Qci[i][0] / after_dft_size;//mean
        Qci[i][2] = cblas_sdot( after_dft_size, &Ci[i*after_dft_size], 1,&Ci[i*after_dft_size],1);//dot
        Qci[i][3] = sqrt((Qci[i][2] + after_dft_size*Qci[i][1]*Qci[i][1] - 2*Qci[i][0]*Qci[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
//#pragma omp parallel for
    for (j=0;j<after_dft_size;j++){
//        Qcj[j][0] = cblas_sasum( after_dft_size, &Cj[j*after_dft_size], 1);//sum
        Qcj[j][0] = MYSUM(after_dft_size,&Cj[j*after_dft_size]);
        Qcj[j][1] = Qcj[j][0] / after_dft_size;//mean
        Qcj[j][2] = cblas_sdot( after_dft_size, &Cj[j*after_dft_size],1, &Cj[j*after_dft_size],1);//dot
        Qcj[j][3] = sqrt((Qcj[j][2] + after_dft_size*Qcj[j][1]*Qcj[j][1] - 2*Qcj[j][0]*Qcj[j][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }


    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            //    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
            value[i][j] = (cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1, &Cj[j*after_dft_size],1 )+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cmlncv_tuple NCC_QT(float **Qci,float **Qcj,float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float Qci[after_dft_size][4];
//    float Qcj[after_dft_size][4];

/*
#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
        Qci[i][0] = cblas_sasum( after_dft_size, &Ci[i*after_dft_size], 1);//sum
        Qci[i][1] = Qci[i][0] / after_dft_size;//mean
        Qci[i][2] = cblas_sdot( after_dft_size, &Ci[i*after_dft_size], 1,&Ci[i*after_dft_size],1);//dot
        Qci[i][3] = sqrt((Qci[i][2] + after_dft_size*Qci[i][0]*Qci[i][0] - 2*Qci[i][0]*Qci[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
#pragma omp parallel for
    for (i=0;i<after_dft_size;i++){
        Qcj[i][0] = cblas_sasum( after_dft_size, &Cj[i*after_dft_size], 1);//sum
        Qcj[i][1] = Qcj[i][0] / after_dft_size;//mean
        Qcj[i][2] = cblas_sdot( after_dft_size, &Cj[i*after_dft_size],1, &Cj[i*after_dft_size],1);//dot
        Qcj[i][3] = sqrt((Qci[i][2] + after_dft_size*Qcj[i][0]*Qcj[i][0] - 2*Qcj[i][0]*Qcj[i][1])/after_dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
    }
*/
//    printf("see Qci\n");
//    printf("%f",Qci[0][1]);

    //mpi here
    //old code
    /*
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            //    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
            value[i][j] = (cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1, &Cj[j*after_dft_size],1 )+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
//            printf("%f",value[i][j]);
        }

    }
    */
    //new code,complete it with sgemm
    //float C1[after_dft_size*after_dft_size];
    float C[after_dft_size*after_dft_size];
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,after_dft_size,after_dft_size,after_dft_size,1,Ci,after_dft_size,Cj,after_dft_size,0,C,after_dft_size);

    for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }





    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
    //to deal with value_ini<0.7,the ncc_value shouldn't be too small
    if (value_ini<0.5){
        ret.x=-1;
        ret.y=-1;
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

float max_float(float *infloat,int size_of_array){
    int i;
    float max_return;
    max_return=infloat[0];
    for (i=1;i<size_of_array;i++){
        if (max_return < infloat[i]){
            max_return=infloat[i];
        }
    }
    return max_return;
}

int max_float_index(float *infloat,int size_of_array){
    int i;
    float max_float;
    int max_index_return;
    max_index_return=0;
    max_float=infloat[0];
    for (i=1;i<size_of_array;i++){
        if (max_float < infloat[i]){
            max_float = infloat[i];
            max_index_return=i;
        }
    }
    return max_index_return;
}
//__device__ float 
/*
__global__ void parent_ncc_kernel(float *d_data,int *d_ctr_id1,int *d_ctr_id2,float *d_sum,float *d_mean,float *d_stdv,int N,int *Sx,int *Sy,float *Svalue){
     //获取局部id
     //设置cublas环境，启动cublas_sgemm
     //设置局部变量C3,接受sgemmm的结果，估计为160K,调用子内核时，不能使用local memory,必须把C3分配在global memory
     //调整方案，不使用子内核调用，直接部署代码
    int globalThreadID=threadIdx.x+blockDim.x*blockIdx.x;
    int image_a=d_ctr_id1[globalThreadID];
    int image_b=d_ctr_id2[globalThreadID];
    //long int postion_a=L_power*image_a;
    //long int postion_b=L_power*image_b;
//    int L_power=L*L;
    int i,j;
    cublasHandle_t handle;
    cublasStatus_t status;
    status=cublasCreate(&handle);
    const float alpha=1.0;
    const float beta=0.0;
    const int si=140;
//    float C1[L_power];
//    float C2[L_power];
//    for (i=0;i<L_power;i++){
//	C1[i]=d_data[L_power*image_a+i];
//	C2[i]=d_data[L_power*image_b+i];
//	}
    float *C3=(float*)malloc(sizeof(float)*si*si);
//    float C4[si][si];
 //   if(status != CUBLAS_STATUS_SUCCESS) {
//	printf("cublasCreate fail\n");
//}
    //cudaMalloc((void**)&C3,L*L*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[L_power*image_b],L,&d_data[L_power*image_a],L,&beta,C3,L);
//    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,C2,L,C1,L,&beta,C3,L);
    cublasDestroy(handle);
    //help矩阵排列，直接在主机端排列好？还是kernel调用？
    for (i=0;i<L;i++){
        //image_1*L+i
        for (j=0;j<L;j++){
            //image_2*L+j
//            C[i*L+j]=(C[i*L+j]+L*help[image_1*3].y*help[image_2*3].y-xy-yx)/(N*z*z);
            C3[i*L+j]=(C3[i*L+j]+L*d_mean[image_a*L+i]*d_mean[image_b*L+j]-d_sum[image_a*L+i]*d_mean[image_b*L+j]-d_mean[image_a*L+i]*d_sum[image_b*L+j])/(L*d_stdv[image_a*L+i]*d_stdv[image_b*L+j]);
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
    Svalue[globalThreadID]=max_value;
    Sx[globalThreadID]=max_index_i;
    Sy[globalThreadID]=max_index_j;
    free(C3);
}
*/

/*
void wrapper_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    //wrapper_kernel前应该完成，数据打包成一个长数组
    //设置控制矩阵
    //读取数据接口，数值矩阵，，返回值矩阵，设置cuda环境，启动kernel
    //返回值矩阵，包括cml_matrix的value和坐标
    int control_size=N*(N-1)/2;
    int i,j;
    //printf("377\n");
    //printf("N\t%d\n",N);
//    int BLOCK_SIZE;//理论上没有上限
 //   int THREAD_PER_BLOCK;//(<512,根据显卡设备的cuda参数定)
    //配置控制矩阵，alpha_ij序数控制,ctr为alphaij序数
//    ctr = (int *)malloc(control_size);
   // int *ctr;
    int *ctr_id1;
    int *ctr_id2;
    //ctr= new int [control_size];
    ctr_id1 = new int [control_size];
    ctr_id2 = new int [control_size];
    //printf("389\n");
    //for (i=0;i<control_size;i++){
    //    ctr[i]=i;
    //}
    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++){
//an error here ,the id is not eq i;should be modified later.((2*local_N-1-i)*i/2+j-(i+1))
            ctr_id1[((2*N-1-i)*i/2+j-i-1)] = i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)] = j;
        }
    }
    //配置辅助矩阵help.拆分成三个数组，每个数组为N×L
    float *m_sum;
    float *m_mean;
    float *m_stdv;
    m_sum = new float [N*cml_size];
    m_mean = new float [N*cml_size];
    m_stdv = new float [N*cml_size];
    for (i=0;i<N;i++){
        for (j=0;j<cml_size;j++){
            m_sum[i*cml_size+j]=help[i][j][0];
            m_mean[i*cml_size+j]=help[i][j][1];
            m_stdv[i*cml_size+j]=help[i][j][3];
        }
    }
    //printf("412\n");
    //int *d_ctr;
    int *d_ctr_id1;
    int *d_ctr_id2;
    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    cml_retstruc *d_S;
    int *d_Sx;
    int *d_Sy;
    float *d_Svalue;
    //printf("421\n");
    cudaMalloc((void **) &d_sum,sizeof(float)*N*cml_size);
    cudaMalloc((void **) &d_mean,sizeof(float)*N*cml_size);
    cudaMalloc((void **) &d_stdv,sizeof(float)*N*cml_size);
    cudaMalloc((void **) &d_data,sizeof(float)*N*cml_size*cml_size);
    //cudaMalloc((void **) &d_ctr,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id1,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id2,sizeof(int)*control_size);

//    cudaMalloc((void **) &d_S,sizeof(cml_retstruc)*control_size);
    cudaMalloc((void **) &d_Sx,sizeof(int)*control_size);
    cudaMalloc((void **) &d_Sy,sizeof(int)*control_size);
    cudaMalloc((void **) &d_Svalue,sizeof(float)*control_size);

    cudaMemcpy(d_sum,m_sum,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,m_mean,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,m_stdv,N*L*sizeof(float),cudaMemcpyHostToDevice);
    //cudaMemcpy(d_ctr,ctr,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id1,ctr_id1,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,data,N*cml_size*cml_size*sizeof(float),cudaMemcpyHostToDevice);
//	printf("439\n");
    dim3 dimGrid(control_size/100,1,1);
    dim3 dimBlock(100,1,1);
    parent_ncc_kernel<<<dimGrid,dimBlock>>>(d_data,d_ctr_id1,d_ctr_id2,d_sum,d_mean,d_stdv,N,d_Sx,d_Sy,d_Svalue);
    cudaDeviceSynchronize(); 
//    cudaMemcpy(S,d_S,sizeof(cml_retstruc)*control_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Sx,d_Sx,sizeof(int)*control_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Sy,d_Sy,sizeof(int)*control_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*control_size,cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    //cudaFree(d_ctr);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
//    cudaFree(d_S);
    cudaFree(d_Sx);
    cudaFree(d_Sy);
    cudaFree(d_Svalue);
    delete[] m_sum;
    delete[] m_mean;
    delete[] m_stdv;
    //delete[] ctr;
    delete[] ctr_id1;
    delete[] ctr_id2;

    //使用一个简单的kernel,不使用child kernel调用。
    //flambda需要的辅助矩阵，设置为线性格式

    //分配cuda内存，把数据矩阵、辅助矩阵存入

    //返回值矩阵，线性，分配内存

    //设置网格、线程参数，启动parent_ncc_kernel



}
*/
__global__ void simple_max_kernel(float *data,float *d_max_value,int *d_max_index){
    float maxvalue=data[0];
    int i;
    int index=0;
    for (i=0;i<L_power;i++){
        if (maxvalue<data[i]){
            maxvalue=data[i];
            index=i;
        }
    }
    *d_max_value=maxvalue;
    *d_max_index=index;
}
__global__ void my_reduction_kernel1(float *d_in,float *d_out,int *d_out_index,int N){
    const int globalid=(blockDim.x*blockIdx.x+threadIdx.x);
    int i;
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
__global__ void flambda(float *d_process,float *d_sum,float *d_mean,float *d_stdv,int image_a,int image_b){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    int i=blockIdx.x;
    int j=threadIdx.x;
    d_process[id]=(d_process[id]+L*d_mean[image_a*L+i]*d_mean[image_b*L+j]-d_sum[image_a*L+i]*d_mean[image_b*L+j]-d_mean[image_a*L+i]*d_sum[image_b*L+j])/(L*d_stdv[image_a*L+i]*d_stdv[image_b*L+j]);
}
/*
__global__ void huge_kernel(int ctrid,int *d_ctr1,int *d_ctr2,float *d_data,float *d_sum,float *d_mean,float *d_stdv,float *d_buffer,float *d_p,int *d_pi,float *d_Svalue,int *d_max_index,int N){
    int gid=threadIdx.x+blockDim.x*blockIdx.x;
    int A=d_ctr1[gid+ctrid];
    int B=d_ctr2[gid+ctrid];
    const float alpha=1.0f;
    const float beta=0.0f;
    cublasHandle_t  handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[B*L_power],L,&d_data[A*L_power],L,&beta,&d_buffer[gid*L_power],L);
    cublasDestroy(handle);
    flambda<<<L,L>>>(&d_buffer[gid*L_power],d_sum,d_mean,d_stdv,A,B);
    my_reduction_kernel1<<<1,L>>>(&d_buffer[gid*L_power],&d_p[gid*L],&d_pi[gid*L],L);
    my_reduction_kernel2<<<1,1>>>(&d_p[gid*L],&d_pi[gid*L],&d_Svalue[ctrid+gid],&d_max_index[ctrid+gid],L);
}
*/
__global__ void block_kernel(int *d_ctr1,int *d_ctr2,float *d_data,float *d_sum,float *d_mean,float *d_stdv,float *d_Svalue,int *d_max_index){
    //blockid.x,.y->image_a,image_b;

    //do a pair of NCC in a block
    //size=128*128,16 per thread
    //use 1024 thread to calculate
    //16 sdot needed to used first
    //shared memory 1024 float and 1024 int
    __shared__ float sp[1024];
    __shared__ int si[1024];

    int i;
    float threadmax=0.0f;
    int threadindexmax=0;
    float dot_result[16];
    int index_a[16];
    int index_b[16];
    int index[16];
    int globalblockid=blockIdx.x+gridDim.y*blockIdx.y;
    int image_a=d_ctr1[globalblockid];
    int image_b=d_ctr2[globalblockid];
    int globalthread=threadIdx.x+threadIdx.y*32;
//    cublasHandle_t handle;
//    cublasCreate(&handle);
    //threadIdx.x,threadIdx.y
    //dot_result[0]=cublasSdot(handle,L,,1,,1);
    index_a[0]=threadIdx.x*4;
    index_b[0]=threadIdx.y*4;

    index_a[1]=index_a[0]+1;
    index_a[2]=index_a[0]+2;
    index_a[3]=index_a[0]+3;
    index_a[4]=index_a[0];
    index_a[5]=index_a[0]+1;
    index_a[6]=index_a[0]+2;
    index_a[7]=index_a[0]+3;
    index_a[8]=index_a[0];
    index_a[9]=index_a[0]+1;
    index_a[10]=index_a[0]+2;
    index_a[11]=index_a[0]+3;
    index_a[12]=index_a[0];
    index_a[13]=index_a[0]+1;
    index_a[14]=index_a[0]+2;
    index_a[15]=index_a[0]+3;

    index_b[1]=index_b[0]+1;
    index_b[2]=index_b[0]+2;
    index_b[3]=index_b[0]+3;
    index_b[4]=threadIdx.y*4;
    index_b[5]=index_b[0]+1;
    index_b[6]=index_b[0]+2;
    index_b[7]=index_b[0]+3;
    index_b[8]=threadIdx.y*4;
    index_b[9]=index_b[0]+1;
    index_b[10]=index_b[0]+2;
    index_b[11]=index_b[0]+3;
    index_b[12]=threadIdx.y*4;
    index_b[13]=index_b[0]+1;
    index_b[14]=index_b[0]+2;
    index_b[15]=index_b[0]+3;

    index[0]=index_b[0]*L+index_a[0];
    index[1]=index_b[1]*L+index_a[1];
    index[2]=index_b[2]*L+index_a[2];
    index[3]=index_b[3]*L+index_a[3];
    index[4]=index_b[4]*L+index_a[4];
    index[5]=index_b[5]*L+index_a[5];
    index[6]=index_b[6]*L+index_a[6];
    index[7]=index_b[7]*L+index_a[7];
    index[8]=index_b[8]*L+index_a[8];
    index[9]=index_b[9]*L+index_a[9];
    index[10]=index_b[10]*L+index_a[10];
    index[11]=index_b[11]*L+index_a[11];
    index[12]=index_b[12]*L+index_a[12];
    index[13]=index_b[13]*L+index_a[13];
    index[14]=index_b[14]*L+index_a[14];
    index[15]=index_b[15]*L+index_a[15];

//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[0]*L],1,&d_data[image_b*L_power+index_b[0]*L],1,&dot_result[0]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[1]*L],1,&d_data[image_b*L_power+index_b[1]*L],1,&dot_result[1]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[2]*L],1,&d_data[image_b*L_power+index_b[2]*L],1,&dot_result[2]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[3]*L],1,&d_data[image_b*L_power+index_b[3]*L],1,&dot_result[3]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[4]*L],1,&d_data[image_b*L_power+index_b[4]*L],1,&dot_result[4]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[5]*L],1,&d_data[image_b*L_power+index_b[5]*L],1,&dot_result[5]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[6]*L],1,&d_data[image_b*L_power+index_b[6]*L],1,&dot_result[6]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[7]*L],1,&d_data[image_b*L_power+index_b[7]*L],1,&dot_result[7]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[8]*L],1,&d_data[image_b*L_power+index_b[8]*L],1,&dot_result[8]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[9]*L],1,&d_data[image_b*L_power+index_b[9]*L],1,&dot_result[9]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[10]*L],1,&d_data[image_b*L_power+index_b[10]*L],1,&dot_result[10]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[11]*L],1,&d_data[image_b*L_power+index_b[11]*L],1,&dot_result[11]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[12]*L],1,&d_data[image_b*L_power+index_b[12]*L],1,&dot_result[12]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[13]*L],1,&d_data[image_b*L_power+index_b[13]*L],1,&dot_result[13]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[14]*L],1,&d_data[image_b*L_power+index_b[14]*L],1,&dot_result[14]);
//    cublasSdot(handle,L,&d_data[image_a*L_power+index_a[15]*L],1,&d_data[image_b*L_power+index_b[15]*L],1,&dot_result[15]);
    for (i=0;i<16;i++){
	dot_result[i]=0.0f;
//	for (int j=0;j<128;j++){
//		dot_result[i]+=d_data[image_a*L_power+index_a[i]*L+j]*d_data[image_b*L_power+index_b[i]*L+j];
//		}
	}
//    cublasDestroy(handle);
    //flambda
    for (i=0;i<16;i++){
    dot_result[i]=(dot_result[i]+L*d_mean[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_sum[image_a*L+index_a[i]]*d_mean[image_b*L+index_b[i]]-d_mean[image_a*L+index_a[i]]*d_sum[image_b*L+index_b[i]])/(L*d_stdv[image_a*L+index_a[i]]*d_stdv[image_b*L+index_b[i]]);
    }
//    dot_result[0]=(dot_result[0]+L*d_mean[image_a*L+index_a[0]]*d_mean[image_b*L+index_b[0]]-d_sum[image_a*L+index_a[0]]*d_mean[image_b*L+index_b[0]]-d_mean[image_a*L+index_a[0]]*d_sum[image_b*L+index_b[0]])/(L*d_stdv[image_a*L+index_a[0]]*d_stdv[image_b*L+index_b[0]]);


    //cal the max of 16
    threadmax=dot_result[0];
    threadindexmax=index[0];
    for (i=1;i<16;i++){
        threadmax=fmaxf(threadmax,dot_result[i]);
        if (threadmax==dot_result[i]){
            threadindexmax=index[i];
        }
    }
    sp[globalthread]=threadmax;
    si[globalthread]=threadindexmax;


    __syncthreads();

    for (int activethreads  = 1024 >>1;activethreads>32;activethreads >>=1){
        if (globalthread < activethreads){
            sp[globalthread] = fmaxf(sp[globalthread],sp[globalthread+activethreads]);
            if (sp[globalthread]==sp[globalthread+activethreads]){
                si[globalthread]=si[globalthread+activethreads];
            }
        }
        __syncthreads();
    }

    if (globalthread<32){
        volatile float *ws=sp;
        volatile int *wi=si;
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+32]);
        if (ws[globalthread]==ws[globalthread+32]){
            wi[globalthread]=wi[globalthread+32];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+16]);
        if (ws[globalthread]==ws[globalthread+16]){
            wi[globalthread]=wi[globalthread+16];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+8]);
        if (ws[globalthread]==ws[globalthread+8]){
            wi[globalthread]=wi[globalthread+8];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+4]);
        if (ws[globalthread]==ws[globalthread+4]){
            wi[globalthread]=wi[globalthread+4];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+2]);
        if (ws[globalthread]==ws[globalthread+2]){
            wi[globalthread]=wi[globalthread+2];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+1]);
        if (ws[globalthread]==ws[globalthread+1]){
            wi[globalthread]=wi[globalthread+1];
        }
        if (globalthread==0){
            volatile int *wi=si;
            volatile float *ws=sp;
            d_Svalue[globalblockid]=ws[0];
            d_max_index[globalblockid]=wi[0];
        }
    }




}

__global__ void block_max_kernel(float *d_data,float *d_max_value,int *d_max_index){

    __shared__ float sp[1024];
    __shared__ int si[1024];

    int i;
    float threadmax=0.0f;
    int threadindexmax=0;
    int index_a[16];
    int index_b[16];
    int index[16];
    int globalthread=threadIdx.x+threadIdx.y*32;
//    cublasHandle_t handle;
//    cublasCreate(&handle);
    //threadIdx.x,threadIdx.y
    //dot_result[0]=cublasSdot(handle,L,,1,,1);
    index_a[0]=threadIdx.x*4;
    index_b[0]=threadIdx.y*4;

    index_a[1]=index_a[0]+1;
    index_a[2]=index_a[0]+2;
    index_a[3]=index_a[0]+3;
    index_a[4]=index_a[0];
    index_a[5]=index_a[0]+1;
    index_a[6]=index_a[0]+2;
    index_a[7]=index_a[0]+3;
    index_a[8]=index_a[0];
    index_a[9]=index_a[0]+1;
    index_a[10]=index_a[0]+2;
    index_a[11]=index_a[0]+3;
    index_a[12]=index_a[0];
    index_a[13]=index_a[0]+1;
    index_a[14]=index_a[0]+2;
    index_a[15]=index_a[0]+3;

    index_b[1]=index_b[0]+1;
    index_b[2]=index_b[0]+2;
    index_b[3]=index_b[0]+3;
    index_b[4]=threadIdx.y*4;
    index_b[5]=index_b[0]+1;
    index_b[6]=index_b[0]+2;
    index_b[7]=index_b[0]+3;
    index_b[8]=threadIdx.y*4;
    index_b[9]=index_b[0]+1;
    index_b[10]=index_b[0]+2;
    index_b[11]=index_b[0]+3;
    index_b[12]=threadIdx.y*4;
    index_b[13]=index_b[0]+1;
    index_b[14]=index_b[0]+2;
    index_b[15]=index_b[0]+3;

    index[0]=index_b[0]*L+index_a[0];
    index[1]=index_b[1]*L+index_a[1];
    index[2]=index_b[2]*L+index_a[2];
    index[3]=index_b[3]*L+index_a[3];
    index[4]=index_b[4]*L+index_a[4];
    index[5]=index_b[5]*L+index_a[5];
    index[6]=index_b[6]*L+index_a[6];
    index[7]=index_b[7]*L+index_a[7];
    index[8]=index_b[8]*L+index_a[8];
    index[9]=index_b[9]*L+index_a[9];
    index[10]=index_b[10]*L+index_a[10];
    index[11]=index_b[11]*L+index_a[11];
    index[12]=index_b[12]*L+index_a[12];
    index[13]=index_b[13]*L+index_a[13];
    index[14]=index_b[14]*L+index_a[14];
    index[15]=index_b[15]*L+index_a[15];


    //cal the max of 16
    threadindexmax=d_data[index[0]];
    for (i=1;i<16;i++){
        float	tmp=d_data[index[i]];
        threadmax=fmaxf(threadmax,tmp);
        if (threadmax==tmp){
            threadindexmax=index[i];
        }
    }
    sp[globalthread]=threadmax;
    si[globalthread]=threadindexmax;


    __syncthreads();

    for (int activethreads  = 1024 >>1;activethreads>32;activethreads >>=1){
        if (globalthread < activethreads){
            sp[globalthread] = fmaxf(sp[globalthread],sp[globalthread+activethreads]);
            if (sp[globalthread]==sp[globalthread+activethreads]){
                si[globalthread]=si[globalthread+activethreads];
            }
        }
        __syncthreads();
    }

    if (globalthread<32){
        volatile float *ws=sp;
        volatile int *wi=si;
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+32]);
        if (ws[globalthread]==ws[globalthread+32]){
            wi[globalthread]=wi[globalthread+32];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+16]);
        if (ws[globalthread]==ws[globalthread+16]){
            wi[globalthread]=wi[globalthread+16];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+8]);
        if (ws[globalthread]==ws[globalthread+8]){
            wi[globalthread]=wi[globalthread+8];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+4]);
        if (ws[globalthread]==ws[globalthread+4]){
            wi[globalthread]=wi[globalthread+4];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+2]);
        if (ws[globalthread]==ws[globalthread+2]){
            wi[globalthread]=wi[globalthread+2];
        }
        ws[globalthread] = fmaxf(ws[globalthread],ws[globalthread+1]);
        if (ws[globalthread]==ws[globalthread+1]){
            wi[globalthread]=wi[globalthread+1];
        }
        if (globalthread==0){
            volatile int *wi=si;
            volatile float *ws=sp;
            *d_max_value=ws[0];
            *d_max_index=wi[0];
        }
    }




}


void stream_wrapper_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    int c_size=N*(N-1)/2;
    int a,b;
    if (N%2==0){
        a=N-1;
        b=N/2;
    }
    else {
        a=N;
        b=(N-1)/2;
    }

    float *my_sum;
    float *my_mean;
    float *my_stdv;
    int *max_index;

    max_index = new int [c_size];

    my_sum = new float [N*L];
    my_mean = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            my_sum[i*L+j]=help[i][j][0];
            my_mean[i*L+j]=help[i][j][1];
            my_stdv[i*L+j]=help[i][j][3];
        }
    }


    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    int *d_Sx;
//    int *d_Sy;

    int *d_max_index;
    float *d_Svalue;

    float *d_buffer;


    cudaMalloc((void **)&d_data,sizeof(float)*N*L_power);

    cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
    cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
    cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

//    cudaMalloc((void **)&d_Sx,sizeof(int)*c_size);
//    cudaMalloc((void **)&d_Sy,sizeof(int)*c_size);
    cudaMalloc((void **)&d_max_index,sizeof(int)*c_size);
    cudaMalloc((void **)&d_Svalue,sizeof(float)*c_size);

    cudaMemcpy(d_data,data,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);

    //d_buffer should be estimated to not over max_memory on GPU;
    cudaMalloc((void **)&d_buffer,sizeof(float)*a*L_power);

    //allocate mem for max_kernel's buffer,d_p,d_i,length=a*L;
    float *d_p;
    int *d_pi;

    cudaMalloc((void **)&d_p,sizeof(float)*L*a);
    cudaMalloc((void **)&d_pi,sizeof(int)*L*a);


//    int ctr_id1[c_size];
//    int ctr_id2[c_size];
    int *ctr_id1;
    int *ctr_id2;
    ctr_id1 = new int [c_size];
    ctr_id2 = new int [c_size];
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            ctr_id1[((2*N-1-i)*i/2+j-i-1)]=i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)]=j;
        }
    }
    //printf("598\n");
    cudaStream_t stream[a];
    cublasHandle_t handle[a];
    for (int i=0;i<a;i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
        cublasSetStream(handle[i],stream[i]);
    }
    //printf("599\n");
    const float alpha=1.0;
    const float beta=0.0;
    //每个流使用一个buffer，由于stream中的每个操作是队列排序的，因此的对应的buffer同一时刻只有一个kernel正在使用。
    for (int i=0;i<b;i++){
        for (int j=0;j<a;j++){
            int image_A=ctr_id1[a*i+j];
            int image_B=ctr_id2[a*i+j];
//            cublasSgemm(handle[j],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[ctr_id2[a*i+j]],L,&d_data[ctr_id1[a*i+j]],L,&beta,&d_buffer[j*L_power],L);
            cublasSgemm(handle[j],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[image_B*L_power],L,&d_data[image_A*L_power],L,&beta,&d_buffer[j*L_power],L);
            flambda<<<L,L,0,stream[j]>>>(&d_buffer[j*L_power],d_sum,d_mean,d_stdv,image_A,image_B);
//            simple_max_kernel<<<1,1,0,stream[j]>>>(&d_buffer[j*L_power],&d_Svalue[a*i+j],&d_max_index[a*i+j]);
            my_reduction_kernel1<<<1,L,0,stream[j]>>>(&d_buffer[j*L_power],&d_p[j*L],&d_pi[j*L],L);
            my_reduction_kernel2<<<1,1,0,stream[j]>>>(&d_p[j*L],&d_pi[j*L],&d_Svalue[a*i+j],&d_max_index[a*i+j],L);
        }
//        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(max_index,d_max_index,sizeof(int)*c_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*c_size,cudaMemcpyDeviceToHost);

    for(int i=0;i<a;i++){
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }
    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L;i++){
        for (int j=0;j<L;j++){
            ctr_L1[i*L+j]=i;
            ctr_L2[i*L+j]=j;
        }
    }
    for(int i=0;i<c_size;i++){
        Sx[i] = ctr_L1[max_index[i]];
        Sy[i] = ctr_L2[max_index[i]];
    }
    delete[] ctr_L1;
    delete[] ctr_L2;
    cudaFree(d_data);
    cudaFree(d_buffer);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_max_index);
    cudaFree(d_Svalue);
    cudaFree(d_p);
    cudaFree(d_pi);
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}
void stream_wrapper_2_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    int c_size=N*(N-1)/2;
    int a,b;
    if (N%2==0){
        a=N-1;
        b=N/2;
    }
    else {
        a=N;
        b=(N-1)/2;
    }

    float *my_sum;
    float *my_mean;
    float *my_stdv;
    int *max_index;

    max_index = new int [c_size];

    my_sum = new float [N*L];
    my_mean = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            my_sum[i*L+j]=help[i][j][0];
            my_mean[i*L+j]=help[i][j][1];
            my_stdv[i*L+j]=help[i][j][3];
        }
    }


    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    int *d_Sx;
//    int *d_Sy;

    int *d_max_index;
    float *d_Svalue;

    float *d_buffer;


    cudaMalloc((void **)&d_data,sizeof(float)*N*L_power);

    cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
    cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
    cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

//    cudaMalloc((void **)&d_Sx,sizeof(int)*c_size);
//    cudaMalloc((void **)&d_Sy,sizeof(int)*c_size);
    cudaMalloc((void **)&d_max_index,sizeof(int)*c_size);
    cudaMalloc((void **)&d_Svalue,sizeof(float)*c_size);

    cudaMemcpy(d_data,data,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);

    //d_buffer should be estimated to not over max_memory on GPU;
    cudaMalloc((void **)&d_buffer,sizeof(float)*a*L_power);

    //allocate mem for max_kernel's buffer,d_p,d_i,length=a*L;
    //float *d_p;
    //int *d_pi;

    //cudaMalloc((void **)&d_p,sizeof(float)*L*a);
    //cudaMalloc((void **)&d_pi,sizeof(int)*L*a);


//    int ctr_id1[c_size];
//    int ctr_id2[c_size];
    int *ctr_id1;
    int *ctr_id2;
    ctr_id1 = new int [c_size];
    ctr_id2 = new int [c_size];
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            ctr_id1[((2*N-1-i)*i/2+j-i-1)]=i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)]=j;
        }
    }
    //printf("598\n");
    cudaStream_t stream[a];
    cublasHandle_t handle[a];
    for (int i=0;i<a;i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
        cublasSetStream(handle[i],stream[i]);
    }
    //printf("599\n");
    const float alpha=1.0;
    const float beta=0.0;
    dim3 dimBlock(32,32,1);

    //每个流使用一个buffer，由于stream中的每个操作是队列排序的，因此的对应的buffer同一时刻只有一个kernel正在使用。
    for (int i=0;i<b;i++){
        for (int j=0;j<a;j++){
            int image_A=ctr_id1[a*i+j];
            int image_B=ctr_id2[a*i+j];
//            cublasSgemm(handle[j],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[ctr_id2[a*i+j]],L,&d_data[ctr_id1[a*i+j]],L,&beta,&d_buffer[j*L_power],L);
            cublasSgemm(handle[j],CUBLAS_OP_T,CUBLAS_OP_N,L,L,L,&alpha,&d_data[image_B*L_power],L,&d_data[image_A*L_power],L,&beta,&d_buffer[j*L_power],L);
            flambda<<<L,L,0,stream[j]>>>(&d_buffer[j*L_power],d_sum,d_mean,d_stdv,image_A,image_B);
//            simple_max_kernel<<<1,1,0,stream[j]>>>(&d_buffer[j*L_power],&d_Svalue[a*i+j],&d_max_index[a*i+j]);
            block_max_kernel<<<1,dimBlock,0,stream[j]>>>(&d_buffer[j*L_power],&d_Svalue[a*i+j],&d_max_index[a*i+j]);
//            my_reduction_kernel1<<<1,L,0,stream[j]>>>(&d_buffer[j*L_power],&d_p[j*L],&d_pi[j*L],L);
//            my_reduction_kernel2<<<1,1,0,stream[j]>>>(&d_p[j*L],&d_pi[j*L],&d_Svalue[a*i+j],&d_max_index[a*i+j],L);
        }
//        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(max_index,d_max_index,sizeof(int)*c_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*c_size,cudaMemcpyDeviceToHost);

    for(int i=0;i<a;i++){
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }
    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L;i++){
        for (int j=0;j<L;j++){
            ctr_L1[i*L+j]=i;
            ctr_L2[i*L+j]=j;
        }
    }
    for(int i=0;i<c_size;i++){
        Sx[i] = ctr_L1[max_index[i]];
        Sy[i] = ctr_L2[max_index[i]];
    }
    delete[] ctr_L1;
    delete[] ctr_L2;
    cudaFree(d_data);
    cudaFree(d_buffer);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_max_index);
    cudaFree(d_Svalue);
    //cudaFree(d_p);
    //cudaFree(d_pi);
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}

/*
void huge_wrapper_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    int c_size=N*(N-1)/2;
    int a,b;
    if (N%2==0){
        a=N-1;
        b=N/2;
    }
    else {
        a=N;
        b=(N-1)/2;
    }

    float *my_sum;
    float *my_mean;
    float *my_stdv;
    int *max_index;

    max_index = new int [c_size];

    my_sum = new float [N*L];
    my_mean = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            my_sum[i*L+j]=help[i][j][0];
            my_mean[i*L+j]=help[i][j][1];
            my_stdv[i*L+j]=help[i][j][3];
        }
    }


    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    int *d_Sx;
//    int *d_Sy;

    int *d_max_index;
    float *d_Svalue;

    float *d_buffer;


    cudaMalloc((void **)&d_data,sizeof(float)*N*L_power);

    cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
    cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
    cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

//    cudaMalloc((void **)&d_Sx,sizeof(int)*c_size);
//    cudaMalloc((void **)&d_Sy,sizeof(int)*c_size);
    cudaMalloc((void **)&d_max_index,sizeof(int)*c_size);
    cudaMalloc((void **)&d_Svalue,sizeof(float)*c_size);

    cudaMemcpy(d_data,data,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);

    //d_buffer should be estimated to not over max_memory on GPU;
    cudaMalloc((void **)&d_buffer,sizeof(float)*a*L_power);

    //allocate mem for max_kernel's buffer,d_p,d_i,length=a*L;
    float *d_p;
    int *d_pi;

    cudaMalloc((void **)&d_p,sizeof(float)*L*a);
    cudaMalloc((void **)&d_pi,sizeof(int)*L*a);


//    int ctr_id1[c_size];
//    int ctr_id2[c_size];
    int *ctr_id1;
    int *ctr_id2;
    ctr_id1 = new int [c_size];
    ctr_id2 = new int [c_size];
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            ctr_id1[((2*N-1-i)*i/2+j-i-1)]=i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)]=j;
        }
    }
    int *d_ctr_id1;
    int *d_ctr_id2;

    cudaMalloc((void **)&d_ctr_id1,sizeof(int)*c_size);
    cudaMalloc((void **)&d_ctr_id2,sizeof(int)*c_size);
    
    cudaMemcpy(d_ctr_id1,ctr_id1,sizeof(int)*c_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,sizeof(int)*c_size,cudaMemcpyHostToDevice);


    //printf("598\n");
//    cudaError_t status;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //printf("599\n");

    for (int i=0;i<b;i++){
       huge_kernel<<<1,a,0,stream>>>(a*i,d_ctr_id1,d_ctr_id2,d_data,d_sum,d_mean,d_stdv,d_buffer,d_p,d_pi,d_Svalue,d_max_index,L);
		
    }
    cudaDeviceSynchronize();
    cudaMemcpy(max_index,d_max_index,sizeof(int)*c_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*c_size,cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L;i++){
        for (int j=0;j<L;j++){
            ctr_L1[i*L+j]=i;
            ctr_L2[i*L+j]=j;
        }
    }
    for(int i=0;i<c_size;i++){
        Sx[i] = ctr_L1[max_index[i]];
        Sy[i] = ctr_L2[max_index[i]];
    }
    delete[] ctr_L1;
    delete[] ctr_L2;
    cudaFree(d_data);
    cudaFree(d_buffer);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_max_index);
    cudaFree(d_Svalue);
    cudaFree(d_p);
    cudaFree(d_pi);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}
*/
void block_wrapper_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
    int c_size=N*(N-1)/2;
    int a,b;
    if (N%2==0){
        a=N-1;
        b=N/2;
    }
    else {
        a=N;
        b=(N-1)/2;
    }

    float *my_sum;
    float *my_mean;
    float *my_stdv;
    int *max_index;

    max_index = new int [c_size];

    my_sum = new float [N*L];
    my_mean = new float [N*L];
    my_stdv = new float [N*L];

    for (int i=0;i<N;i++){
        for (int j=0;j<L;j++){
            my_sum[i*L+j]=help[i][j][0];
            my_mean[i*L+j]=help[i][j][1];
            my_stdv[i*L+j]=help[i][j][3];
        }
    }


    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
//    int *d_Sx;
//    int *d_Sy;

    int *d_max_index;
    float *d_Svalue;

    //float *d_buffer;


    cudaMalloc((void **)&d_data,sizeof(float)*N*L_power);

    cudaMalloc((void **)&d_sum,sizeof(float)*N*L);
    cudaMalloc((void **)&d_mean,sizeof(float)*N*L);
    cudaMalloc((void **)&d_stdv,sizeof(float)*N*L);

//    cudaMalloc((void **)&d_Sx,sizeof(int)*c_size);
//    cudaMalloc((void **)&d_Sy,sizeof(int)*c_size);
    cudaMalloc((void **)&d_max_index,sizeof(int)*c_size);
    cudaMalloc((void **)&d_Svalue,sizeof(float)*c_size);

    cudaMemcpy(d_data,data,sizeof(float)*N*L_power,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum,my_sum,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,my_mean,sizeof(float)*N*L,cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,my_stdv,sizeof(float)*N*L,cudaMemcpyHostToDevice);

//    int ctr_id1[c_size];
//    int ctr_id2[c_size];
    int *ctr_id1;
    int *ctr_id2;
    ctr_id1 = new int [c_size];
    ctr_id2 = new int [c_size];
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            ctr_id1[((2*N-1-i)*i/2+j-i-1)]=i;
            ctr_id2[((2*N-1-i)*i/2+j-i-1)]=j;
        }
    }
    int *d_ctr_id1;
    int *d_ctr_id2;

    cudaMalloc((void **)&d_ctr_id1,sizeof(int)*c_size);
    cudaMalloc((void **)&d_ctr_id2,sizeof(int)*c_size);

    cudaMemcpy(d_ctr_id1,ctr_id1,sizeof(int)*c_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,sizeof(int)*c_size,cudaMemcpyHostToDevice);
    dim3 dimGrid(a,b,1);
    dim3 dimBlock(32,32,1);
    unsigned int sharedsize= 1024*(sizeof(float)+sizeof(int));
    printf("line 1110\n");
    block_kernel<<<dimGrid,dimBlock,sharedsize>>>(d_ctr_id1,d_ctr_id2,d_data,d_sum,d_mean,d_stdv,d_Svalue,d_max_index);

    //for (int i=0;i<b;i++){
//       huge_kernel<<<1,a,0,stream>>>(a*i,d_ctr_id1,d_ctr_id2,d_data,d_sum,d_mean,d_stdv,d_buffer,d_p,d_pi,d_Svalue,d_max_index,L);

//    }
    cudaDeviceSynchronize();
    cudaMemcpy(max_index,d_max_index,sizeof(int)*c_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(Svalue,d_Svalue,sizeof(float)*c_size,cudaMemcpyDeviceToHost);
    //cudaStreamDestroy(stream);

    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L;i++){
        for (int j=0;j<L;j++){
            ctr_L1[i*L+j]=i;
            ctr_L2[i*L+j]=j;
        }
    }
    for(int i=0;i<c_size;i++){
        Sx[i] = ctr_L1[max_index[i]];
        Sy[i] = ctr_L2[max_index[i]];
    }
    delete[] ctr_L1;
    delete[] ctr_L2;
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_max_index);
    cudaFree(d_Svalue);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}

void List_wrapper(int *inList,FILE *f,FILE *log,FILE *particle_log,int dft_size,int dft_size_pow,FILE *debug_f,FILE *hist_f,std::vector<voted> &good_Particle){
    int time_Start,time_Readfile,time_Ncc,time_Voting,time_Sort,time_End;
    printf("in List_wrapper\n");
    time_Start = time(NULL);
    //standard cal unit
    //int local_N=5000;
    int local_N=5000;
    //allocate memory
    int *cml_Pair_Matrix[local_N];
    for (int i=0;i<local_N;i++){
        cml_Pair_Matrix[i]= new int [local_N];
    }
    long malloc_Size=local_N*dft_size_pow;
    float *data_Matrix= new float[malloc_Size];

    for (int i=0;i<local_N;i++){
        fseek(f,inList[i]*dft_size_pow*sizeof(float),SEEK_SET);
        fread(&data_Matrix[i*dft_size_pow],sizeof(float),dft_size_pow,f);
    }

    time_Readfile = time(NULL);

    //pre NCC calculate
    float  **pre_Ncc[local_N];
    for (int i=0;i<local_N;i++){
        pre_Ncc[i]= new float *[dft_size];
    }
    for (int i=0;i<local_N;i++){
        for (int j=0;j<dft_size;j++){
            pre_Ncc[i][j] = new float [4];
        }
    }
    for (int i=0;i<local_N;i++){
        for (int j=0;j<dft_size;j++){
            long postion=i*dft_size_pow+j*dft_size;
            pre_Ncc[i][j][0] = MYSUM(dft_size,&data_Matrix[postion]);//sum
            pre_Ncc[i][j][1] = pre_Ncc[i][j][0] / dft_size;//mean
            pre_Ncc[i][j][2] = cblas_sdot( dft_size, &data_Matrix[postion], 1,&data_Matrix[postion],1);//dot
            pre_Ncc[i][j][3] = sqrt((pre_Ncc[i][j][2] + dft_size*pre_Ncc[i][j][1]*pre_Ncc[i][j][1] - 2*pre_Ncc[i][j][0]*pre_Ncc[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
        }
    }
    //give work to gpu
    int *Sx;
    int *Sy;
    float *Svalue;
    Sx= new int [local_N*(local_N-1)/2];
    Sy= new int [local_N*(local_N-1)/2];
    Svalue= new float [local_N*(local_N-1)/2];
    stream_wrapper_2_kernel(data_Matrix,local_N,dft_size,pre_Ncc,Sx,Sy,Svalue);
    time_Ncc = time(NULL);
    //get svalue,and apply a threshold
    float *sysSvalue[local_N];
    for (int i=0;i<local_N;i++){
    	sysSvalue[i] = new float [local_N];
	}
    int ncc_filter=1;
    float ncc_threshold=0.5f;
    if (ncc_filter){
	float *svalue_bak;
	int r=4;
	long svalue_size=local_N*(local_N-1)/2;
	svalue_bak = new float[local_N*(local_N-1)/2];
    	int threshold_Top=floor((1-(r/sqrt(local_N)))*svalue_size);
    	for (int i=0;i<svalue_size;i++){
		svalue_bak[i] = Svalue[i];
		}
	sort(svalue_bak,svalue_bak+svalue_size);
	ncc_threshold = svalue_bak[threshold_Top-1];
	delete[] svalue_bak;
	}
	
    //convert index
    for (int i=0;i<local_N;i++){
        for (int j=i+1;j<local_N;j++){
	    sysSvalue[i][j]=Svalue[(2*local_N-1-i)*i/2+j-(i+1)];
	    sysSvalue[j][i]=Svalue[(2*local_N-1-i)*i/2+j-(i+1)];
//            if (Svalue[(2*local_N-1-i)*i/2+j-(i+1)]>0.5){
	    if (Svalue[(2*local_N-1-i)*i/2+j-(i+1)]>ncc_threshold){
//                        cml_pair_matrix_help[i][j]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].x;
//                        cml_pair_matrix_help[j][i]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].y;
            cml_Pair_Matrix[i][j]=Sx[(2*local_N-1-i)*i/2+j-(i+1)];
            cml_Pair_Matrix[j][i]=Sy[(2*local_N-1-i)*i/2+j-(i+1)];
            }

        else{
            cml_Pair_Matrix[i][j]=-1;
            cml_Pair_Matrix[j][i]=-1;
            }
        }
    }
//debug print s_value
    for (int i=0;i<local_N;i++){
//	fprintf(debug_f,"%d\t",inList[i]);
	for (int j=0;j<local_N;j++){
//		fprintf(debug_f,"%f\t",sysSvalue[i][j]);
		}
//	fprintf(debug_f,"\n");
	}
    for (int i=0;i<local_N;i++){
	delete[] sysSvalue[i];
	}
    //clear memory won't be used
    delete[] Sx;
    delete[] Sy;
    delete[] Svalue;
    for (int i=0;i<local_N;i++){
        for (int j=0;j<dft_size;j++){
            delete[] pre_Ncc[i][j];
        }
    }
    for (int i=0;i<local_N;i++){
        delete[] pre_Ncc[i];
    }
    //do voting
    int T=72;
    float sigma=180.0/T;
    float *hist_Peak =  new float[local_N*(local_N-1)/2];
    int *hist_Index = new int[local_N*(local_N-1)/2];
    float half_pow_pi=sqrt(2*M_PI)*sigma;
//    float four_sigma_pow=4*sigma*sigma;
    float two_sigma_pow=2*sigma*sigma;
    float cons=2*M_PI/dft_size;
    float Trecurse=180.0/T;
    //voting core
    for (int i=0;i<local_N;i++){
        for (int j=i+1;j<local_N;j++){
            long int alpha_ij=((2*local_N-1-i)*i/2+j-(i+1));
            if (cml_Pair_Matrix[i][j]!=-1){
                float tmp_Voting[local_N];
                float tmp_Hist[72]={0.0};
#pragma omp parallel for
                for (int k=0;k<local_N;k++){
                    if (k!=i and k!=j){
                        if (cml_Pair_Matrix[i][k]!=-1 and cml_Pair_Matrix[j][k]!=-1){
                            tmp_Voting[k]=cvoting(cml_Pair_Matrix[i][j],cml_Pair_Matrix[i][k],cml_Pair_Matrix[j][i],cml_Pair_Matrix[j][k],cml_Pair_Matrix[k][i],cml_Pair_Matrix[k][j],cons);
                        }
                        else {tmp_Voting[k]=-10.0;}
                    }
                    else {tmp_Voting[k]=-10.0;}
                }
                for (int m=0;m<local_N;m++){
                    float tmp=tmp_Voting[m];
                    if (tmp!=-10.0){
#pragma omp parallel for
                        for (int l=0;l<T;l++){
                            float alpha_t_alpha12=Trecurse*l-tmp;
                            tmp_Hist[l]=tmp_Hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/two_sigma_pow)/half_pow_pi;
                        }
                    }
                }
                hist_Peak[alpha_ij]=max_float(tmp_Hist,T);
            }
            else{
            hist_Peak[alpha_ij]=0.0;
            }
        }
    }
    time_Voting = time(NULL);
    //get voted_value
    int r=4;

    float *hist_Peak_Bak = new float[local_N*(local_N-1)/2];
    long hist_Peak_Size = local_N*(local_N-1)/2;
    int threshold_Top=floor((1-(r/sqrt(local_N)))*hist_Peak_Size);
    for (int i=0;i<hist_Peak_Size;i++){
        hist_Peak_Bak[i] = hist_Peak[i];
    }
    sort(hist_Peak_Bak,hist_Peak_Bak+hist_Peak_Size);
    fprintf(log,"threshold_top %d\n",threshold_Top);
    float threshold = hist_Peak_Bak[threshold_Top-1];

    delete[] hist_Peak_Bak;


    int Result_Voted[local_N];
    for (int i=0;i<local_N;i++){
        Result_Voted[i]=0;
    }
    //print out hist data
    fprintf(hist_f,"*Vertices %d\n",local_N);
    for (int i=0;i<local_N;i++){
//	 fprintf(hist_f," %d \"%d\"\n",i+1,i+1);
	}
    fprintf(hist_f,"*Arcs\n");
    for (int i=0;i<local_N;i++){
	for (int j=i+1;j<local_N;j++){
		int index = (2*local_N-1-i)*i/2+j-(i+1);
		if (hist_Peak[index]>threshold){
//			fprintf(hist_f," %d %d %f\n",j+1,i+1,1.0);
			}
		}
   	 }
    //find high cml_pair
    for (int i=0;i<local_N;i++){
        for (int j=i+1;j<local_N;j++){
            int index = (2*local_N-1-i)*i/2+j-(i+1);
            if (hist_Peak[index]>threshold){
                Result_Voted[i]=Result_Voted[i]+1;
                Result_Voted[j]=Result_Voted[j]+1;
            }
        }
    }
    //sort voted_value
    std::vector<voted> V;
    for (int i=0;i<local_N;i++){
        voted s;
        s.index=i;
        s.value=Result_Voted[i];
        V.push_back(s);
    }
    std::sort(V.begin(),V.end(),comp);

    time_Sort = time(NULL);
    //if noise is known
    if (0==1){
        int num_Noise = 1;//as noise_flag
        int counter=0;
        for (int i=0;i<local_N;i++){
            if (V[i].index>= num_Noise){
                counter=counter+1;
            }
            if ((i+1)%100==0){
                fprintf(log,"%d\tcounter %d\trate\t%f\n",i,counter,float(counter)/(i+1));
            }
        }
    }
    //write result
    //write result in global index
    for (int i=0;i<local_N;i++){
        fprintf(particle_log,"%d\t%d\t%d\n",i,inList[V[i].index],V[i].value);
    }
    //write good particle in global_good_particle
    for (int i=0;i<local_N;i++){
        voted tmp;
        tmp.index=inList[V[i].index];
        tmp.value=V[i].value;
        good_Particle.push_back(tmp);
    }
    //free memory
    delete[] data_Matrix;
    delete[] hist_Index;
    delete[] hist_Peak;
    for (int i=0;i<local_N;i++){
        delete[] cml_Pair_Matrix[i];
    }
    time_End = time(NULL);

    //write log
    fprintf(log,"Ncc_Time %d\n",time_Ncc-time_Readfile);
    fprintf(log,"Voting_Time %d\n",time_Voting-time_Ncc);
    fprintf(log,"Sort_Time %d\n",time_Sort-time_Voting);


}

int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N_particles=0;
    int iteration=0;



    char* filename;
    char* good_particle;
    char* logfile;
    char* debugfilename;
    char* histfilename;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:f:i:l:o:d:n:h")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
        case 'f':
            filename=optarg;
            break;
        case 'i':
            iteration=atoi(optarg);
            break;
        case 'l':
            good_particle=optarg;
            break;
        case 'o':
            logfile=optarg;
            break;
        case 'd':
	    debugfilename=optarg;
	    break;
        case 'n':
	    histfilename=optarg;
	    break;
        case 'h':
            printf("CML_NONPART is a program that picks high quality particles and throw out non-particles\n");
            printf("Author:\tQian JiaQiang ,Fudan Univesity,15210700078@fudan.edu.cn\tHuangQiang Lab\n");
            printf("-s the particle size after dft and linear polar,default is 128(only 128 supported\n");
            printf("-n the number of particles\n");
            printf("-f the mrcs file which contains the particles data\n");
            printf("-l your output filename,which will contain the particles,-o your output log file\n");
            printf("example\n");
            printf("cml_noise -s 128 -f ~/data/data.mrc -i 5 -l ~/output/particles -o ~/output/log\n");
            exit(EXIT_SUCCESS);
            break;
        }
    }



    if (!filename) {
        printf("-f filename's abstract path is needed\n");
        exit(EXIT_FAILURE);
    }
    if (!good_particle){
        printf("-l output file name 's abstract path is needed\n");
    }
//    if (N==-1 or cml_size==0){
//        printf("-n N and -s dft_cml_size are both needed,if N=0,then N=max\n");
//        exit(EXIT_FAILURE);
//    }
    if (!logfile){
        printf("-o logfile is needed.\n");
        exit(EXIT_FAILURE);
    }


    FILE *OUTFILE;
    OUTFILE=fopen(logfile,"a+");
    fprintf(OUTFILE,"cml_size\t%d\n",cml_size);
    //fprintf(OUTFILE,"N\t%d\n",N);

//全局参数设定
    int dft_size=cml_size;
    int dft_size_pow=dft_size*dft_size;
    //int T;
//    T=72;
//    int i,j,k;
//    float sigma=180.0/float(T);
    FILE *f;
    FILE *outputfile;
    FILE *debugfile;
    FILE *histfile;
//    FILE *fnoise;
    char mybuf[1024];
    char mybuf2[1024];
    char mybuf3[1024];
    char mybuf4[1024];

    f=fopen(filename,"rb");


    outputfile=fopen(good_particle,"a+");
    debugfile=fopen(debugfilename,"a+");
    histfile=fopen(histfilename,"a+");
    setvbuf(outputfile,mybuf,_IOLBF,1024);
    setvbuf(OUTFILE,mybuf2,_IOLBF,1024);
    setvbuf(debugfile,mybuf3,_IOLBF,1024);
    setvbuf(histfile,mybuf4,_IOLBF,1024);
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);

    rewind(f);
    fprintf(OUTFILE,"length\t%ld\n",filelength);

    N_particles=filelength/dft_size_pow/sizeof(float);

    int t_start,t_all_end;

    t_start=time(NULL);




//iteration 决定了迭代次数
//数据集的对齐部分直接每轮随机迭代。剩余部分随机补齐计算。
//数据集在进入计算前先随机一次。所有轮次的对齐部分和随机补齐部分是固定不变的。
    std::default_random_engine dre((unsigned)time(NULL));
    int List_Particle[N_particles];
    /*
    int align_p;
    int remain_p;
	*/
    for (int i=0;i<N_particles;i++){
        List_Particle[i]=i;
    }
	/*
    if (N_particles%5000==0){
        remain_p=0;
        align_p=N_particles;
    }
    else {
            remain_p=N_particles%5000;
            align_p=N_particles-remain_p;
            }
    */
    //全列表随机
    std::shuffle(List_Particle,List_Particle+N_particles,dre);
  /*  
    for (int control_iteration=iteration;control_iteration>0;control_iteration=control_iteration-1){
        //随机列表靠前5000*n部分
        std::shuffle(List_Particle,List_Particle+align_p,dre);
        //先做align_p
        //准备工作
        int local_N=5000;
	//do test to use debug
	///////////
	//align_p=5000;
	//remain_p=0;
	///////////
        for (int child=0;child<align_p/5000;child=child+1){
            List_wrapper(&List_Particle[child*5000],f,OUTFILE,outputfile,dft_size,dft_size_pow,debugfile,histfile);
            fprintf(OUTFILE,"%d/%d completed\n",child,align_p/5000);
            }
        if (remain_p!=0){
         //do something
            int extra_Particles = local_N - remain_p;
         //construct a new list
            int *extra_List;
            extra_List = new int [local_N];
            for (int i=0;i<extra_Particles;i++){
                extra_List[i]=List_Particle[i];
            }
	    for (int i=0;i<remain_p;i++){
		extra_List[extra_Particles+i]=List_Particle[align_p+i];
	    }
            List_wrapper(extra_List,f,OUTFILE,outputfile,dft_size,dft_size_pow,debugfile,histfile);
	    delete[] extra_List;

        }


}*/

    ///////////////
    ///we use a complex and recursive method to achieve high-quality particles
    ///////////////
    //全部粒子vector
    std::vector<int> Global_Particle(List_Particle,List_Particle+sizeof(List_Particle)/sizeof(*List_Particle));
    printf("%ld\n",Global_Particle.size());
    //筛选出的vector
    std::vector<int> Global_Good_Particle;
    std::vector<int> Global_Bad_Particle;
    for (int control_iteration=iteration;control_iteration>0;control_iteration=control_iteration-1){
        //int local_N=5000;
	int local_N=5000;
        //构建局部粒子列表
        int List_Current[N_particles-Global_Good_Particle.size()];
	printf("List_Current %ld\n",N_particles-Global_Good_Particle.size());
        std::vector<int> Particle_Current;
        for (int x : Global_Particle){
	    /*
            std::vector<int>::iterator iter=std::find(Global_Good_Particle.begin(),Global_Good_Particle.end(),x);
            if (iter==Global_Good_Particle.end()){
                Particle_Current.push_back(x);
            }
	    */
	    std::vector<int>::iterator iter=std::find(Global_Bad_Particle.begin(),Global_Bad_Particle.end(),x);
	    if (iter==Global_Bad_Particle.end()){
		Particle_Current.push_back(x);
	    }
        }
        for (int i=0;i<Particle_Current.size();i++){
            List_Current[i]=Particle_Current[i];
        }
        int Current_Align_P,Current_Remain_P;
        if (Particle_Current.size()%local_N==0){
            Current_Remain_P=0;
            Current_Align_P=Particle_Current.size();
        }
        else {
                Current_Remain_P=Particle_Current.size()%local_N;
                Current_Align_P=Particle_Current.size()-Current_Remain_P;
                }
	printf("before List_wrapper\n");
	printf("Particle_Current.size%ld\n",Particle_Current.size());
	printf("align %d\t%d\n",Current_Remain_P,Current_Align_P);
        //随机align部分粒子
        std::shuffle(List_Current,List_Current+Current_Align_P,dre);
        //start
        std::vector<voted> this_turn;
        for (int child=0;child<Current_Align_P/local_N;child=child+1){
            List_wrapper(&List_Current[child*local_N],f,OUTFILE,outputfile,dft_size,dft_size_pow,debugfile,histfile,this_turn);
            fprintf(OUTFILE,"%d/%d completed\n",child,Current_Align_P/local_N);
            }
        if (Current_Remain_P!=0){
         //do something
            int extra_Particles = local_N - Current_Remain_P;
            //construct a new list
            int *extra_List;
            extra_List = new int [local_N];
            for (int i=0;i<extra_Particles;i++){
                extra_List[i]=List_Current[i];
            }
            for (int i=0;i<Current_Remain_P;i++){
                extra_List[extra_Particles+i]=List_Current[Current_Align_P+i];
            }
            List_wrapper(extra_List,f,OUTFILE,outputfile,dft_size,dft_size_pow,debugfile,histfile,this_turn);
            delete[] extra_List;
        }
        //see this_turn

        std::map<int,float> score_countainer;
        std::map<int,int> score_counter;
        //initial map
        for (auto x : Particle_Current){
            score_countainer[x]=0.0f;
            score_counter[x]=0;
        }
        for (auto x : this_turn){
            score_countainer[x.index]+=x.value;
            score_counter[x.index]+=1;
        }
        for (auto x : Particle_Current){
            score_countainer[x]=score_countainer[x]/score_counter[x];
        }

        std::vector<voted> V;
        for (auto x : Particle_Current){
            voted s;
            s.index=x;
            s.value=score_countainer[x];
            V.push_back(s);
        }

        std::sort(V.begin(),V.end(),comp);
	/*
        float rate=Particle_Current.size()-0.05*Global_Particle.size();
        for (int i=int(rate);i<Particle_Current.size();i++){
                Global_Good_Particle.push_back(V[i].index);
        }
	*/
	//float rate=0.2*Global_Particle.size();
	int rate=0;//rate=0.0,so it works as directive iteration;
	for (int i=0;i<Particle_Current.size();i++){
		if (i<rate){
		Global_Bad_Particle.push_back(V[i].index);
		}
	}
        fprintf(OUTFILE,"iteration\t%d\tcompleted\n",control_iteration);
        //need some other code

    }

    fprintf(OUTFILE,"Get Particles\n");
    /*
    for (auto x: Global_Good_Particle){
        fprintf(OUTFILE,"%d\n",x);
    }
    */
    for (auto x: Global_Particle ){
	std::vector<int>::iterator iter=std::find(Global_Bad_Particle.begin(),Global_Bad_Particle.end(),x);
	    if (iter==Global_Bad_Particle.end()){
		fprintf(OUTFILE,"%d\n",x);	    
	}
	}
    ///////////////
        fclose(f);
        fclose(outputfile);
//        fclose(fnoise);
        fclose(OUTFILE);
        fclose(debugfile);
        fclose(histfile);






        t_all_end=time(NULL);
        printf("all time %dhour\n",(t_all_end-t_start)/3600);

        return 0;
}

