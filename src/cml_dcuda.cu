
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
void stream_kernel(float *data,int N,int cml_size,float ***help,int *Sx,int *Sy,float *Svalue){
}
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
    printf("389\n");
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

__global__ void flambda(float *d_process,float *d_sum,float *d_mean,float *d_stdv,int image_a,int image_b){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    int i=blockIdx.x;
    int j=threadIdx.x;
    d_process[id]=(d_process[id]+L*d_mean[image_a*L+i]*d_mean[image_b*L+j]-d_sum[image_a*L+i]*d_mean[image_b*L+j]-d_mean[image_a*L+i]*d_sum[image_b*L+j])/(L*d_stdv[image_a*L+i]*d_stdv[image_b*L+j]);
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
    printf("598\n");
    cudaStream_t stream[a];
    cublasHandle_t handle[a];
    for (int i=0;i<a;i++){
        cudaStreamCreate(&stream[i]);
        cublasCreate(&handle[i]);
        cublasSetStream(handle[i],stream[i]);
    }
    printf("599\n");
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
            simple_max_kernel<<<1,1,0,stream[j]>>>(&d_buffer[j*L_power],&d_Svalue[a*i+j],&d_max_index[a*i+j]);
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
    int L_size=L_power;
    int *ctr_L1;
    int *ctr_L2;
    ctr_L1 = new int [L_power];
    ctr_L2 = new int [L_power];
    for (int i=0;i<L_power;i++){
        for (int j=i+1;j<L_power;j++){
            ctr_L1[((2*L-1-i)*i/2+j-i-1)]=i;
            ctr_L2[((2*L-1-i)*i/2+j-i-1)]=j;
        }
    }
    for(int i=0;i<L_power;i++){
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
    delete[] my_sum;
    delete[] my_mean;
    delete[] my_stdv;
    delete[] ctr_id1;
    delete[] ctr_id2;
    delete[] max_index;

}

int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N=-1;
    int N_noise=-1;
    int START=1;
    int version=-1;
    int hist_flag=0;
    int iteration=0;
    int debug_flag=0;
    int iteration_SIZE=0;
//    int noise_flag=0;
    char* filename;
    char* good_particle;
    char* logfile;
    char* noisefile;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:p:v:k:i:d:l:o:hN:")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
        case 'n':
            N=atoi(optarg);
            break;
        case 'N':
            noisefile=optarg;
            break;
        case 'f':
            filename=optarg;
            break;
        case 'p':
            START=atoi(optarg);
            break;
        case 'v':
            version=atoi(optarg);
            break;
        case 'k':
            hist_flag=atoi(optarg);
            break;
        case 'i':
            iteration=1;
            iteration_SIZE=atoi(optarg);
            break;
        case 'd':
            debug_flag=1;
            break;
        case 'l':
            good_particle=optarg;
            break;
        case 'o':
            logfile=optarg;
            break;
        case 'h':
            printf("CML_NONPART is a program that picks high quality particles and throw out non-particles\n");
            printf("Author:\tQian jiaqiang ,Fudan Univesity,15210700078@fudan.edu.cn\tHuangQiang Lab\n");
            printf("-s the particle size after dft and linear polar\n");
            printf("-n the number of particles\n");
            printf("-N the file contains noise images\n");
            printf("-f the mrcs file which contains the particles data\n");
            printf("-p the start position of particles in mrcs file,default 1\n");
            //printf("-v the calculate method we used,default -1,needn't changed");
            printf("-i if you use -i,then we will try to throw out non-particles\n");
            printf("-l your output filename,which will contain the particles\n");
            printf("example\n");
            printf("cml_noise -s 144 -n 0 -N ~/data/noise.mrc -f ~/data/data.mrc -i -l ~/output/particles -o ~/output/log\n");
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
    if (N==-1 or cml_size==0){
        printf("-n N and -s dft_cml_size are both needed,if N=0,then N=max\n");
        exit(EXIT_FAILURE);
    }
    if (!logfile){
        printf("-o logfile is needed.\n");
        exit(EXIT_FAILURE);
    }
    if (!noisefile){
        printf("-N noise file is needed.\n");
        exit(EXIT_FAILURE);
    }
    FILE *OUTFILE;
    OUTFILE=fopen(logfile,"a+");
    fprintf(OUTFILE,"cml_size\t%d\n",cml_size);
    fprintf(OUTFILE,"N\t%d\n",N);

//全局参数设定
    int dft_size=cml_size;
    int dft_size_pow=dft_size*dft_size;
    int T;
    T=72;
    int i,j,k;
    float sigma=180.0/float(T);
    FILE *f;
    FILE *outputfile;
    FILE *fnoise;
    char mybuf[1024];
    char mybuf2[1024];

//    f=fopen("/home/qjq/data/qjq_200_data","rb");
    f=fopen(filename,"rb");
    fnoise=fopen(noisefile,"rb");

    outputfile=fopen(good_particle,"a+");
    setvbuf(outputfile,mybuf,_IOLBF,1024);
    setvbuf(OUTFILE,mybuf2,_IOLBF,1024);
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);
    if (START-1<0 or (START-1+N)*dft_size_pow>filelength){
        printf("-p START can't L.E 0 or GREATER THAN filelength");
        exit(EXIT_FAILURE);
    }
    rewind(f);
    fprintf(OUTFILE,"length\t%ld\n",filelength);
    if (N==0){
        N=filelength/dft_size_pow/sizeof(float);
    }
    if (N>filelength/dft_size_pow/sizeof(float)){
        fprintf(OUTFILE,"N can't be larger than the max numofItem\n");
        exit(EXIT_FAILURE);
    }

    long noisefilelength;
    fseek(fnoise,0,SEEK_END);
    noisefilelength=ftell(fnoise);
    rewind(fnoise);
    N_noise=noisefilelength/dft_size_pow/sizeof(float);
    int List_Noise[N_noise];
    for (i=0;i<N_noise;i++){
        List_Noise[i]=i;
    }

    int t_start,t_read_file,t_ncc_value,t_end,t_all_end,t_vote_1,t_vote_2,t_vote_3;
    int t_ncc_gpu;
    struct timeval tsBegin,tsEnd;
    t_start=time(NULL);


//    std::vector<int> Global_good_particle;
//    int alpha_ij;
    int iteration_size;
    if (iteration==1){
        //可以选择读取所有粒子数据到硬盘，也可以选择每次单独读取，先选择每次单独读取，节约内存资源
        iteration_size=iteration_SIZE;
        int n_iteration;

        int last_iteration=N%iteration_size;
        if (last_iteration==0){
            n_iteration=N/iteration_size;
        }
        else {
            n_iteration=N/iteration_size+1;
        }
        fprintf(OUTFILE,"n_iteration \t%d\n",n_iteration);
        int control_struct[n_iteration][2];
        //control_struct[local_start][this iteration's particles num]切分数据集，每个数据集为iteration_size
        for (int t=0;t<n_iteration;t++){
            control_struct[t][0]=iteration_size*t;//记录这个划分的第一个粒子的位置，编号-1;
            control_struct[t][1]=iteration_size;//这个子集的粒子数量;
        }
        if (last_iteration!=0){
        control_struct[n_iteration-1][1]=last_iteration;
        }


        int control=0;
        for  (control=0;control<n_iteration;control++){//在每次control中，完成iteration_size的计算
            //初始化cml矩阵
            int local_N=control_struct[control][1];//这次control中的粒子数量
            int local_start=control_struct[control][0]*dft_size_pow;//文件中，这次control的粒子的起始位置
            int double_local_N=2*local_N;
//            int local_size_index=local_size*(local_size-1)/2;
            int *cml_pair_matrix[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i]= new int[double_local_N];
                }
            float *cml_pair_matrix_help[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix_help[i] = new float [double_local_N];
                }
                //初始化数据集
            long malloc_size=double_local_N*dft_size_pow;
            float *lineardft_matrix=new float[malloc_size];
            fseek(f,local_start*sizeof(float),SEEK_SET);
            fread(lineardft_matrix,sizeof(float),local_N*dft_size_pow,f);
            //读入Noise文件，将Noise文件乱序，取和local_N等量的噪音图像
            std::random_shuffle(List_Noise,List_Noise+N_noise);
            for (i=0;i<local_N;i++){
                fseek(fnoise,List_Noise[i]*dft_size_pow*sizeof(float),SEEK_SET);
                fread(&lineardft_matrix[(local_N+i)*dft_size_pow],sizeof(float),dft_size_pow,fnoise);
            }
            fprintf(OUTFILE,"read finished\n");

            //维护混合数据List，记录粒子信息,(？不需要维护局部表）
//            int maintain_list[double_local_N];
            //局部粒子矩阵，不需要做粒子数据偏移，但对粒子的序号在输出时要做偏移
            //计算cml_pair_matrix 旧方法,a cblas version
            if (version == 0){
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
                            tmp=NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            //tmp=CMLNCV::NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }
            }
//	    printf("702\n");
            //using a faster version
            if (version == 1){
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
                            //tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }
            }
//		printf("703\n");
            t_start=time(NULL);
            //the fastest cpu version
            if (version == -1){
            //cml_pair_matrix 新方法
                float **total_nccq[double_local_N];
            //    float ;
            //    total_nccq = new float** [N];
                int postion;
                for (i=0;i<double_local_N;i++){
                    total_nccq[i] = new float* [dft_size];
                }
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        total_nccq[i][j] = new float[4];
                    }
                }
            //    printf("000005\n");
//                gettimeofday(&tsBegin,NULL);
                for (i=0;i<double_local_N;i++){
            //        #pragma omp parallel for,can't openmp here
                    for (j=0;j<dft_size;j++){
                        postion=i*dft_size_pow+j*dft_size;
            //            printf("000006\n");
//                        total_nccq[i][j][0] = cblas_sasum( dft_size, &lineardft_matrix[postion], 1);//sum
                        total_nccq[i][j][0] = MYSUM(dft_size,&lineardft_matrix[postion]);//sum
            //            printf("000007\n");
                        total_nccq[i][j][1] = total_nccq[i][j][0] / dft_size;//mean
            //            printf("000008\n");
                        total_nccq[i][j][2] = cblas_sdot( dft_size, &lineardft_matrix[postion], 1,&lineardft_matrix[postion],1);//dot
            //            printf("000009\n");
                        total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + dft_size*total_nccq[i][j][1]*total_nccq[i][j][1] - 2*total_nccq[i][j][0]*total_nccq[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
                    }
                }


//calculate ncc with cpu
                for (i=0;i<double_local_N;i++){
                #pragma omp parallel for
                    for (j=i+1;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
            //                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            cml_pair_matrix[i][j]=tmp.x;
                            cml_pair_matrix[j][i]=tmp.y;
                        }
                    }
                }
//calculate ncc with gpu
//        cml_retstruc *S;
        //
        int *Sx;
        int *Sy;
        float *Svalue;
        Sx= new int [double_local_N*(double_local_N-1)/2];
        Sy= new int [double_local_N*(double_local_N-1)/2];
        Svalue= new float [double_local_N*(double_local_N-1)/2];
//                S = new cml_retstruc[double_local_N*(double_local_N-1)/2];
		printf("before enter wrappper\n");
		t_ncc_gpu=time(NULL);
       //         wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
		stream_wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,Sx,Sy,Svalue);
                for (i=0;i<double_local_N;i++){
                    for (j=i+1;j<double_local_N;j++){
                        if (Svalue[(2*double_local_N-1-i)*i/2+j-(i+1)]>0.5){
//                        cml_pair_matrix_help[i][j]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].x;
//                        cml_pair_matrix_help[j][i]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].y;
                        cml_pair_matrix_help[i][j]=Sx[(2*double_local_N-1-i)*i/2+j-(i+1)];
                        cml_pair_matrix_help[j][i]=Sy[(2*double_local_N-1-i)*i/2+j-(i+1)];
                        }

                    else{
                        cml_pair_matrix_help[i][j]=-1;
                        cml_pair_matrix_help[j][i]=-1;
                    }}
                }
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i][i]=-1;
                    cml_pair_matrix_help[i][i]=-1;
                }
                //test GPU with cpu
                float diff=0.0f;
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<double_local_N;j++){
                        diff+=(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j])*(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j]);
                        printf("%d\t%d\t%f\n",i,j,(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j])*(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j]));
                    }
                }
		//for (i=0;i<double_local_N;i++){
		//	for (j=0;j<double_local_N;j++){
		//	printf("%d\t",cml_pair_matrix_help[i][j]);}
		//	printf("\n");
//}
		for (i=0;i<double_local_N*(double_local_N-1)/2;i++){
			printf("%d\t%d\t%d\t%f\n",i,Sx[i],Sy[i],Svalue[i]);
		}
		printf("\n");
                diff=sqrt(diff/double_local_N/double_local_N);
                printf("diff between gpu_ncc with cpu_ncc\t%f\n",diff);
		delete[] Sx;
		delete[] Sy;
		delete[] Svalue;
                //test
                /*
                for (i=0;i<double_local_N;i++){
//                #pragma omp parallel for
                    for (j=0;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
            //                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=CMLNCV::NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            if (cml_pair_matrix[i][j]!=tmp.x) printf("error\n");
                            if (cml_pair_matrix[j][i]!=tmp.y) printf("error\n");
                        }
                    }
                }
                */
                //NCC计算完成，所有common line被算出，释放计算辅助的数据存储矩阵
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        delete[] total_nccq[i][j];
                        }
                    }
                for (i=0;i<double_local_N;i++){
                        delete[] total_nccq[i];
                    }

            }
            fprintf(OUTFILE,"ncc cal finished\n");
            t_ncc_value=time(NULL);

            float *hist_peak =  new float[double_local_N*(double_local_N-1)/2];
            int *hist_index = new int[double_local_N*(double_local_N-1)/2];
            float half_pow_pi=sqrt(M_2_PI)*sigma;
            float four_sigma_pow=4*sigma*sigma;
//            float alpha_t_alpha12;
            float cons=M_2_PI/dft_size;
            float Trecurse=180.0/T;
            //开始voting算法，先计算一遍voting，算出hist数组、hist_index数组

            //combine the voting and peak
//            for (i=0;i<local_N;i++){
//                        for (j=i+1;j<local_N;j++){
//                            //be sure ij,ji!=-1;

//                            alpha_ij=((2*local_N-1-i)*i/2+j-(i+1));
//                            float tmp_voting[local_N];
//                            float tmp_hist[72]={0.0};
//                            #pragma omp parallel for
//                            for (k=0;k<local_N;k++){
//                                //....#
//                //                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
//                                if (k!=i and k!=j and cml_pair_matrix[i][j]>-1 and cml_pair_matrix[i][k]>-1 and cml_pair_matrix[j][i]>-1 and cml_pair_matrix[j][k]>-1 and cml_pair_matrix[k][i]>-1 and cml_pair_matrix[k][j]>-1) {
//                                    tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
//                                }
//                                else {
//                                    tmp_voting[k]=-10.0;
//                                }
//                            }

//                            for (int m=0;m<local_N;m++){

//                                    float tmp=tmp_voting[m];
//                                    if (tmp!=-10.0 and tmp!=-9.0){
//            #pragma omp parallel for
//                                    for (int l=0;l<T;l++){
//                                        float alpha_t_alpha12=(180.0*l/float(T))-tmp;
//                                        tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/(four_sigma_pow))/half_pow_pi;
//                                    }
//                                }
//                            }
//                            hist_peak[alpha_ij]=CMLNCV::max_float(tmp_hist,T);
//                           hist_index[alpha_ij]=CMLNCV::max_float_index(tmp_hist,T);

//                        }
//                    }


            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    long int alpha_ij=((2*double_local_N-1-i)*i/2+j-(i+1));
                    if (cml_pair_matrix[i][j]!=-1){
                        float tmp_voting[double_local_N];
                        float tmp_hist[72]={0.0};
#pragma omp parallel for
                        for (k=0;k<double_local_N;k++){
                            if (k!=i and k!=j){
                                if (cml_pair_matrix[i][k]!=-1 and cml_pair_matrix[j][k]!=-1){
                                    tmp_voting[k]=cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
                                }
                                else {tmp_voting[k]=-10.0;}
                            }
                            else {tmp_voting[k]=-10.0;}
                        }
                        for (int m=0;m<double_local_N;m++){
                            float tmp=tmp_voting[m];
                            if (tmp!=-10.0){
#pragma omp parallel for
                                for (int l=0;l<T;l++){
                                    float alpha_t_alpha12=Trecurse*l-tmp;
                                    tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/four_sigma_pow)/half_pow_pi;
                                }
                            }
                        }
                        hist_peak[alpha_ij]=max_float(tmp_hist,T);
                    }
                    else{
                    hist_peak[alpha_ij]=0.0;
                    }
                }
            }
            t_vote_1=time(NULL);
            if (hist_flag){
            fprintf(outputfile,"alpha_ij\ti\tj\thist_index\tpeak_value\n");
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index=(2*double_local_N-1-i)*i/2+j-(i+1);
                    fprintf(outputfile,"alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[index],hist_peak[index]);
                }
            }
            }
            fprintf(OUTFILE,"\n311\n");
            //从CML_Pair中找出优秀粒子保留，等于剔除non-particle
            //计算每一个alphaij的voting序号
            int NumOfHighPeak=0;

            //固定threshold值，从右侧开始取第一个峰值,先固定threshold
            //threshold 为r/sqrt（local_N),r为2、4、8等。排序取local_N*threshold个为符合条件者。
//            float max_hist_peak=CMLNCV::max_float(hist_peak,local_N*(local_N-1)/2);
            int r=4;

            //复制一个hist_peak的备份;
            float *hist_peak_cp =  new float[double_local_N*(double_local_N-1)/2];
            long hist_peak_size=double_local_N*(double_local_N-1)/2;
            int threshold_top=floor((1-(r/sqrt(double_local_N)))*hist_peak_size);
            for (i=0;i<hist_peak_size;i++){
                hist_peak_cp[i]=hist_peak[i];
            }
            sort(hist_peak_cp,hist_peak_cp+hist_peak_size);
            fprintf(OUTFILE,"threshold_top %d\n",threshold_top);
            float threshold=hist_peak_cp[threshold_top-1];
            delete[] hist_peak_cp;
            t_vote_2=time(NULL);
            int Result_voted[double_local_N];
            for (i=0;i<double_local_N;i++){
                Result_voted[i]=0;
            }
            //找出Peak值较高的CML_pair
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index = (2*double_local_N-1-i)*i/2+j-(i+1);
                    if (hist_peak[index]>threshold) {
//                        NumOfHighPeak = NumOfHighPeak +1;
                        Result_voted[i]=Result_voted[i]+1;
                        Result_voted[j]=Result_voted[j]+1;
                        }
                    }
                }
            float mean_noise=0.0;
            for (i=local_N;i<double_local_N;i++){
                mean_noise+=Result_voted[i];

            }
            mean_noise=mean_noise/local_N;
            fprintf(OUTFILE,"mean_noise\t%f\n",mean_noise);
            //排序result_voted
            std::vector<voted> V;
            for (i=0;i<double_local_N;i++){
                voted s;
                s.index=i;
                s.value=Result_voted[i];
                V.push_back(s);
            }
            std::sort(V.begin(),V.end(),comp);
            //输出排序值和结果,分两个，noise排位的平均值;随着排位上升，noise的比例
            for (i=0;i<double_local_N;i++){
                fprintf(outputfile,"%d\t%d\t%d\n",i,V[i].index,V[i].value);
            }
            int counter=0;

            for (i=0;i<double_local_N;i++){
                if (V[i].index>=local_N){
                    counter=counter+1;
                }
                if ((i+1)%100==0){
                    fprintf(OUTFILE,"%d\tcounter %d\trate\t%f\n",i,counter,counter/float(i+1));
                }
            }

            //取出高可信粒子
            /*
            const int step=sizeof(Result_voted)/sizeof(int);
            float threshold_voted=(*std::max_element(Result_voted,Result_voted+step))*0.2;
            fprintf(OUTFILE,"threshold_voted is %f\n",threshold_voted);
//            float threshold_voting=*std::max_element(Result_voting,Reslut_voting+step)*0.8;
            //only voted particles are trusted.
            std::vector<int> local_good_particle;
            for (i=0;i<double_local_N;i++){
                if (Result_voted[i]>threshold_voted){
                    int good_voted_particle=control_struct[control][0]+i;
                    local_good_particle.push_back(good_voted_particle);
                }
            }



            if (local_good_particle.size()==0){
                fprintf(outputfile,"the local_good_particle num is 0\n");
            }
            for (auto m:local_good_particle){
                fprintf(outputfile,"%d\n",m);
            }
            */
            //放入全局Global_good_particle中
//            for (auto m:local_good_particle){
//                Global_good_particle.push_back(m);
//            }

            //销毁所有堆
            delete[] lineardft_matrix;
            delete[] hist_index;
            delete[] hist_peak;
            for (i=0;i<double_local_N;i++){
                delete[] cml_pair_matrix[i];
                delete[] cml_pair_matrix_help[i];
            }
            delete[] cml_pair_matrix;
            delete[] cml_pair_matrix_help;
            t_end=time(NULL);
            fprintf(OUTFILE,"ncc_time %d\n",t_ncc_value-t_start);
	    fprintf(OUTFILE,"ncc_gpu %d\n",t_ncc_value-t_ncc_gpu);
            fprintf(OUTFILE,"voting time %d\n",t_end-t_ncc_value);
            fprintf(OUTFILE,"only voting time %d\n",t_vote_1-t_ncc_value);
            fprintf(OUTFILE,"sort time %d\n",t_vote_2-t_vote_1);
            fprintf(OUTFILE,"%d/%d\tcompleted\n",control,n_iteration);

        }

}

        fclose(f);
        fclose(outputfile);
        fclose(fnoise);
        fclose(OUTFILE);






        t_all_end=time(NULL);
        printf("all time %dhour\n",(t_all_end-t_start)/3600);

        return 0;
}

