#include "cml_cuda.h"

namespace CMLNCV{


float NCC0(float *cml1,float *cml2,int CML_SIZE){
    //cml1,cml2,length=CML_SIZE,one-dem array
    //mpi
    float ncc;
    float sigma1;
    float sigma2;
    float mean1;
    float mean2;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sigma1_dot=0.0f;
    float sigma2_dot=0.0f;
    int i;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
    for (i=0;i<CML_SIZE;i++){
        sigma1_dot=sigma1_dot+cml1[i]*cml1[i];
        sigma2_dot=sigma2_dot+cml2[i]*cml2[i];
    }

    sigma1=sqrt((sigma1_dot+CML_SIZE*mean1*mean1-2*mean1*sum1)/CML_SIZE);
    sigma2=sqrt((sigma2_dot+CML_SIZE*mean2*mean2-2*mean2*sum2)/CML_SIZE);
    float coeff;
    coeff = 1/(float(CML_SIZE)*sigma1*sigma2);
//    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
    float ncc_fft=0.0;//ncc_0
    float ncc_2;
    float ncc_3;
    float ncc_4;
    ncc_2 = CML_SIZE*mean1*mean2;
    ncc_3 = mean1*sum2;
    ncc_4 = mean2*sum1;
//    float sum_ncc_fft=0.0;


    for(i=0;i<CML_SIZE;i++){
        ncc_fft=ncc_fft+cml1[i]*cml2[i];

    }

//    for (i=0;i<CML_SIZE;i++){
//        sum_ncc_fft=sum_ncc_fft+ncc_fft[i] ;
//    }
    ncc=coeff*(ncc_fft+ncc_2-ncc_3-ncc_4);
    //printf("\n in NCC0\n");
//    printf("\t%f\t",ncc);
    return ncc;
}


float FNCC(float *cml1,float *cml2,int CML_SIZE){
    //cml1,cml2,length=CML_SIZE,one-dem array
    //mpi
    float ncc;
    float mean1;
    float mean2;
    float sum1 = 0.0;
    float sum2 = 0.0;

    int i;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
    int Ncc=0;
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
    for (i=0;i<CML_SIZE;i++){
        if (cml1[i]>=mean1) {
            if (cml2[i]>=mean2){
                Ncc+=1;
            }
            else Ncc+=-1;
        }
        else if (cml2[i]>=mean2){
            Ncc+=-1;
        }
        else Ncc+=1;
    }
    ncc=(float)Ncc/CML_SIZE;
    return ncc;
}

float MYSUM(int Num,const float *p){
    float re=0.0f;
    int i;
    for(i=0;i<Num;i++){
        re=re+p[i];
    }
    return re;
}

float BNCC(const float *cml1,const float *cml2,int CML_SIZE){
    float ncc;
    float sigma1;
    float sigma2;
    float mean1;
    float mean2;
    float sum1;
    float sum2;
    float sigma1_pow;
    float sigma2_pow;
    sum1=MYSUM(CML_SIZE,cml1);
    sum2=MYSUM(CML_SIZE,cml2);
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
    sigma1_pow=cblas_sdot(CML_SIZE,cml1,1,cml1,1)+CML_SIZE*mean1*mean1-2*mean1*sum1;
    sigma2_pow=cblas_sdot(CML_SIZE,cml2,1,cml2,1)+CML_SIZE*mean2*mean2-2*mean2*sum2;

    sigma1=sqrt(sigma1_pow/CML_SIZE);
    sigma2=sqrt(sigma2_pow/CML_SIZE);
    float coeff;
    coeff = 1/(float(CML_SIZE)*sigma1*sigma2);
//    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
    float ncc_fft;
    float ncc_2;
    float ncc_3;
    float ncc_4;
    ncc_2 = CML_SIZE*mean1*mean2;
    ncc_3 = mean1*sum2;
    ncc_4 = mean2*sum1;
    ncc_fft=cblas_sdot(CML_SIZE,cml1,1,cml2,1);
    ncc=coeff*(ncc_fft+ncc_2-ncc_3-ncc_4);
    //printf("\n in NCC0\n");
//    printf("\t%f\t",ncc);
    return ncc;
}

float cal_angle(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size){
    double a,b,c;
//    double two_pi=6.28318530;
    float cos_angleij,angleij;
    a = cos(float(cmlkj-cmlki)*M_2_PI/float(after_dft_size));
    b = cos(float(cmljk-cmlji)*M_2_PI/float(after_dft_size));
    c = cos(float(cmlik-cmlij)*M_2_PI/float(after_dft_size));
    cos_angleij = (a-b*c)/(sqrt(1-b*b)*sqrt(1-c*c));
//    printf("\ncos_angle %f\n",cos_angleij);
    angleij = acos(cos_angleij);
    return angleij*180.0/M_PI;
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

bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,float cons){
    double a,b,c;
//    double two_pi=6.28318530;
//    float cos_angleij,angleij;
    a = cos((cmlkj-cmlki)*cons);
    b = cos((cmljk-cmlji)*cons);
    c = cos((cmlik-cmlij)*cons);
    if ((1+2*a*b*c)>a*a+b*b+c*c) {
        return true;
    }
        else
    {return false;}
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
            value[i][j] = BNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
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
            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
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
    //cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,after_dft_size,after_dft_size,after_dft_size,1,Ci,after_dft_size,Cj,after_dft_size,0,C1,after_dft_size);
	/*
    for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }
*/
    //simple test cublassgemm,if it work more quickly?
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_a,*d_b,*d_c;
/*
    checkCudaErrors(cudaMalloc((void**)&d_a,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_b,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_c,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Ci,1,d_a,1));
    checkCudaErrors(cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Cj,1,d_a,1));
    cudaThreadSynchronize();
    checkCudaErrors(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,after_dft_size,after_dft_size,after_dft_size,1,d_b,after_dft_size,d_a,after_dft_size,0,d_c,after_dft_size));
*/
    float alpha=1.0;
    float beta=0.0;
    cudaMemcpy(C,d_c,after_dft_size*after_dft_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMalloc((void**)&d_a,after_dft_size*after_dft_size*sizeof(float));
    cudaMalloc((void**)&d_b,after_dft_size*after_dft_size*sizeof(float));
    cudaMalloc((void**)&d_c,after_dft_size*after_dft_size*sizeof(float));
    cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Ci,1,d_a,1);
    cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Cj,1,d_b,1);
    cudaThreadSynchronize();
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,after_dft_size,after_dft_size,after_dft_size,&alpha,d_b,after_dft_size,d_a,after_dft_size,&beta,d_c,after_dft_size);
    cudaMemcpy(C,d_c,after_dft_size*after_dft_size*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
	for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }

/* 
    float sum=0.0;
    for(i=0;i<after_dft_size*after_dft_size;i++){
	sum+=(C1[i]-C[i])*(C1[i]-C[i]);
    }
    sum=sqrt(sum/after_dft_size/after_dft_size);
    printf("diff betwenn cublas and blas\t %f\n",sum);*/
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


cmlncv_tuple NCC_QT_check(float **Qci,float **Qcj,float *Ci,float *Cj,int after_dft_size){
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
    float C_trd[after_dft_size*after_dft_size];
    float C_cpublas[after_dft_size*after_dft_size];
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,after_dft_size,after_dft_size,after_dft_size,1,Ci,after_dft_size,Cj,after_dft_size,0,C_cpublas,after_dft_size);
    for (i=0;i<after_dft_size;i++){
        for (j=0;j<after_dft_size;j++){
            C_trd[i*after_dft_size+j]=cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1,&Cj[j*after_dft_size],1);
        }
    }


    /*
    for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }
*/
    //simple test cublassgemm,if it work more quickly?
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_a,*d_b,*d_c;
/*
    checkCudaErrors(cudaMalloc((void**)&d_a,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_b,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_c,after_dft_size*after_dft_size*sizeof(float)));
    checkCudaErrors(cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Ci,1,d_a,1));
    checkCudaErrors(cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Cj,1,d_a,1));
    cudaThreadSynchronize();
    checkCudaErrors(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,after_dft_size,after_dft_size,after_dft_size,1,d_b,after_dft_size,d_a,after_dft_size,0,d_c,after_dft_size));
*/
    float alpha=1.0;
    float beta=0.0;
    cudaMemcpy(C,d_c,after_dft_size*after_dft_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMalloc((void**)&d_a,after_dft_size*after_dft_size*sizeof(float));
    cudaMalloc((void**)&d_b,after_dft_size*after_dft_size*sizeof(float));
    cudaMalloc((void**)&d_c,after_dft_size*after_dft_size*sizeof(float));
    cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Ci,1,d_a,1);
    cublasSetVector(after_dft_size*after_dft_size,sizeof(float),Cj,1,d_b,1);
    cudaThreadSynchronize();
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,after_dft_size,after_dft_size,after_dft_size,&alpha,d_b,after_dft_size,d_a,after_dft_size,&beta,d_c,after_dft_size);
    cudaMemcpy(C,d_c,after_dft_size*after_dft_size*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    //check GPU_cudablas CPU_sgemm CPU_sdot,C,C_cpublas,C_trd
    float sum=0.0;
    for (i=0;i<after_dft_size*after_dft_size;i++){
        sum+=(C[i]-C_cpublas[i])*(C[i]-C_cpublas[i]);
    }
    sum=sqrt(sum/after_dft_size/after_dft_size);
    printf("diff between cublas and cpublas\t %f\n",sum);

    sum=0.0;
    for (i=0;i<after_dft_size*after_dft_size;i++){
        sum+=(C[i]-C_trd[i])*(C[i]-C_trd[i]);
    }
    sum=sqrt(sum/after_dft_size/after_dft_size);
    printf("diff between cublas and cpu_sdot\t %f\n",sum);

    sum=0.0;
    for (i=0;i<after_dft_size*after_dft_size;i++){
        sum+=(C_cpublas[i]-C_trd[i])*(C_cpublas[i]-C_trd[i]);
    }
    sum=sqrt(sum/after_dft_size/after_dft_size);
    printf("diff between cpu_blas and cpu_sdot\t %f\n",sum);




    for (i=0;i<after_dft_size;i++){
//#pragma omp parallel for
        for (j=0;j<after_dft_size;j++){
            value[i][j]=(C[i*after_dft_size+j]+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
        }
    }

/*
    float sum=0.0;
    for(i=0;i<after_dft_size*after_dft_size;i++){
    sum+=(C1[i]-C[i])*(C1[i]-C[i]);
    }
    sum=sqrt(sum/after_dft_size/after_dft_size);
    printf("diff betwenn cublas and blas\t %f\n",sum);*/
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

cmlncv_tuple NCC_valuet(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cmlncv_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    printf("\n");
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for

        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
            float fncc_v= BNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
            printf("%f\t%f\t\n",value[i][j],fncc_v);
//            printf("\n000003");
        }
        printf("\n");

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


/*
void cml_read(float *data,fileNameToCoodinateList intable,int cml_size){
    auto map_it = intable.cbegin();
    int ck=0;
    while (map_it != intable.cend()){
        FILE *tmpf;
        const char *tmpmrcfile = (map_it->first).c_str();
        tmpf = fopen(tmpmrcfile,"r");
        struct _mrchead hf;
        hf = MrcProcess::readhead(tmpf);
        float mrc[cml_size*cml_size];
//        mrc = MrcProcess::mrcOrigin(tmpf);//it has Gauss_Equal_Norm;
        MrcProcess::readmrcdata(tmpf,mrc,hf);
        for (auto m:(map_it -> second)){
            CML::getsubmrc(mrc,m.x,m.y,cml_size,&data[ck*cml_size*cml_size],hf);
            ck++;
        }
    ++map_it;
    fclose(tmpf);
    }
}
*/


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




/*
void flambda(){
    //简单kernel

}

//来源于网络
__global__ void Max_Reduce(int *d_array, int array_len, int *max_value, int *max_idx)
            {
                           __share__ int temp_value_share[warp_size];
                           __share__ int temp_idx_share[warp_size];
                          int tid=thread.x+blockDim.x*blockIdx.x;
                          int i,temp_value,temp_value1,temp_idx,temp_idx1;
                          int warpid=thread.x/warp_size,laneid=thread.x%warp_size;
                            temp_value=-1e30;
                           temp_idx=thread.x;
                          if(tid<n)
                          {
                                    temp_value=d_array[tid];
                           }
                                 for(i=warp_size/2;i>=1;i/=2)
                                 {
                                         temp_value1=shft_xor(temp_value,i,warp_size);
                                         temp_idx1=shft_xor(temp_idx,i,warp_size);
                                        if(temp_value<temp_value1)
                                        {
                                                temp_value=temp_value1;
                                                temp_idx=temp_idx1;
                                         }
                                       else if(temp_value=temp_value1)
                                       {
                                                 if(temp_idx>temp_idx1)
                                                {
                                                         temp_idx=temp_idx1;
                                                }
                                        }
                                  }
                               if(!laneid)
                               {
                                      temp_value_share[warpid]=temp_value;
                                      temp_idx_share[warpid]=temp_idx;
                              }
                            __sychthreads();
                           if(thread.x<warp_size)
                          {
                                   temp_value=temp_value_share[thread.x];
                                   temp_idx=temp_idx_share[thread.x];
                                  for(i=warp_size/2;i>=1;i/=2)
                                 {
                                         temp_value1=shft_xor(temp_value,i,warp_size);
                                         temp_idx1=shft_xor(temp_idx,i,warp_size);
                                        if(temp_value<temp_value1)
                                        {
                                                temp_value=temp_value1;
                                                temp_idx=temp_idx1;
                                         }
                                         else if(temp_value=temp_value1)
                                       {
                                                 if(temp_idx>temp_idx1)
                                                {
                                                         temp_idx=temp_idx1;
                                                }
                                        }
                                  }
                           }
                             if(!thread.x)
                            {
                                    max_value[blockIdx.x]=temp_value;
                                    max_idx[block.x]=temp_idx;
                             }
             }

void parent_ncc_kernel(float *d_data,int *d_ctr,int *d_ctr_id1,int *d_ctr_id2,float *d_sum,float *d_mean,float *d_stdv,int N,int L,cml_retstruc *S,){
     //获取局部id
     //设置cublas环境，启动cublas_sgemm
     //设置局部变量C3,接受sgemmm的结果，估计为160K,调用子内核时，不能使用local memory,必须把C3分配在global memory
     //调整方案，不使用子内核调用，直接部署代码
    int globalThreadID=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    int image_1=ctr_id1[globalThreadID];
    int image_2=ctr_id2[globalThreadID];
    int L_power=L*L;
    int i,j;
    cublasHandle_t handle;
    double alpha=1.0;
    double beta=0.0;
    __local__ float C3[L*L];
    //cudaMalloc((void**)&C3,L*L*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,&data[L_power*image_2],N,&data[L_power*image_1],N,&beta,C3,N);
    cublasDestroy(handle);
    //help矩阵排列，直接在主机端排列好？还是kernel调用？
    for (i=0;i<L;i++){
        //image_1*L+i
        for (j=0;j<L;j++){
            //image_2*L+j
            C[i*L+j]=(C[i*L+j]+L*help[image_1*3].y*help[image_2*3].y-xy-yx)/(N*z*z);
            C[i*L+j]=(C[i*L+j]+L*d_mean[image_1*L+i]*d_mean[image_2*L+j]-d_sum[image_1*L+i]*d_mean[iamge_2*L+j]-d_mean[image_1*L+i]*d_sum[image_2*L+j])/(N*d_stdv[iamge_1*L+i]*d_stdv[image_2*L+j])
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
        if (C[i*L+j]>max_value){
            max_value=C[i*L+j];
            max_index_i=i;
            max_index_j=j;
            }
        }
    }
    S[globalThreadID].value=max_value;
    S[globalThreadID].x=max_index_i;
    S[globalThreadID].y=max_index_j;

}

void wrapper_kernel(float *data,int N,int cml_size,float ***help,cml_retstruc *S){
    //wrapper_kernel前应该完成，数据打包成一个长数组
    //设置控制矩阵
    //读取数据接口，数值矩阵，，返回值矩阵，设置cuda环境，启动kernel
    //返回值矩阵，包括cml_matrix的value和坐标
    int control_size=N*(N-1)/2;
    int i,j;

    int BLOCK_SIZE;//理论上没有上限
    int THREAD_PER_BLOCK;//(<512,根据显卡设备的cuda参数定)
    //配置控制矩阵，alpha_ij序数控制,ctr为alphaij序数
    ctr= new int [control_size];
    ctr_id1 = new int [control_size];
    ctr_id2 = new int [control_size];
    for (i=0;i<control_size;i++){
        ctr[i]=i;
    }
    for (i=0;i<control_size;i++){
        for (j=i+1;j<control_size;j++){
            ctr_id1 = i;
            ctr_id2 = j;
        }
    }
    //配置辅助矩阵help.拆分成三个数组，每个数组为N×L
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
    Kernel parent_ncc_kernel<<<dimGrid,dimBlock>>>(d_data,d_ctr,d_ctr_id1,d_ctr_id2,d_sum,d_mean,d_stdv,N,L,d_S);
    cudaMemcpy(S,d_S,sizeof(cml_retstruc)*control_size,cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_ctr);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    cudaFree(d_S);
    //使用一个简单的kernel,不使用child kernel调用。
    //flambda需要的辅助矩阵，设置为线性格式

    //分配cuda内存，把数据矩阵、辅助矩阵存入

    //返回值矩阵，线性，分配内存

    //设置网格、线程参数，启动parent_ncc_kernel



}


*/

}
