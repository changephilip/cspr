#include "cml_cl.h"

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
    float sigma1_pow=0.0f;
    float sigma2_pow=0.0f;
    int i;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
    for (i=0;i<CML_SIZE;i++){
        sigma1_pow=sigma1_pow+(cml1[i]-mean1)*(cml1[i]-mean1);
        sigma2_pow=sigma2_pow+(cml2[i]-mean2)*(cml2[i]-mean2);
    }

    sigma1=sqrt(sigma1_pow/CML_SIZE);
    sigma2=sqrt(sigma2_pow/CML_SIZE);
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
    sum1=cblas_sasum(CML_SIZE,cml1,1);
    sum2=cblas_sasum(CML_SIZE,cml2,1);
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

float CLBNCC(const float *cml1,const float *cml2,int CML_SIZE){
    float ncc;
    float sigma1;
    float sigma2;
    float mean1;
    float mean2;
    float sum1;
    float sum2;
    float sigma1_pow;
    float sigma2_pow;
    float ncc_fft;
    struct timeval tsBegin,tsEnd;
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;

    cl_mem bufcml1, bufsum1, bufcml2, bufsum2, buf_scratch, bufsigma1_pow, bufsigma2_pow, buf_scratch2, buffft, buf_scratch_fft;
    cl_event event = NULL;


    gettimeofday(&tsBegin,NULL);

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clBLAS */
    err = clblasSetup( );

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufcml1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,CML_SIZE*sizeof(*cml1),NULL, &err);
    bufcml2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,CML_SIZE*sizeof(float),NULL, &err);
    buf_scratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*CML_SIZE*sizeof(*cml1), NULL, &err);


    bufsum1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,sizeof(float), NULL, &err);
    bufsum2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,sizeof(float), NULL, &err);
    bufsigma1_pow = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    bufsigma2_pow = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    buf_scratch2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, CML_SIZE*sizeof(float), NULL, &err);
    buffft = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    buf_scratch_fft = clCreateBuffer(ctx, CL_MEM_READ_WRITE, CML_SIZE*sizeof(float), NULL, &err);

    err = clEnqueueWriteBuffer( queue, bufcml1, CL_FALSE, 0,
        CML_SIZE * sizeof( float ), cml1, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufcml2, CL_FALSE, 0,
        CML_SIZE * sizeof( float ), cml2, 0, NULL, NULL);
//    printf("\n enqueue write buffer %d\n",err);
/* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */

//    sum1=cblas_sasum(CML_SIZE,cml1,1);

    err = clblasSasum( CML_SIZE, bufsum1, 0, bufcml1, 0, 1, buf_scratch, 1, &queue, 0, NULL, &event);
    /* Wait for calculations to be finished. */
//    err = clWaitForEvents( 1, &event );
    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadBuffer( queue, bufsum1, CL_FALSE, 0, sizeof(sum1), &sum1, 0, NULL, NULL);



    err = clblasSasum( CML_SIZE, bufsum2, 0, bufcml2, 0, 1, buf_scratch, 1, &queue, 0, NULL, &event);
//    err = clWaitForEvents( 1, &event );
    err = clEnqueueReadBuffer( queue, bufsum2, CL_FALSE, 0, sizeof(sum2), &sum2, 0, NULL, NULL);


    err = clblasSdot( CML_SIZE, bufsigma1_pow, 0, bufcml1, 0, 1, bufcml1, 0, 1, buf_scratch2, 1, &queue, 0,NULL,&event);
//    err = clWaitForEvents( 1, &event );
    err = clEnqueueReadBuffer( queue, bufsigma1_pow, CL_FALSE, 0,sizeof(sigma1_pow), &sigma1_pow, 0, NULL, NULL);

    err = clblasSdot( CML_SIZE, bufsigma2_pow, 0, bufcml2, 0, 1, bufcml2, 0, 1, buf_scratch2, 1, &queue, 0,NULL,&event);
//    err = clWaitForEvents( 1, &event );
    err = clEnqueueReadBuffer( queue, bufsigma2_pow, CL_FALSE, 0,sizeof(sigma2_pow), &sigma2_pow, 0, NULL, NULL);

    err = clblasSdot( CML_SIZE, buffft, 0, bufcml1, 0, 1, bufcml2, 0, 1, buf_scratch_fft, 1, &queue, 0,NULL,&event);
//    err = clWaitForEvents( 1, &event );
    err = clEnqueueReadBuffer( queue, buffft, CL_FALSE, 0,sizeof(ncc_fft), &ncc_fft, 0, NULL, NULL);

    clReleaseMemObject( bufcml1 );
    clReleaseMemObject( bufcml2 );
    clReleaseMemObject( buf_scratch );
    clReleaseMemObject( bufsigma1_pow);
    clReleaseMemObject( bufsigma2_pow);
    clReleaseMemObject( buf_scratch2);
    clReleaseMemObject( buffft);
    clReleaseMemObject( buf_scratch_fft );
    clblasTeardown( );
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );
//    printf("\nsum1 %f\n",sum1);
//    sum2=cblas_sasum(CML_SIZE,cml2,1);
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
//    sigma1_pow=cblas_sdot(CML_SIZE,cml1,1,cml1,1)+CML_SIZE*mean1*mean1-2*mean1*sum1;
//    sigma2_pow=cblas_sdot(CML_SIZE,cml2,1,cml2,1)+CML_SIZE*mean2*mean2-2*mean2*sum2;
    sigma1_pow=sigma1_pow+CML_SIZE*mean1*mean1-2*mean1*sum1;
    sigma2_pow=sigma2_pow+CML_SIZE*mean2*mean2-2*mean2*sum2;

    sigma1=sqrt(sigma1_pow/CML_SIZE);
    sigma2=sqrt(sigma2_pow/CML_SIZE);
    float coeff;
    coeff = 1/(float(CML_SIZE)*sigma1*sigma2);
//    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));

    float ncc_2;
    float ncc_3;
    float ncc_4;
    ncc_2 = CML_SIZE*mean1*mean2;
    ncc_3 = mean1*sum2;
    ncc_4 = mean2*sum1;
//    ncc_fft=cblas_sdot(CML_SIZE,cml1,1,cml2,1);
    ncc=coeff*(ncc_fft+ncc_2-ncc_3-ncc_4);
    //printf("\n in NCC0\n");
    gettimeofday(&tsEnd,NULL);
    printf("\t%ld\t",1000000L*(tsEnd.tv_sec-tsBegin.tv_sec)+tsEnd.tv_usec-tsBegin.tv_usec);
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

float cvoting(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size){
    double a,b,c;
//    double two_pi=6.28318530;
    float angleij;
    double cons=180.0/M_PI;
    a = cos(float(cmlkj-cmlki)*M_2_PI/float(after_dft_size));
    b = cos(float(cmljk-cmlji)*M_2_PI/float(after_dft_size));
    c = cos(float(cmlik-cmlij)*M_2_PI/float(after_dft_size));
//    if ((1-c*c)<0 or (1-b*b)<0){
//        printf("error in cvoting\n");
//        exit(EXIT_FAILURE);
//    }
    float t;
    if ((1+2*a*b*c)>a*a+b*b+c*c) {
        t=(a-b*c)/(sqrt(1-b*b)*sqrt(1-c*c));
        if (t<=1 and t>=-1){
        angleij = acos(t)*cons;
        }
        else if(t>=1){
         angleij=acos(1.0)*cons;
        }
        else if (t<=-1){
         angleij=acos(-1.0)*cons;
        }
    }
    else {angleij=-9.0;}
    return angleij;
}

bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size){
    double a,b,c;
//    double two_pi=6.28318530;
//    float cos_angleij,angleij;
    a = cos(float(cmlkj-cmlki)*M_2_PI/float(after_dft_size));
    b = cos(float(cmljk-cmlji)*M_2_PI/float(after_dft_size));
    c = cos(float(cmlik-cmlij)*M_2_PI/float(after_dft_size));
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
//    cl_int err;
//    cl_platform_id platform = 0;
//    cl_device_id device = 0;
//    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
//    cl_context ctx = 0;
//    cl_command_queue queue = 0;
    //mpi here
    for(i=0;i<after_dft_size;i++){

//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            value[i][j] = CLBNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
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
            printf("%f\t%f\t",value[i][j],fncc_v);
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




}


