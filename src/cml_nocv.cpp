#include "cml_nocv.h"

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
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
//#pragma omp parallel for
        for(j=0;j<after_dft_size;j++){
            //    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
            value[i][j] = (cblas_sdot(after_dft_size,&Ci[i*after_dft_size],1, &Cj[j*after_dft_size],1 )+after_dft_size*Qci[i][1]*Qcj[j][1]-Qci[i][1]*Qcj[j][0]-Qci[i][0]*Qcj[j][1])/(after_dft_size*Qci[i][3]*Qcj[j][3]);
//            printf("%f",value[i][j]);
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




}

