#include "cml.h"
#include <time.h>
#include <sstream>
#include "mrcprocess.h"
//need link mpi
int main()
{
    int cml_size;
    int i;


    cml_size=200;
    int dft_size=cv::getOptimalDFTSize(cml_size);
    int dft_size_pow=dft_size*dft_size;
    printf("%d\tdFt_size\n",dft_size);

    FILE *f;
    FILE *fmrc;

    fmrc=fopen("/home/qjq/data/shiny-200-pf.mrcs","rb");
    f=fopen("/home/qjq/data/qjq_200_data_posi","wb");
    printf("\nmalloc success\n");
    long filelength;
    fseek(fmrc,0,SEEK_END);
    filelength=ftell(fmrc);
    int numItem=(filelength-1024)/(4*cml_size*cml_size);

    printf("numItem %d\n",numItem);
    rewind(fmrc);


    fseek(fmrc,1024,SEEK_SET);
//    fseek(f,0,SEEK_SET);
    int pow_cml_size=cml_size*cml_size;


//write to disk
    for(i=0;i<4;i++){
        float tmp[pow_cml_size];
        float tmpdft[dft_size_pow];
        fseek(fmrc,i*pow_cml_size*sizeof(float)+1024,SEEK_SET);
        long postion=ftell(fmrc);
        printf("posi %ld\n",postion);
        fread(tmp,sizeof(float),pow_cml_size,fmrc);


        cv::Mat image_tmp=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);
        cv::Mat afdft_img=CML::imdft(image_tmp);
//        MrcProcess::showimagecpp(afdft_img);
        cv::Mat ldft_img(afdft_img.size(),afdft_img.type());
        CML::linearpolar(afdft_img,ldft_img);
        cv::normalize(ldft_img,ldft_img,0,1,CV_MINMAX);
//        MrcProcess::showimagecpp(ldft_img);
        CML::image_to_mat(ldft_img,tmpdft,dft_size);

        fwrite(tmpdft,sizeof(float),dft_size_pow,f);
    }





//    fclose(f);


    //test
    FILE *ftest;
    ftest=fopen("/home/qjq/data/qjq_200_data_posi","rb");
//    cv::Mat testwrite=cv::Mat(dft_size,dft_size,CV_32FC1,test);
//    MrcProcess::showimagecpp(testwrite);

    fseek(ftest,0,SEEK_SET);
    fseek(fmrc,1024,SEEK_SET);
    float a[dft_size_pow];
    float b[dft_size_pow];
    float c[dft_size_pow];
    float d[dft_size_pow];
    float a2[dft_size_pow];
    float b2[dft_size_pow];
    float c2[dft_size_pow];
    float d2[dft_size_pow];
    for (i=0;i<4;i++){
        float tmp[pow_cml_size];
        float tmpdft[dft_size_pow];
        float testread[dft_size_pow];
        fseek(fmrc,i*pow_cml_size*sizeof(float),SEEK_SET);
        printf("fmrc\t%ld\tf\t%ld\t",ftell(fmrc),ftell(ftest));

        fread(tmp,sizeof(tmp[0]),pow_cml_size,fmrc);

        cv::Mat image_tmp=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);
        cv::Mat afdft_img=CML::imdft(image_tmp);

        cv::Mat ldft_img(afdft_img.size(),afdft_img.type());
        CML::linearpolar(afdft_img,ldft_img);
        cv::normalize(ldft_img,ldft_img,0,1,CV_MINMAX);
        CML::image_to_mat(ldft_img,tmpdft,dft_size);

        fseek(ftest,i*dft_size_pow*sizeof(float),SEEK_SET);
        fread(testread,sizeof(float),dft_size_pow,ftest);

        if (i==0){
            CML::image_to_mat(ldft_img,a,dft_size);
        }
        if (i==1){
            CML::image_to_mat(ldft_img,b,dft_size);
        }
        if (i==2){
            CML::image_to_mat(ldft_img,c,dft_size);
        }
        if (i==3){
            CML::image_to_mat(ldft_img,c,dft_size);
        }
        float sum_pow=0.0f;
        for (int k=0;k<dft_size_pow;k++){
//            if (testread[k]!=tmpdft[k]){
//                printf("diff %d\t%d\t%f\n",i,k,testread[k]-tmpdft[k]);
//            }
            sum_pow+=(testread[k]-tmpdft[k])*(testread[k]-tmpdft[k]);
        }
        sum_pow=sqrt(sum_pow/dft_size_pow);
        printf("i=%d\tsumpow=%f\n",i,sum_pow);
    }





    printf("test 0\n");

    fseek(ftest,0,SEEK_SET);
    fread(a2,sizeof(a2[0]),200*200,ftest);
    fread(b2,sizeof(b2[0]),200*200,ftest);
    fread(c2,sizeof(c2[0]),200*200,ftest);
    fread(d2,sizeof(d2[0]),200*200,ftest);
    printf("a\n");
    for (i=0;i<200;i++){
        printf("%f\t",a[i]);
    }
    printf("\n");
    for (i=0;i<200;i++){
        printf("%f\t",a2[i]);
    }
    printf("\n");
    printf("b\n");
    for (i=0;i<200;i++){
        printf("%f\t",b[i]);
    }
    printf("\n");
    for (i=0;i<200;i++){
        printf("%f\t",b2[i]);
    }
    printf("c\n");
    for (i=0;i<200;i++){
        printf("%f\t",c[i]);
    }
    printf("\n");
    for (i=0;i<200;i++){
        printf("%f\t",c2[i]);
    }
    printf("d\n");
    for (i=0;i<200;i++){
        printf("%f\t",d[i]);
    }
    printf("\n");
    for (i=0;i<200;i++){
        printf("%f\t",d2[i]);
    }
    fclose(fmrc);
    fclose(ftest);
    return 0;
    }

