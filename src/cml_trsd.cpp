#include "cml.h"
#include <time.h>
#include <sstream>
//need link mpi
int main()
{
    int cml_size;
    int i;


    cml_size=200;
    int dft_size=cv::getOptimalDFTSize(cml_size);
    int dft_size_pow=dft_size*dft_size;
//    std::ifstream mainMrcStarFile("/home/qjq/data/shiny-200-pf.star");
    printf("%d\tdFt_size\n",dft_size);
//    fileNameToCoodinateList table=CML::CNNpicker(mainMrcStarFile);
    int numItem = 25530;
    FILE *f;
    FILE *fmrc;
    fmrc=fopen("/home/qjq/data/shiny-200-pf.mrcs","rb");
    printf("000001\n");
    f=fopen("/home/qjq/data/qjq_200_data","wb");
    printf("000002\n");
    printf("\nnumItem\t%d\n",numItem);
    printf("\nmalloc success\n");
    char sbuf[1960000];
    setvbuf(f,sbuf,_IOFBF,1960000);
    fseek(fmrc,0,SEEK_END);
    long filelength;

    filelength=ftell(fmrc);
    printf("mrcs length %ld\n",filelength);
    rewind(fmrc);
    fseek(fmrc,1024,SEEK_SET);
    fseek(f,0,SEEK_SET);
    int pow_cml_size=cml_size*cml_size;
    for(i=0;i<numItem;i++){
        float tmp[pow_cml_size];
        float tmpdft[dft_size_pow];
        fread(tmp,sizeof(float),pow_cml_size,fmrc);
        cv::Mat image_tmp=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);
        cv::Mat afdft_img=CML::imdft(image_tmp);
        cv::Mat ldft_img(afdft_img.size(),afdft_img.type());
        CML::linearpolar(afdft_img,ldft_img);
        CML::image_to_mat(ldft_img,tmpdft,dft_size);
        fwrite(tmpdft,sizeof(float),dft_size_pow,f);
    }


    fclose(f);
    fclose(fmrc);

    return 0;
    }

