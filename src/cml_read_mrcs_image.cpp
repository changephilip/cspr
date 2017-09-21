#include "cml.h"
#include <time.h>
#include <sstream>
#include "mrcprocess.h"
int main(int argc,char * argv[]){
    int cml_size=0;
    int indexOfmrc;
    int i,j,k,l;
    int oc;
//    int dft_size;
    char *inname;

    FILE *inmrc;

    double scale=cml_size/144.0;
    const int SIZE_COMP=128;

    while ((oc = getopt(argc,argv,"s:i:n:")) != -1)
    {
        switch(oc)
        {
            case 's':
            cml_size=atoi(optarg);
            break;
        case 'i':
            inname=optarg;
            break;
        case 'n':
            indexOfmrc=atoi(optarg);
            break;
        }
    }
    if (cml_size==0 or !inname or !indexOfmrc){
        printf("cml_size should be an int which is larger than 0\n");
        printf("or the input mrc file and output file are both needed\n");
        exit(EXIT_FAILURE);
    }

     

//    dft_size=cv::getOptimalDFTSize(cml_size);
//    int dft_size_pow=dft_size*dft_size;
    int cml_size_pow=cml_size*cml_size;

//    inmrc=fopen("/home/qjq/data/shiny-200-pf.mrcs","rb");
//    outbin=fopen("/home/qjq/data/qjq-200-data-posi","wb");
    inmrc=fopen(inname,"rb");

    long filelength;
    fseek(inmrc,0,SEEK_END);
    filelength=ftell(inmrc);
    if (((filelength-1024)%(4*cml_size_pow))!=0){
        printf("your mrcs file is wrong!\n");
        printf("the filelength of mrcs file cna't be modded by float*cml_size^2\n");
        exit(EXIT_FAILURE);
    }
    int numItem=(filelength-1024)/(4*cml_size_pow);
    printf("%d images need to be processed\n",numItem);
    if (indexOfmrc>numItem){
        printf("index out of range");
        exit(EXIT_FAILURE);
    }

    rewind(inmrc);

    //fseek(inmrc,1024,SEEK_SET+indexOfmrc*4*cml_size_pow);



        float tmp[cml_size_pow];

//        float tmpdft[dft_size_pow];

        fseek(inmrc,indexOfmrc*cml_size_pow*sizeof(tmp[0])+1024,SEEK_SET);
        fread(tmp,sizeof(tmp[0]),cml_size_pow,inmrc);

        cv::Mat image_mrc=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);


//        cv::Mat afdft_mrc=CML::imdft(image_mrc);
        cv::Mat lp_mrc(image_mrc.size(),image_mrc.type());
		
        CML::linearpolar(image_mrc,lp_mrc);
		cv::Mat B,A;
		image_mrc.convertTo(B,CV_8UC1,255,0);
		MrcProcess::showimagecpp(B);
		cv::GaussianBlur(B,B,cv::Size(43,43),7);
		cv::normalize(B,B,255.0,0.0,cv::NORM_MINMAX);
            MrcProcess::showimagecpp(B);
            MrcProcess::showimagecpp(lp_mrc);


//        MrcProcess::showimagecpp(lpdft_mrc);
        float tmp_comp[SIZE_COMP*SIZE_COMP];
        cv::Size dsize=cv::Size(SIZE_COMP,SIZE_COMP);
        cv::Mat image_comp=cv::Mat(dsize,CV_32FC1);
        cv::resize(lp_mrc,image_comp,dsize);
//        cv::normalize(lpdft_mrc,lpdft_mrc,0,1,CV_MINMAX);
        CML::image_to_mat(image_comp,tmp_comp,SIZE_COMP);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }
        cv::Size cutsize=cv::Size(SIZE_COMP-36,SIZE_COMP);
//		cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1);
        float tmp_cut[SIZE_COMP][SIZE_COMP-36];

        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP-36;k++){
                tmp_cut[j][k]=lp_mrc.at<float>(j,k+36);
            }
        }
        cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1,tmp_cut);
        cv::resize(image_cut,image_comp,dsize);
        CML::image_to_mat(image_comp,tmp_comp,SIZE_COMP);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }

            MrcProcess::showimagecpp(image_cut);
            MrcProcess::showimagecpp(image_comp);






    printf("write processed data to file successfully!\n");


    fclose(inmrc);
    return 0;

}


