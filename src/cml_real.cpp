#include "cml.h"
#include <time.h>
#include <sstream>
#include "mrcprocess.h"
int main(int argc,char * argv[]){
    int cml_size=0;
    int i,j,k,l;
    int oc;
//    int dft_size;
    char *inname;
    char *outname;
    FILE *inmrc;
    FILE *outbin;
    double scale=cml_size/144.0;
    const int SIZE_COMP=128;

    while ((oc = getopt(argc,argv,"s:i:o:")) != -1)
    {
        switch(oc)
        {
            case 's':
            cml_size=atoi(optarg);
            break;
        case 'i':
            inname=optarg;
            break;
        case 'o':
            outname=optarg;
            break;
        }
    }
    if (cml_size==0 or !inname or !outname){
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
    outbin=fopen(outname,"wb");

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
    rewind(inmrc);

    fseek(inmrc,1024,SEEK_SET);


    for (i=0;i<numItem;i++){
        float tmp[cml_size_pow];

//        float tmpdft[dft_size_pow];

        fseek(inmrc,i*cml_size_pow*sizeof(tmp[0])+1024,SEEK_SET);
        fread(tmp,sizeof(tmp[0]),cml_size_pow,inmrc);

        cv::Mat image_mrc=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);


//        cv::Mat afdft_mrc=CML::imdft(image_mrc);
        cv::Mat lp_mrc(image_mrc.size(),image_mrc.type());
        CML::linearpolar(image_mrc,lp_mrc);
        if (i==0 or i==5 or i==43){
            MrcProcess::showimagecpp(image_mrc);
            MrcProcess::showimagecpp(lp_mrc);
        }
		
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
        if (i==0 or i==5 or i==43){
            MrcProcess::showimagecpp(image_cut);
            MrcProcess::showimagecpp(image_comp);
        }
        fwrite(tmp_comp,sizeof(tmp_comp[0]),SIZE_COMP*SIZE_COMP,outbin);

    }

    fclose(outbin);
    printf("write processed data to file successfully!\n");
    //test

    fseek(inmrc,1024,SEEK_SET);

    FILE *testbin;
//    testbin=fopen("/home/qjq/data/qjq-200-data-posi","rb");
    testbin=fopen(outname,"rb");

    for (i=0;i<numItem;i++){
        float tmp[cml_size_pow];
//        float tmpdft[dft_size_pow];

        fseek(inmrc,i*cml_size_pow*sizeof(tmp[0])+1024,SEEK_SET);
        fread(tmp,sizeof(tmp[0]),cml_size_pow,inmrc);

        cv::Mat image_mrc=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);
//        cv::Mat afdft_mrc=CML::imdft(image_mrc);
        cv::Mat lp_mrc(image_mrc.size(),image_mrc.type());
        CML::linearpolar(image_mrc,lp_mrc);\

//        MrcProcess::showimagecpp(lpdft_mrc);

//        cv::normalize(lpdft_mrc,lpdft_mrc,0,1,CV_MINMAX);
        float tmp_comp[SIZE_COMP*SIZE_COMP];
        cv::Size dsize=cv::Size(SIZE_COMP,SIZE_COMP);
        cv::Mat image_comp=cv::Mat(dsize,CV_32FC1);
        cv::resize(lp_mrc,image_comp,dsize);
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


        float tmptest[SIZE_COMP*SIZE_COMP];
        fread(tmptest,sizeof(tmptest[0]),SIZE_COMP*SIZE_COMP,testbin);

        float sum_pow=0.0;
        for ( j=0;j<SIZE_COMP*SIZE_COMP;j++){
            sum_pow=sum_pow+(tmptest[j]-tmp_comp[j])*(tmptest[j]-tmp_comp[j]);
        }
        sum_pow=sqrt(sum_pow/SIZE_COMP*SIZE_COMP);
        if (sum_pow>0.0001){
            printf("i=%d\tsumpow=%f\n",i,sum_pow);
        }
        /*
        printf("i=%d\n",i);
        for ( j=0;j<dft_size*dft_size;j++){
            printf("%f\t%f\t%f\n",tmptest[j],tmpdft[j],tmptest[j]-tmpdft[j]);
        }
        printf("\n");
        */
//        for (int j=0;j<2*dft_size;j++){
//            printf("%f\t",);
//        }
//        printf("\n");


    }

    printf("Test is Done\nSee the screen log\n");

    fclose(testbin);

    fclose(inmrc);
    return 0;

}

