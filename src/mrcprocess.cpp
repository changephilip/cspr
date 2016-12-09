#include "mrcprocess.h"

/*
 MrcProcess::MrcProcess()
{
}
*/

namespace MrcProcess {

struct _mrchead readhead(FILE *f){
    //struct
    //int n;
    //n=1;

    struct _mrchead mrc;
    if (NULL==f){
        printf("cannot open file\n");}
    fread(&mrc,sizeof(struct  _mrchead),1,f);
 /*
    printf("%ld\n",sizeof(struct _mrchead));
    printf("%d\n",mrc.nx);
    printf("%d\n",mrc.ny);
    printf("%d\n",mrc.nz);
    printf("%f\n",mrc.mode);
*/

    return mrc;
}
//struct _mrchead readhead(std::fstream &file){
//    struct _mrchead mrc;
//    //#start here
//    if     (file.is_open()) {
//        file.read((char*)&mrc,sizeof(struct _mrchead));
//    }
//    else std::cout<<"cannot open file" <<std::endl;
//    std::cout<<mrc.nx<<"\t"<<mrc.ny<<std::endl;
//    return mrc;
//}

void readmrcdata(FILE  *f,float* dataperfile,struct _mrchead mrchead){
    int columns;
    int rows;
    columns=mrchead.nx;
    rows=mrchead.ny;
    //printf("%d\n",columns);
    //printf("%d\n",rows);
    fseek(f,mrcsize,0);
    //we can change the prog to decrease nums of fseek,then may save time
    for (int i=0;i<columns*rows;++i){
        //fseek(f,mrcsize+4*i,0);
        //size_t s=sizeof(struct _mrchead);
        fread(&dataperfile[i],4,1,f);//float 4;
        //printf("%d\t",i);
        //printf("%f\n",dataperfile[i]);
        }
    //fread(dataperfile,4,1,f);
    //printf("ssss%f\n",dataperfile[columns*rows-1]);
    //return 0;
}

cv::Mat mynorm(const cv::Mat &image)
{
    double maxVal=0;
    double minVal=0;
    cv::Point maxLoc;
    cv::minMaxLoc(image,&minVal,&maxVal);
    //#####you need start at here
    //iterator to change the element of matrix
    cv::Mat_<uchar> image2=image;
    cv::Mat_<uchar>::iterator it = image2.begin();
    cv::Mat_<uchar>::iterator itend = image2.end();
    for (;it!=itend;++it){
       (*it)=255.0*((*it)-minVal)/(maxVal-minVal);
    }
    return image2;
}


int showimagecpp(cv::Mat image){
        cv::namedWindow("image",CV_WINDOW_NORMAL);
        cv::imshow("image",image);
        cv::waitKey(0);
        return 0;
}

cv::Mat Gauss_Equal_Norm(cv::Mat image){
    cv::GaussianBlur(image,image,cv::Size(15,15),sqrt(2.0));
    image=mynorm(image);
    cv::equalizeHist(image,image);
    image.convertTo(image,CV_8UC1,1,0);
    //cv::normalize(xm1,xm,0,255,cv::NORM_MINMAX);normalize
    //xm.convertTo(xm,CV_8U,a*,+b);another way to adjust contrast and brightness,+b:shift)
    return image;
}

cv::Mat mrcOrigin(FILE *f){

//    FILE *f;
    int head_col;
    int head_row;
//    float ele;
    struct _mrchead header;
//    f=fopen("/home/pcchange/swap/1.mrc","rb");
//    std::fstream f("/home/pcchange/swap/1.mrc",std::ios_base::binary);
    header=MrcProcess::readhead(f);
    head_col=header.nx;
    head_row=header.ny;
    /*printf("\n");
    printf("%d\n",header.nx);
    printf("%d\n",header.ny);
    printf("%d\n",header.nz);
    printf("%f\n",header.mode);*/
    float *datap=new float[head_col*head_row];
//    float *datap2=new float[head_col*head_row];
    MrcProcess::readmrcdata(f,datap,header);
    cv::Mat xm_origin=cv::Mat(head_row,head_col,CV_32FC1,datap);
    cv::Mat xm;
    xm=MrcProcess::Gauss_Equal_Norm(xm_origin);
    //std::cout<<"type after mrcorigin"<<xm_origin.type()<<std::endl;
    delete datap;
    /*
    Histogram1D  h;
    cv::Mat histo=h.getHistogram(xm);
    cv::namedWindow("Histogram",CV_WINDOW_NORMAL);
    cv::imshow("Histogram",h.getHistogramImage(xm));
    cv::waitKey(0);
*/
    //for ( int i=0;i<256;i++)
      //  std::cout << "value " << i << " = " << histo.at<float>(i)  << std::endl;
//logo
    //cv::Mat logo=cv::Mat(300,300,CV_8U,cv::Scalar(255));
    //cv::Mat imageROI=xm(cv::Range(xm.rows-logo.rows,xm.rows),
      //                  cv::Range(xm.cols-logo.cols,xm.cols));
    //logo.copyTo(imageROI);

    //cv::Mat stretched=h.stretch(xm,0.01f);
    /*Histogram1D h2;
    cv::namedWindow("Histogram1",CV_WINDOW_NORMAL);
    cv::imshow("Histogram1",h2.getHistogramImage(xm));
    cv::waitKey(0);
    */
    //MrcProcess::showimagecpp(xm);
    //lete datap;
    //fclose(f);
    //printf("ok");
    return xm;

}
}
