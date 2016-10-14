#include "mrcprocess.h"
//using namespace MrcProcess;
int main(){
    FILE *f;
    int head_col;
    int head_row;
//    float ele;
    struct _mrchead header;
    f=fopen("/home/pcchange/swap/1.mrc","rb");
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
    MrcProcess::showimagecpp(xm);
    //lete datap;
    fclose(f);
    printf("ok");
    return 0;
}
