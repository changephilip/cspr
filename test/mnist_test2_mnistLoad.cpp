#include "mnistload.h"
#include <math.h>
//using namespace mnistLoad::;
using namespace std;
int main()
{
    char mnist_path[]="/home/pcchange/swap/train-images.idx3-ubyte";
    string aa="sdhjfkalshfuiw";
    const char* bb=aa.c_str();
    printf("bb %s\n",bb);
    FILE *f=fopen(mnist_path,"rb");
     struct _mnistHead  header;
     header=MnistLoad::readMnistHead(f);
    printf("head.msb %d\n",header.msb);
    printf("head.items %d\n",header.items);
    printf("head.rows %d\n",header.rows);
    printf("head.cols %d\n",header.cols);
    //unsigned char a;
    //read a 28x28 mat from f;
    fseek(f,16,0);
    /*
    unsigned char *litm=new unsigned char[28*28];
    for (int i=0;i<784;++i){
        unsigned char tmp;
        fread(&tmp,1,1,f);
        litm[i]=255-tmp;
    }
    */
    unsigned char litm[784];
    fread(&litm,sizeof(litm),1,f);
    for (int i=0;i<784;++i){
        litm[i]=255-litm[i];
}
    cv::Mat A(28,28,CV_8U,litm);
    //cv::Mat A(28,28,CV_8U,litm);
    /*printf("size struct %ld\n",sizeof(unsigned char));
    fseek(f,20,0);
    fread(&a,1,1,f);
    //a=ReverseInt(a);
    printf("a char %c\n",a);
    unsigned int b;
    b=(unsigned int)a;
    printf("b %d\n",b);*/
    cv::namedWindow("Histogram",CV_WINDOW_NORMAL);
    cv::imshow("Histogram",A);
    cv::waitKey(0);
    FILE *fnew=fopen("/home/pcchange/swap/mnnew.idx3-ubyte","ab+");
    MnistLoad::create_CNN_MnistHead(A,4,fnew);
    fseek(fnew,0,0);
    struct _mnistHead newhead;
    newhead=MnistLoad::readMnistHead(fnew);
    printf("head.msb %d\n",newhead.msb);
    printf("head.items %d\n",newhead.items);
    printf("head.rows %d\n",newhead.rows);
    printf("head.cols %d\n",newhead.cols);
    fseek(fnew,16,0);
    MnistLoad::create_CNN_MnistData(A,8,8,fnew);
    //delete litm;
    fclose(fnew);
    fclose(f);
/*    double x=log(2);
    cout<<x<<endl;
    return 0;*/
}
