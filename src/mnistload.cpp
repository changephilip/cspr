#include "mnistload.h"

namespace MnistLoad{
/*MnistLoad::MnistLoad()
{
}
*/

int  ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

struct _mnistHead readMnistHead(FILE* f){
    //ifstream filemnist(filepath,ios::binary);
    struct _mnistHead head;
    if ((NULL==f)){
        printf("cannot open file\n");}
    fread(&head,sizeof(struct  _mnistHead),1,f);
    head.msb=ReverseInt(head.msb);
    //head.msb=ReverseInt(head.msb);
    //head.msb=ReverseInt(head.msb);
    head.items=ReverseInt(head.items);
    head.rows=ReverseInt(head.rows);
    head.cols=ReverseInt(head.cols);

    return head;
};

void create_CNN_MnistHead(cv::Mat &image,int step,FILE *f)
{
    int cols=image.cols;
    int rows=image.rows;
    int  MSB=2051;
    int numItems;
    numItems=(cols/step)*(rows/step);
    MSB=ReverseInt(MSB);
    numItems=ReverseInt(numItems);
    rows=ReverseInt(rows);
    cols=ReverseInt(cols);
    fwrite(&MSB,4,1,f);
    fwrite(&numItems,4,1,f);
    fwrite(&rows,4,1,f);
    fwrite(&cols,4,1,f);

}
void createMnistHead(FILE *f,int picsize,int num){
    int MSB=2051;
    fseek(f,0,SEEK_SET);
    MSB=ReverseInt(MSB);
    picsize=ReverseInt(picsize);
    num=ReverseInt(num);
    fwrite(&MSB,4,1,f);
    fwrite(&num,4,1,f);
    fwrite(&picsize,4,1,f);
    fwrite(&picsize,4,1,f);//write the rows and the cols,normally they are the same
}

void create_CNN_MnistData(cv::Mat &image,int step,int sub_size,FILE *f)
{
    //fseek(f,16,0);
    int cols=image.cols;
    int rows=image.rows;
//    int sub_cols=0;
 //   int sub_rows=0;
//    int ncols=cols/sub_size;
 //   int nrows=rows/sub_size;
//    int nc=0;
//    int nr=0;
    int pc=0;
    int pr=0;
    //if (sub_size >cols or sub_size >rows)
       // printf("error\n");
/*    for (;nc<=ncols+1;nc=nc+1){
        for (;nr<=nrows+1;nr=nr+1 ){
            if (nc==ncols+1 )
                pc=cols-sub_size-1;
            else  pc=nc*sub_size-1;
            if (nr==nrows+1 )
                pr=rows-sub_size-1;
            else pr=nr*sub_size-1;
            if (pc==-1) pc=0;
            if (pr==-1) pr=0;
            printf("pr %d pc %d\n",pr,pc);
            //start here.
            cv::Mat imageROI(image,cv::Rect(pc,pr,sub_size,sub_size));
            cv::Mat_<uchar> image2=imageROI;
            cv::Mat_<uchar>::iterator it = image2.begin();
            cv::Mat_<uchar>::iterator itend = image2.end();
            for (;it!=itend;++it){
                fwrite(&(*it),1,1,f);
                }
            }
        }*/
    for    (;pr<rows;pr=pr+step){
        for (;pc<cols;pc=pc+step){
            if (pr+sub_size>rows)
                pr=rows-sub_size;
            if (pc+sub_size>cols)
                pc=cols-sub_size;
            cv::Mat imageROI(image,cv::Rect(pc,pr,sub_size,sub_size));
            cv::Mat_<uchar> image2=imageROI;
            cv::Mat_<uchar>::iterator it = image2.begin();
            cv::Mat_<uchar>::iterator itend = image2.end();
            for (;it!=itend;++it){
                fwrite(&(*it),1,1,f);
                }

            }
      }
   }

/*
void getWriteContinuousImageBlock(cv::Mat &image,int pc,int pr,int sub_size,FILE *f){
            //fseek(f,16,0);continous write and don't fseek the flag in the file
            //int cols=image.cols;
            //int rows=image.rows;
            //int pc=0;
            //int pr=0;
            std::cout<<ftell(f)<<std::endl;
            if (pc+sub_size>image.cols){
                pc=image.cols-sub_size;
            }
            if (pr+sub_size>image.rows){
                pr=image.rows-sub_size;
            }
            cv::Mat imageROI(image,cv::Rect(pc,pr,sub_size,sub_size));
            cv::Mat_<uchar> image2=imageROI;
            cv::Mat_<uchar>::iterator it = image2.begin();
            cv::Mat_<uchar>::iterator itend = image2.end();
            for (;it!=itend;++it){
                fwrite(&(*it),1,1,f);
                }

    }
*/


struct _mnistLabelHead readMnistLabelHead(FILE* f){
    struct _mnistLabelHead head;
    if ((NULL==f)){
        printf("cannot open file\n");
    }
    fread(&head,sizeof(struct _mnistLabelHead),1,f);
    head.msb=ReverseInt(head.msb);
    head.items=ReverseInt(head.items);

    return head;
}
void createMnistLabelHead(FILE*f,int num){
    int MSB=2049;
    fseek(f,0,SEEK_SET);
    MSB=ReverseInt(MSB);
    num=ReverseInt(num);
    fwrite(&MSB,4,1,f);
    fwrite(&num,4,1,f);
}

void getImageBlock(const cv::Mat &image,int pc,int pr,int sub_size,FILE *f){
            //fseek(f,16,0);continous write and don't fseek the flag in the file
            //int cols=image.cols;
            //int rows=image.rows;
            //int pc=0;
            //int pr=0;
           // std::cout<<ftell(f)<<std::endl;
            int tpc;
            int tpr;
            if (pc+sub_size>image.cols){
                tpc=image.cols-sub_size;
                //std::cout<<"larger cols"<<std::endl;
            }
            else tpc=pc;
            if (pr+sub_size>image.rows){
                tpr=image.rows-sub_size;
                //std::cout<<"larger rows"<<std::endl;
            }
            else tpr=pr;
           // std::cout<<tpc<<"\t"<<tpr<<std::endl;

            cv::Rect rect(tpc,tpr,sub_size,sub_size);
            //cv::Mat imageROI;
//            cv::Mat imaget;
//            imaget=image(rect);
            cv::Mat imageROI=image(rect);
            //image(rect).copyTo(imageROI);
            //cv::Mat imageROI;
            //image(rect).copyTo(imageROI);
           // std::cout<<"here"<<std::endl;
            //std::cout<<image.type()<<std::endl;
            cv::Mat_<uchar> image2=imageROI;
            cv::Mat_<uchar>::iterator it = image2.begin();
             //std::cout<<"here2"<<std::endl;
            cv::Mat_<uchar>::iterator itend = image2.end();
//            std::cout<<"here3"<<std::endl;
//            int n=imageROI.cols*imageROI.rows;
//            if (imageROI.isContinuous()){
//                std::cout<<"continuous"<,std::endl;
//                imageROI.reshape(1,1);
//            }

//            std::cout<<n<<"\t"<<sizeof(imageROI.data)<<std::endl;
//            fwrite(&imageROI.data,n,1,f);
            int n=sub_size*sub_size;
            uchar blockbuff[n];
            for (int i=0;it!=itend;++it,++i){
                 //std::cout<<"here2"<<std::endl;
                //fwrite(&(*it),sizeof(*it),1,f);
                blockbuff[i]=(*it);
                }
            fwrite(&blockbuff,sizeof(blockbuff),1,f);
            //return imageROI;
//            std::cout<<image.isContinuous()<<std::endl;
//            std::cout<<imageROI.isContinuous()<<std::endl;
//            std::cout<<imageROI.step<<"\t"<<imageROI.cols*imageROI.rows<<std::endl;
//            int n=imageROI.cols*imageROI.rows;
//            uchar dat[n];
//            for (int i=1;i<=n;i++){
//                std::cout<<i<<std::endl;
//                std::cout<<image.at<uchar>(1,1)<<std::endl;
                //dat[i-1]=imageROI.at<char>(i/140+1,i-(1/140+1)*140);
//            int nl=imageROI.rows;
//            int nc=imageROI.cols*imageROI.channels();
//            for (int j=0;j<nl;j++){
//                uchar* data=imageROI.ptr<uchar>(j);
//                for (int i=0;i<nc;i++){
//                    fwrite(&data[i],1,1,f);
//                }
//            }
            //std::cout<<image.at<uchar>(1,1)<<std::endl;
//            cv::Mat_<uchar> image2(image);
//            std::cout<<image2(50,100)<<std::endl;

//            cv::Mat_<uchar>::iterator it = image2.begin();
//            cv::Mat_<uchar>::iterator itend = image2.end();
//            for (;it!=itend;++it){
//               std::cout<<image2.at<uchar>(1,1)<<std::endl;
//            }
            //std::cout<<"arrary finish"<<std::endl;




    }

}
