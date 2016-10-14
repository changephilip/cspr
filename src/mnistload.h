#ifndef MNISTLOAD_H
#define MNISTLOAD_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv/highgui.h>
#define mrcsize 1024

struct _mnistHead{
    int msb;
    int items;
    int rows;
    int cols;
};
struct _mnistLabelHead{
    int msb;
    int items;
};

/*
class MnistLoad
{

public:
    //MnistLoad();
    int ReverseInt(int i);
    struct _mnistHead readMnistHead(FILE* f);
    //struct _mrchead readhead(FILE *f);
    void create_CNN_MnistHead(cv::Mat &image,int step,FILE *f);
    void create_CNN_MnistData(cv::Mat &image,int step,int sub_size,FILE *f);
};
*/

namespace MnistLoad {

    int ReverseInt(int i);
    struct _mnistHead readMnistHead(FILE* f);
    //struct _mrchead readhead(FILE *f);
    void create_CNN_MnistHead(cv::Mat &image,int step,FILE *f);
    void create_CNN_MnistData(cv::Mat &image, int step, int sub_size, FILE *f);
    //void getWriteContinuousImageBlock(cv::Mat &image, int pc, int pr, int sub_size, FILE *f);
    void getImageBlock(const cv::Mat &image,int pc,int pr,int sub_size,FILE *f);
    void createMnistHead(FILE *f,int picsize,int num);
    struct _mnistLabelHead readMnistLabelHead(FILE* f);
    void createMnistLabelHead(FILE*f,int num);
}
#endif // MNISTLOAD_H
