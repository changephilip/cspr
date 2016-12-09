#include "cml_t.cpp"
#include "mrcparser.h"
#include "PickerCNN.cpp"
//need link mpi
void cml_read(float *data,fileNameToCoodinateList intable,int cml_size){
    auto map_it = intable.cbegin();
    int ck=0;
    while (map_it != intable.cend()){
        FILE *tmpf;
        const char *tmpmrcfile = (map_it->first).c_str();
        tmpf = fopen(tmprcfile,"r");
        struct _mrchead hf;
        hf = MrcProcess::readhead(tmpf);
        float mrc[cml_size*cml_size];
//        mrc = MrcProcess::mrcOrigin(tmpf);//it has Gauss_Equal_Norm;
        mrc = MrcProcess::readmrcdata(tmpf,hf);
        for (auto m:(map_it -> second)){
            getsubmrc(mrc,m.x,m.y,data[ck*cml_size*cml_size],hf);
            ck++;
        }
    ++map_it;
    fclose(tmpf);
    }
}

void cml_dftread(float *data,fileNameToCoodinateList intable,int cml_size){
    auto map_it = intable.cbegin();
    int ck=0;
    while (map_it != intable.cend()){
        FILE *tmpf;
        const char *tmpmrcfile = (map_it->first).c_str();
        tmpf = fopen(tmprcfile,"r");
        struct _mrchead hf;
        hf = MrcProcess::readhead(tmpf);
        float mrc[cml_size*cml_size];
//        mrc = MrcProcess::mrcOrigin(tmpf);//it has Gauss_Equal_Norm;
        mrc = MrcProcess::readmrcdata(tmpf,hf);
        for (auto m:(map_it -> second)){
            float tmps[cml_size*cml_size];
            getsubmrc(mrc,m.x,m.y,tmps,data,hf);
            Mat image_tmp= Mat(cml_size,cml_size,CV_32FC1,tmps);
            Mat afdft_img=imdft(image_tmp);
            Mat ldft_img(afdft_img.size(),afdft_img.type());
            linearpolar(afdft_img,ldft_img);
            image_to_mat(ldft_img,data[ck*cml_size*cml_size],cml_size);
            ck++;
        }
    ++map_it;
    fclose(tmpf);
    }
}
int main()
{
    int N,cml_size;
    float *cml_matrix=new float[N*cml_size*cml_size];
    int dft_size=getOptimalDFTSize(cml_size);
    float *lineardft_matrix=new float[N*dft_szie*dft_size];

    ifstream mainMrcStarFile("mrcstar");

    fileNameToCoodinateList table=CNNpicker(mainMrcStarFile);
    int numItem = calMnistItem(table);
    cml_read(cml_matrix,table,cml_size);
    cml_dftread(lineardft_matrix,table,dft_size);
    #pragma omp parallel
    {

    }
}
