#include <iostream>
#include "mrcparser.h"
#include "mnistload.h"
#include "mrcprocess.h"
#include <sstream>
#include <map>
#include <utility>
#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include <unistd.h>
using namespace std;

//namespace pickerCNN {

struct coordinate{
            int x;
            int y;
        };
        typedef map<string,vector<coordinate>> fileNameToCoodinateList;
        //typedef multimap<string,vector<vector<coordinate>>> fileNameToCoodinate;

        fileNameToCoodinateList CNNpicker(ifstream &file){
            struct mrcStarHead thisfileHead;
            struct mrcStarData thisfileData;
            fileNameToCoodinateList fNToCL;
            //fileNameToCoodinate fN;
            thisfileHead=mrcStarParser::parserMrcsHead(file);
            thisfileData=mrcStarParser::mrcStarDataRead(file,thisfileHead);
            cout<<"in CNNpicker"<<endl;
            cout<<thisfileData.MicrographNameList.size()<<endl;
//	    fileNameToCoodinateList fNTCL;
            for (string x:thisfileData.MicrographNameList){
                    vector<coordinate> tmpvector;

                //cout<<"see x"<<endl;
                //cout<<x<<endl;
                    for(vector<string> y: thisfileData.dataDocker){

                        if (y[thisfileHead.positionOfMicrographName]==x){
                            struct coordinate tmptup;
                            stringstream tmpx;
                            stringstream tmpy;
                            tmpx<<y[int(thisfileHead.postionOfCoordinateX)];
                            tmpx>>tmptup.x;
                            tmpy<<y[int(thisfileHead.postionOfCoordinateY)];
                            tmpy>>tmptup.y;
                            tmpvector.push_back(tmptup);
                }
            }
         fNToCL.insert(pair<string,vector<coordinate>>(x,tmpvector));

         tmpvector.clear();
        }
//            int countfile=0;
//            int countline=0;
//            auto map_it=fNToCL.cbegin();
//            while (map_it!=fNToCL.cend()){
//                   cout<< map_it->first << endl;
//                   for (auto m:(map_it->second)){
//                       cout << m.x  <<"\t"<< m.y <<endl;
//                       countline=countline+1;
//                   }
//                   ++map_it;
//                   countfile=countfile+1;
//            }
//           cout<<"countfile\t"<<countfile<<"\tcountline\t"<<countline<<endl;
/*
            for (auto t:thisfileData.MicrographNameList){
            cout<<t<<endl;
            auto map_it = fNToCL.cbegin();
            while (map_it!=fNToCL.cend()){
                cout<<map_it->first <<endl;
                for (auto m:map_it->second){
                    cout<<m<<endl;
                }
                ++map_it;

            }

        }*/
            return fNToCL;
        }
typedef vector<cv::Mat> imagebuffer;
        int calMnistItem(fileNameToCoodinateList intable){
            int countline=0;
            auto map_it=intable.cbegin();
            while (map_it!=intable.cend()){
                    //for (auto m:(map_it->second)){
                        countline=countline+(map_it->second).size();

                    ++map_it;
            }
            return countline;
        }

/*
int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    return 0;
}
*/
//        void readInbuffer();
        void bufferWrite(fileNameToCoodinateList intable,FILE *mnistfile,FILE* labellog,int sub_size){
            //set the buffer size,19600*100
                char sBuf[1960000];
                //char *sBuf=new char[19600000];
                int ck=1;
                setvbuf(mnistfile,sBuf,_IOFBF,1960000);
                auto map_it = intable.cbegin();
                while (map_it!=intable.cend()){
                     FILE *tmpf;
                     const char *tmpmrcfile=(map_it->first).c_str();
                     tmpf=fopen(tmpmrcfile,"r");
                     cv::Mat tmpmrc=MrcProcess::mrcOrigin(tmpf);
                      //not processed

                    for (auto m:(map_it->second)){
                            //not processed
//                            MnistLoad::getWriteContinuousImageBlock(tmpmrc,m.x,m.y,mnistfile)

                            fprintf(labellog,"%s\t%d\t%d\t%d\n",tmpmrcfile,m.x,m.y,ck);//starthere,9.21
                            MnistLoad::getImageBlock(tmpmrc,m.x,m.y,sub_size,mnistfile);
//                            cv::Mat_ <uchar> imageuc=childImage;
//                            cv::Mat_ <uchar>::iterator itcImage=imageuc.begin();
//                            cv::Mat_ <uchar>::iterator itend =imageuc.end();
//                            for (;itcImage!=itend;++itcImage){
//                                fwrite(&(*itcImage),1,1,mnistfile);
//                            }
                            //cout<<ftell(mnistfile)<<endl;
                            //cout<<tmpmrcfile<<"\t"<<m.x<<"\t"<<m.y<<"\t"<<ck<<endl;
                            ck=ck+1;
                            //if (ck>6) exit;
                    }
                    ++map_it;
                    fclose(tmpf);
                    //cout<<"ck"<<"\t"<<ck<<endl;
                }

        }

/*int main(){
    const char* pmn="/home/pcchange/swap/testtmpmnist";
    FILE *mni=fopen(pmn,"wb+");
    int pic_size=140;
    int numItem=1;
    cout<<"here"<<endl;
    MnistLoad::createMnistHead(mni,pic_size,numItem);
      cout<<"here2"<<endl;
    FILE* tmpf=fopen("/home/pcchange/1.mrc","rb");
      cout<<"here3"<<endl;
    cv::Mat tmpmrc=MrcProcess::mrcOrigin(tmpf);
      cout<<"here4"<<endl;
    cout<<tmpmrc.cols<<endl;
    cout<<"before get Imageblock"<<endl;
   MnistLoad::getImageBlock(tmpmrc,90,90,pic_size,mni);
   fclose(mni);
   //
   FILE* readf=fopen(pmn,"rb");
   cout<<"before readMnistHead"<<endl;
   struct _mnistHead h=MnistLoad::readMnistHead(readf);
    cout<<h.msb<<"\t"<<h.items<<"\t"<<h.rows<<"\t"<<h.cols<<endl;
   fseek(readf,16,0);
   unsigned char litm[19600];
   fread(&litm,sizeof(litm),1,readf);
   for (int i=0;i<19600;i++){
       litm[i]=255-litm[i];
   }
   cv::Mat A(140,140,CV_8U,litm);
   cv::namedWindow("Histogram",CV_WINDOW_NORMAL);
   cv::imshow("Histogram",A);
   cv::waitKey(0);
   fclose(readf);
   return 0;
}*/


/*
int main(){
    int starttime=clock();
    string str("/home/pcchange/md_em/particles4class2d.star");
    ifstream f("/home/pcchange/md_em/particles4class2d.star");
    //const char *pc=str.c_str();
    //FILE *fpc=fopen(pc,"r");
    int pic_size=140;
    string mnistfile("/home/pcchange/swap/testtmpmnist");
    const char *pmn=mnistfile.c_str();
    FILE *mni=fopen(pmn,"wb");
    string strlog("/home/pcchange/swap/testmnistlog");
    const char *log=strlog.c_str();
    FILE *mnistlog=fopen(log,"w");
    string label("/home/pcchange/swap/mnistlabel");
    FILE *mnistlabel=fopen(label.c_str(),"wb");
    fileNameToCoodinateList table=CNNpicker(f);
    int numItem=calMnistItem(table);
    cout<<"numofitem"<<"\t"<<numItem<<endl;
    MnistLoad::createMnistHead(mni,pic_size,numItem);
    MnistLoad::createMnistLabelHead(mnistlabel,numItem);
    unsigned char label_write=0;
    fwrite(&label_write,1,numItem,mnistlabel);
    //cout<<ftell(mni)<<endl;
    if (ftell(mni)==16){
    bufferWrite(table,mni,mnistlog,pic_size);
    }
    int endtime=clock();
    cout<<"time"<<"\t"<<endtime-starttime<<endl;
    return 0;
}
*/

int main(int argc, char **argv){

    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */


    char* mainMrcStarFileName;
    char* mainOutMnistDataFileName;
    char* mainLogFileName;
    char* mainMnistLableFileName;
    int pic_size;

    while((oc = getopt(argc, argv, "i:o:g:l:s:")) != -1)
    {
        switch(oc)
        {
            case 'i':
                mainMrcStarFileName=optarg;
                break;
            case 'o':
                mainOutMnistDataFileName=optarg;
                break;
            case 'g':
                mainLogFileName= optarg;
                break;
            case 'l':
                mainMnistLableFileName=optarg;
                break;
            case 's':
                pic_size=atoi(optarg);
             break;
        }
    }
    ifstream mainMrcStarFile(mainMrcStarFileName);
    FILE *mainOutMnistDataFile;
    FILE *mainLogFile;
    FILE *mainMnistLableFile;
    //mainMrcStarFile=fopen(mainMrcStarFileName,"r");
    mainOutMnistDataFile=fopen(mainOutMnistDataFileName,"wb");
    mainLogFile=fopen(mainLogFileName,"w");
    mainMnistLableFile=fopen(mainMnistLableFileName,"wb");

    fileNameToCoodinateList table=CNNpicker(mainMrcStarFile);
    int numItem=calMnistItem(table);
    cout<<"numofitem"<<"\t"<<numItem<<endl;
    MnistLoad::createMnistHead(mainOutMnistDataFile,pic_size,numItem);
    MnistLoad::createMnistLabelHead(mainMnistLableFile,numItem);
    unsigned char label_write=0;
    fwrite(&label_write,1,numItem,mainMnistLableFile);
    //cout<<ftell(mni)<<endl;
    if (ftell(mainOutMnistDataFile)==16){
    bufferWrite(table,mainOutMnistDataFile,mainLogFile,pic_size);
    }
    fclose(mainOutMnistDataFile);
    fclose(mainLogFile);
    fclose(mainMnistLableFile);
   return 0;

}
