#include "mrcparser.h"
using namespace std;
using namespace boost::algorithm ;
/*
MrcParser::MrcParser()
{
}
*/
namespace mrcStarParser{
/*
    class coreMrcsObject{
    public:
        string _rlnMicrographName;
        float   _rlnCoordinateX;
        float   _rlnCoordinateX;
        float   _rlnCoordinateY;
        string  _rlnImageName;
        float   _rlnDefocusU;
        float   _rlnDefocusV;
        float   _rlnVoltage;
        float   _rlnSpericalAberration;
        float   _rlnAmplitudeContrast;
        float   _rlnMagnification;
        float   _rlnDetectorPixelSize;
        float   _rlnCtFigure0fMerit;
        float   _rlnNormCorrection;

    };
*/

 struct mrcStarHead parserMrcsHead(ifstream &file){
     vector<string> splitVec;
     struct mrcStarHead head;
     //char *cline;
     string line;
     int i=0;
     int num_head=0;
     file.seekg(0,ios::beg);
     //fseek(file,0,0);
     //while (fgets(cline,1024,file) and i<=100){
     while(getline(file,line) and i<=100){
         //line=cline;
         if(line.size()>0){
             split(splitVec,line,is_any_of(" "),token_compress_on);
             if(splitVec[0].string::find("_")==0){
                 num_head=num_head+1;
                 head.mrcsHead.push_back(splitVec[0]);
             }
         }
         i=i+1;
     }
    vector<string>::iterator iter;
    //cout<<"in the function mrcparser.cpp"<<endl;
    iter=find(head.mrcsHead.begin(),head.mrcsHead.end(),"_rlnMicrographName");
    if (iter!=head.mrcsHead.end()){
        head.positionOfMicrographName=iter-head.mrcsHead.begin();
        //cout<<iter-head.mrcsHead.begin()<<endl;
        //cout<<"in the function mrcparser.cpp"<<endl;
        //cout<<head.positionOfMicrographName<<endl;
    }
    iter=find(head.mrcsHead.begin(),head.mrcsHead.end(),"_rlnCoordinateX");
    if (iter!=head.mrcsHead.end())
    {head.postionOfCoordinateX=iter-head.mrcsHead.begin();}
    iter=find(head.mrcsHead.begin(),head.mrcsHead.end(),"_rlnCoordinateY");
    if (iter!=head.mrcsHead.end())
    {head.postionOfCoordinateY =iter-head.mrcsHead.begin();}
    //cout<<head.positionOfMicrographName<<endl;
    //cout<<head.postionOfCoordinateX<<endl;
    //cout<<head.postionOfCoordinateY<<endl;
    //cout<<"out of the function"<<endl;
    return head;
 }

 void elimDups(vector<string> &words)
 {
     sort(words.begin(),words.end());
     auto end_unique=unique(words.begin(),words.end());
     words.erase(end_unique,words.end());
 }

 struct mrcStarData mrcStarDataRead(ifstream &file,struct mrcStarHead head){

     //fseek(file,0,0);
     file.seekg(0,ios::beg);
     struct mrcStarData data;
     vector<string> dataline;
     string line;
     //char *cline;
     int i=0;
     //while(fgets(cline,1024,file)){
     int num=head.mrcsHead.size()+3;
     while(getline(file,line)){
        //line=cline;


         if(line.size()>0 and i>num){
             split(dataline,line,is_any_of("\t"),token_compress_on);
             data.dataDocker.push_back(dataline);
             data.MicrographNameList.push_back(dataline[0]);
            //cout<<dataline[0]<<endl;
         }
        i=i+1;
     }
    elimDups(data.MicrographNameList);
    //for (auto x: data.MicrographNameList){
    //    cout<<x<<endl;
    //}
    return data;
 }



}
