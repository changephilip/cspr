#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include<sstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
//#include <regex>

using namespace std;
using namespace boost::algorithm;
void elimDups(vector<string> &words)
{
    sort(words.begin(),words.end());
    auto end_unique=unique(words.begin(),words.end());
    words.erase(end_unique,words.end());
}

int main()
{
    fstream file("/home/pcchange/md_em/particles4class2d.star");
    vector<string> splitvec;
    vector<string> mrcsHead;
    //map<int,string> nMrcsHead;
    //vector<int> mrcsHeadNum;
    string line;
    int i=0;
    int num_head=0;
    while(getline(file,line) and i<=30){
        if (line.size()>0){
        //istringstream sline(line);
        split(splitvec,line,is_any_of(" "),token_compress_on);
        if (splitvec[0].string::find("_")==0){
            num_head=num_head+1;
            mrcsHead.push_back(splitvec[0]);
            //mrcsHeadNum.push_back(int(splitvec[1][1]);
            }
        }
        i=i+1;
    }
    if (!mrcsHead.empty()){
        for (auto x: mrcsHead)
            cout<<x<<endl;
    }
    vector<string>::iterator iter=std::find(mrcsHead.begin(),mrcsHead.end(),"_rlnMicrographName");
    if (iter!=mrcsHead.end())
        cout<<iter-mrcsHead.begin()<<endl;
    int positionofFile=iter-mrcsHead.begin();
    //cout<<dist<<endl;
    vector<string>::iterator iter2=std::find(mrcsHead.begin(),mrcsHead.end(),"_rlnCoordinateX");
    if (iter!=mrcsHead.end())
        cout<<iter2-mrcsHead.begin()<<endl;
    int positionofX=iter2-mrcsHead.begin();
    vector<string>::iterator iter3=std::find(mrcsHead.begin(),mrcsHead.end(),"_rlnCoordinateY");
    if (iter!=mrcsHead.end())
        cout<<iter3-mrcsHead.begin()<<endl;
    int positionofY=iter3-mrcsHead.begin();
    cout<<positionofY<<endl;
    file.seekg(0,ios::beg);
    vector<string> dataline;
    vector<vector<string>> datadocker;
    vector<string> fileNameDocker;
    i=0;
    while(getline(file,line)){
        if (line.size()>0 and i>num_head+3){
            split(dataline,line,is_any_of("\t"),token_compress_on);
            //for (auto x: dataline)
               // cout<<x<<endl;
            datadocker.push_back(dataline);
            fileNameDocker.push_back(dataline[0]);
        }
        i=i+1;
        //if (i>50)
           // break;
    }
    cout<<datadocker.size()<<endl;
    cout<<datadocker[0][0]<<endl;
    /*string str1("hello abc ABC aBc goodbye");
    vector<string> SplitVec; // #2: Search for tokens
    split(SplitVec, str1, is_any_of(" "), token_compress_on); // SplitVec == { "hello abc","ABC","aBc goodbye" }
    cout<<SplitVec[0]<<endl;*/
    elimDups(fileNameDocker);
    cout<<fileNameDocker.size()<<endl;
    for (auto x:fileNameDocker){
        cout<<x<<endl;
    }
    return 0;
}
