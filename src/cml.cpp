#include "cml.h"

namespace CML{

void getsubmrc(float *p,int x,int y,int CML_SIZE,float *s,const struct _mrchead header)
{
    int nx,ny;
    //if the coordinate x and y from 0 to max,not from 1 to max;
    if (x>header.nx-CML_SIZE-1){
        nx=header.nx-CML_SIZE-1;
    }else {
        nx=x;
    }
    if (y>header.ny-CML_SIZE-1){
        ny=header.ny-CML_SIZE-1;
    }else {
        ny=y;
    }
//    printf("\ninside test p[0] %f\n",p[0]);
//    if (x<=header.nx-CML_SIZE-1 and y<=header.ny-CML_SIZE-1)
//    {
    int i=0;
    int j=0;
    int k=0;
    for (j=0;j<CML_SIZE;j++){
        for (i=0;i<CML_SIZE;i++){
            s[k]=p[header.nx*(ny+j)+nx+i];
            k=k+1;
            }
    }
}


cv::Mat imdft(cv::Mat &I)
{

    if ( I.empty() )
        //return -1;
        printf("no image error");
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(I.rows);
        int n = cv::getOptimalDFTSize(I.cols);
        cv::copyMakeBorder(I,padded,0,m- I.rows,0,n-I.cols,cv::BORDER_CONSTANT,cv::Scalar::all(0));

        cv::Mat planes[] = {cv::Mat_<float>(padded),cv::Mat::zeros(padded.size(),CV_32F)};
        cv::Mat complexI;
        cv::merge(planes,2,complexI);

        cv::dft(complexI,complexI);

        cv::split(complexI,planes);
        cv::magnitude(planes[0],planes[1],planes[0]);
        cv::Mat magI = planes[0];

        magI += cv::Scalar::all(1);
        log(magI,magI);

        magI= magI(cv::Rect(0,0,magI.cols & -2 ,magI.rows & -2));
        int cx = magI.cols/2;
        int cy = magI.rows/2;

        cv::Mat q0(magI,cv::Rect(0,0,cx,cy));
        cv::Mat q1(magI,cv::Rect(cx,0,cx,cy));
        cv::Mat q2(magI,cv::Rect(0,cy,cx,cy));
        cv::Mat q3(magI,cv::Rect(cx,cy,cx,cy));

        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        cv::normalize(magI,magI,0,1,CV_MINMAX);

        //imshow("Input Image",I);
        //imshow("spectrum magnitude",magI);
        //MrcProcess::showimagecpp(magI);
//        printf("\nmagI.cols %d magI.rows %d\n",magI.cols,magI.rows);
//        if (magI.isContinuous()){
//            printf("isContinuous\n");
//        }
//        else printf("not isContinuous\n");
        //magI.copyTo(I);
        return magI;
}


float NCC0(float *cml1,float *cml2,int CML_SIZE){
    //cml1,cml2,length=CML_SIZE,one-dem array
    //mpi
    float ncc;
    float sigma1;
    float sigma2;
    float mean1;
    float mean2;
    float sum1 = 0.0;
    float sum2 = 0.0;
    int i;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
//    mean1 = sum1 / CML_SIZE;
//    mean2 = sum2 / CML_SIZE;
    cv::Mat a,b;
    a = cv::Mat(1,CML_SIZE,CV_32FC1,cml1);
    b = cv::Mat(1,CML_SIZE,CV_32FC1,cml2);
    cv::Mat tmp_m1,tmp_m2,tmp_sig1,tmp_sig2;
    cv::meanStdDev(a,tmp_m1,tmp_sig1);
    cv::meanStdDev(b,tmp_m2,tmp_sig2);
    mean1 = tmp_m1.at<double>(0,0);
    mean2 = tmp_m2.at<double>(0,0);
    sigma1 = tmp_sig1.at<double>(0,0);
    sigma2 = tmp_sig2.at<double>(0,0);
    float coeff;
    coeff = 1/(float(CML_SIZE)*sigma1*sigma2);
//    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
    float ncc_fft=0.0;//ncc_0
    float ncc_2;
    float ncc_3;
    float ncc_4;
    ncc_2 = CML_SIZE*mean1*mean2;
    ncc_3 = mean1*sum2;
    ncc_4 = mean2*sum1;
//    float sum_ncc_fft=0.0;


    for(i=0;i<CML_SIZE;i++){
        ncc_fft=ncc_fft+cml1[i]*cml2[i];

    }

//    for (i=0;i<CML_SIZE;i++){
//        sum_ncc_fft=sum_ncc_fft+ncc_fft[i] ;
//    }
    ncc=coeff*(ncc_fft+ncc_2-ncc_3-ncc_4);
    //printf("\n in NCC0\n");
//    printf("\t%f\t",ncc);
    return ncc;
}


float FNCC(float *cml1,float *cml2,int CML_SIZE){
    //cml1,cml2,length=CML_SIZE,one-dem array
    //mpi
    float ncc;
    float mean1;
    float mean2;
    float sum1 = 0.0;
    float sum2 = 0.0;

    int i;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
    int Ncc=0;
    mean1 = sum1 / CML_SIZE;
    mean2 = sum2 / CML_SIZE;
    for (i=0;i<CML_SIZE;i++){
        if (cml1[i]>=mean1) {
            if (cml2[i]>=mean2){
                Ncc+=1;
            }
            else Ncc+=-1;
        }
        else if (cml2[i]>=mean2){
            Ncc+=-1;
        }
        else Ncc+=1;
    }
    ncc=(float)Ncc/CML_SIZE;
    return ncc;
}


float cal_angle(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size){
    double a,b,c;
//    double two_pi=6.28318530;
    float cos_angleij,angleij;
    a = cos((cmlkj-cmlki)*M_2_PI/float(after_dft_size));
    b = cos((cmljk-cmlji)*M_2_PI/float(after_dft_size));
    c = cos((cmlik-cmlij)*M_2_PI/float(after_dft_size));
    cos_angleij = (a-b*c)/(sqrt(1-b*b)*sqrt(1-c*c));
//    printf("\ncos_angle %f\n",cos_angleij);
    angleij = acos(cos_angleij);
    return angleij*180.0/M_PI;
}


bool voting_condition(int cmlij,int cmlik,int cmlji,int cmljk,int cmlki,int cmlkj,int after_dft_size){
    double a,b,c;
//    double two_pi=6.28318530;
//    float cos_angleij,angleij;
    a = cos((cmlkj-cmlki)*M_2_PI/float(after_dft_size));
    b = cos((cmljk-cmlji)*M_2_PI/float(after_dft_size));
    c = cos((cmlik-cmlij)*M_2_PI/float(after_dft_size));
    if ((1+2*a*b*c)>a*a+b*b+c*c) {
        return TRUE;
    }
        else
    {return FALSE;}
}


cml_tuple NCC_value(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cml_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
#pragma omp parallel for

        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
            value[i][j] = FNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
//            printf("\n000003");
        }

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

cml_tuple NCC_valuet(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    cml_tuple ret;
    int i,j;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
//    float *p1;
//    float *p2;
    //mpi here
    printf("\n");
    for(i=0;i<after_dft_size;i++){
//        printf("\n000001");
#pragma omp parallel for

        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
//            printf("\n0000002");
            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
            float fncc_v= FNCC(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
            printf(",%f",(fabs((value[i][j]-fncc_v)/fncc_v)));
//            printf("\n000003");
        }
        printf("\n");

    }
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            printf("\t%f\t",value[i][j]);
            if (value[i][j]>value_ini) {
                value_ini = value[i][j];
                ret.x=i;
                ret.y=j;
            }
//            else break;
        }
    }
//    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

void linearpolar(cv::Mat &I,cv::Mat &dst){
    //may release problem here
//    Mat dst(I.size(),I.type());
    IplImage ipl_afdft=I;
    IplImage ipl_dst=dst;
    cvLinearPolar( &ipl_afdft, &ipl_dst, cvPoint2D32f(I.cols/2,I.rows/2),I.cols/2,CV_INTER_CUBIC);
//    return dst;
}

void image_to_mat(cv::Mat &I,float *matp,int after_dft_size){
    int i,j,k;
    k=0;
    cv::Mat_<float> dstf=I;
    cv::Mat_<float>::iterator it = dstf.begin();
    cv::Mat_<float>::iterator itend = dstf.end();
        while (it!=itend){
            for (i=0;i<after_dft_size;i++){
                for (j=0;j<after_dft_size;j++){
                    matp[k]=*it;
                    ++it;
                    k++;
                }
            }
        }
}

void cml_read(float *data,fileNameToCoodinateList intable,int cml_size){
    auto map_it = intable.cbegin();
    int ck=0;
    while (map_it != intable.cend()){
        FILE *tmpf;
        const char *tmpmrcfile = (map_it->first).c_str();
        tmpf = fopen(tmpmrcfile,"r");
        struct _mrchead hf;
        hf = MrcProcess::readhead(tmpf);
        float mrc[cml_size*cml_size];
//        mrc = MrcProcess::mrcOrigin(tmpf);//it has Gauss_Equal_Norm;
        MrcProcess::readmrcdata(tmpf,mrc,hf);
        for (auto m:(map_it -> second)){
            CML::getsubmrc(mrc,m.x,m.y,cml_size,&data[ck*cml_size*cml_size],hf);
            ck++;
        }
    ++map_it;
    fclose(tmpf);
    }
}

void cml_dftread(float *data,fileNameToCoodinateList intable,int cml_size,int dft_size,int N){
    auto map_it = intable.cbegin();
    int ck=0;
//    printf("\n00001\n");
    while (map_it != intable.cend() and ck<N){
        FILE *tmpf;
        const char *tmpmrcfile = (map_it->first).c_str();
//        printf("\n00002\n");
        tmpf = fopen(tmpmrcfile,"r");
//        printf("\n00003\n");
        struct _mrchead hf;
        hf = MrcProcess::readhead(tmpf);
//        printf("\n00004\n");
//        printf("\nnx ny\t%d\t%d\n",hf.nx,hf.ny);
        float *mrc=new float[hf.nx*hf.ny];
//        float bigmrc[hf.nx*hf.ny];
//        printf("\n00005\n");
//        mrc = MrcProcess::mrcOrigin(tmpf);//it has Gauss_Equal_Norm;
        MrcProcess::readmrcdata(tmpf,mrc,hf);
//        printf("\n00006\n");

        for (auto m:(map_it -> second)){
            if (ck<N)
            {
            float tmps[cml_size*cml_size];
            getsubmrc(mrc,m.x,m.y,cml_size,tmps,hf);
//            printf("\n00007\n");
            cv::Mat image_tmp= cv::Mat(cml_size,cml_size,CV_32FC1,tmps);
//            printf("\n00008\n");
            cv::Mat afdft_img=imdft(image_tmp);
            cv::Mat ldft_img(afdft_img.size(),afdft_img.type());
            linearpolar(afdft_img,ldft_img);
            image_to_mat(ldft_img,&data[ck*dft_size*dft_size],dft_size);
            ck++;
            }
            else {
                break;
            }
        }
        delete[] mrc;
    ++map_it;
    fclose(tmpf);
    }
}

float max_float(float *infloat,int size_of_array){
    int i;
    float max_return;
    max_return=infloat[0];
    for (i=1;i<size_of_array;i++){
        if (max_return < infloat[i]){
            max_return=infloat[i];
        }
    }
    return max_return;
}

int max_float_index(float *infloat,int size_of_array){
    int i;
    float max_float;
    int max_index_return;
    max_index_return=0;
    max_float=infloat[0];
    for (i=0;i<size_of_array;i++){
        if (max_float < infloat[i]){
            max_float = infloat[i];
            max_index_return=i;
        }
    }
    return max_index_return;
}

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

void writeDisk(float *p,FILE *filename,long filelength){
    char sBuf[1960000];
    setvbuf(filename,sBuf,_IOFBF,196000);
    fwrite(&p,4,filelength,filename);
}
//int readDisk(float *p,FILE *filename)

}
