#include "mrcparser.h"
#include "mnistload.h"
#include "mrcprocess.h"
//#include "gsl/gsl_matrix.h"
#include <iostream>
#include <math.h>
#include <gsl/gsl_fft.h>
//#include <fftw3.h>
#include "fftw3.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <array>
#include <vector>
//#include "dft.cpp"
//#define CML_INTE 2
//#define CML_NUM 90
//int CML_NUM*int CML_LINE=180
//#define CML_SIZE 140
//#define HALF_CML_SIZE 71
using namespace cv;
/*int main()*/
//{
	//FILE *f;
	//int head_col;
	//int head_row;
	//struct _mrchead header;
	//f=fopen("~/1.mrc","rb");
	//header=MrcProcess::readhead(f);
	//head_col=header.nx;
	//head_row=header.ny;
	//float *datap=new float[head_col*head_row];
	//MrcProcess::readmrcdata(f,datap,header);
	//cv::Mat xm_origin=cv::Mat(head_row,head_col,CV_32FC1,datap);
	//cv::Mat xm;
	//xm=MrcProcess::Gauss_Equal_Norm(xm_origin);
	//delete datap;
	//fclose(f);
	//MrcProcess::showimagecpp(xm);
	//printf("OK");
	//return 0;
/*}*/
/*typedef float cml_mat[180/CML_INTE][140];*/
//gsl_matrix calindi(){
	//gsl_matrix *m = gsl_matrix_alloc(60,2);
	//float theta=3.1415926/180.0;
	//for (int i=0;i<60;i++){
		//printf("ok");
		//theta=theta+1.0;
	//}
/*}*/
typedef struct {
		int x;
		int y;
		}tuple;
//typedef float sinogram[CML_NUM][CML_SIZE];
//typedef tuple tmat[CML_NUM][CML_SIZE];
//typedef float mrcMat[CML_SIZE][CML_SIZE];
//void cml_help_max(tmat chm){
//	/*this function should be used to create line of matrix*/
//	/*according to the matrix's value,we calculate the line of mrc matrix*/
//	/*defaultly,we will have 60 lines in one projection,the size of line is CML_SIZE*/
//	/*struct tuple matrix can't be returned*/
//	/*so the code should be insert beside other code*/
//	//struct tuple m[60][CML_SIZE];
//	float theta=3.1415926/float(CML_NUM);
//	printf("theta %f",theta);
//	int i=0;
//	int j=0;
//	printf("\ncml_help_mat\n");
//	//int cml_inte=180/CML_INTE;
//	for(i=0;i<CML_NUM;i++){
//		float k=tan(i*theta+0.0001);
//		//printf("k %f\n",k);
//		if (k<1.0003 and k>0)
//		{
//			for (j=0;j<CML_SIZE;j++){
//				chm[i][j].y=CML_SIZE-j;
//				chm[i][j].x=int(k*(j)+(CML_SIZE+1)*(1-k)/2.0+0.5);
//				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
//				}
//		}
//		else if (k>1.0003 or k<-1.0003){
//			for (j=CML_SIZE-1;j>=0;j--){
//				chm[i][j].x=j;
//				chm[i][j].y=CML_SIZE-int(((j)+(CML_SIZE+1)*(k-1)/2.0)/k+0.5);
//				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
//				}
//			}
//		else {
//			for (j=CML_SIZE-1;j>=0;j--){
//				chm[i][j].y=CML_SIZE-j;
//				chm[i][j].x=int(k*(j)+(CML_SIZE+1)*(1-k)/2.0+0.5);
//				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
//			}
		
//		}
//	}
	
//}
//void createSinogramold(sinogram s,tmat &mi,mrcMat &mM)
//{
//	int i=0;
//	int j=0;
//	for (i=0;i<CML_NUM;i++)
//	{
//		for (j=0;j<CML_SIZE;j++){
//			s[i][j]=mM[mi[i][j].x][mi[i][j].y];
//		}
//	}
//}

//void createSinogram(sinogram s,tmat &mi,mrcMat &mM)
//{
//	int i=0;
//	int j=0;
//	for (i=0;i<CML_NUM;i++)
//	{
//		for (j=0;j<CML_SIZE;j++){
//			if ( mi[i][j].x*mi[i][j].y!=0 and mi[i][j].x!=CML_SIZE-1 and mi[i][j].y!=CML_SIZE-1){
			
//			s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/9.0;
//			}
//			else if ( mi[i][j].x==0 and mi[i][j].y!=0 and mi[i][j].y!=CML_SIZE-1){
//			//no x-1
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/6.0;
//			}
//			else if ( mi[i][j].x==CML_SIZE-1 and mi[i][j].y!=0 and mi[i][j].y!=CML_SIZE-1){
//			// no x+1
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1])/6.0;
//			}
//			else if ( mi[i][j].y==0 and mi[i][j].x!=0 and mi[i][j].x!=CML_SIZE){
//			// no y-1
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/6.0;
//			}
//			else if ( mi[i][j].y==CML_SIZE-1 and mi[i][j].x!=0 and mi[i][j].x!=CML_SIZE-1){
//			// no y+1
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y])/6.0;
//			}
//			else if ( mi[i][j].x==0 and mi[i][j].y==0){
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y+1])/4.0;
//			}
//			else if ( mi[i][j].x==0 and mi[i][j].y==CML_SIZE-1){
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y-1])/4.0;
//			}
//			else if ( mi[i][j].x==CML_SIZE-1 and mi[i][j].y==0){
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x-1][mi[i][j].y+1])/4.0;
//			}
//			else if ( mi[i][j].x==CML_SIZE-1 and mi[i][j].y==CML_SIZE-1){
//				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y-1])/4.0;
//			}
//		}
//	}
//}




void getsubmrc(float *p,int x,int y,int CML_SIZE,float *s,const struct _mrchead header)
{
	printf("\ninside test p[0] %f\n",p[0]);
	if (x<=header.nx-CML_SIZE-1 and y<=header.ny-CML_SIZE-1)
	{
		int i=0;
		int j=0;
        int k=0;
		for (j=0;j<CML_SIZE;j++){
			for (i=0;i<CML_SIZE;i++){
				
                s[k]=p[header.nx*(y+j)+x+i];
                k=k+1;
				//printf("%dsp",header.nx*(y+j)+x+i);
				//printf("\n");
			}
		}
	}
    //corner and side
	else
	{
		printf("error");
	}
}
//void testp(float *p){
//	printf("\ntestp %f\n",p[0]);
//}
//void testsino(){
//	tmat m;
//	sinogram s;
//	mrcMat mM;
//	int i,j;
//	cml_help_max(m);
//	for (i=0;i<CML_SIZE;i++){
//		for (j=0;j<CML_SIZE;j++){
//			//if (i+j==CML_SIZE [>or i+j==CML_SIZE-1 or i+j==CML_SIZE+1<]){
//				//mM[i][j]=50;
//			//}
//			//else mM[i][j]=0;
//			mM[i][j]=200;
//		}
//	}
//	for (i=40;i<70;i++){
//		for (j=50;j<80;j++){
//			mM[i][j]=0;
//		}
//	}
//	for (i=0;i<CML_NUM;i++){
//		for (j=0;j<CML_SIZE;j++){
//			printf("(%d,%d)",m[i][j].x,m[i][j].y);
//		}
//		printf("\n");
//	}
//	createSinogram(s,m,mM);
//	Mat xm_origin=Mat(CML_SIZE,CML_SIZE,CV_32FC1,mM);
//	MrcProcess::showimagecpp(xm_origin);
//	Mat xms=Mat(CML_NUM,CML_SIZE,CV_32FC1,s);
//	MrcProcess::showimagecpp(xms);
//}
Mat imdft(Mat &I)
{
	
	if ( I.empty() )
		//return -1;
		printf("no image error");
		Mat padded;
		int m = getOptimalDFTSize(I.rows);
		int n = getOptimalDFTSize(I.cols);
		copyMakeBorder(I,padded,0,m- I.rows,0,n-I.cols,BORDER_CONSTANT,Scalar::all(0));
		
		Mat planes[] = {Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
		Mat complexI;
		merge(planes,2,complexI);

		dft(complexI,complexI);
		
		split(complexI,planes);
		magnitude(planes[0],planes[1],planes[0]);
		Mat magI = planes[0];

		magI += Scalar::all(1);
		log(magI,magI);

		magI= magI(Rect(0,0,magI.cols & -2 ,magI.rows & -2));
		int cx = magI.cols/2;
		int cy = magI.rows/2;

//        Mat q0(magI,Rect(0,0,cx,cy));
//        Mat q1(magI,Rect(cx,0,cx,cy));
//        Mat q2(magI,Rect(0,cy,cx,cy));
//        Mat q3(magI,Rect(cx,cy,cx,cy));

//        Mat tmp;
//        q0.copyTo(tmp);
//        q3.copyTo(q0);
//        tmp.copyTo(q3);

//        q1.copyTo(tmp);
//        q2.copyTo(q1);
//        tmp.copyTo(q2);

		normalize(magI,magI,0,1,CV_MINMAX);
		
		//imshow("Input Image",I);
		//imshow("spectrum magnitude",magI);
		//MrcProcess::showimagecpp(magI);	
		printf("\nmagI.cols %d magI.rows %d\n",magI.cols,magI.rows);
		if (magI.isContinuous()){
			printf("isContinuous\n");
		}
		else printf("not isContinuous\n");
		//magI.copyTo(I);
		return magI;
}

//float NCC(float *a,float *b,int CML_SIZE){
//    float ncc;
//    float sigma_a,sigma_b;
//    Mat t;

//}
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
    int i,j,k;
    for (i=0;i<CML_SIZE;i++){
        sum1 = sum1 + cml1[i] ;
        sum2 = sum2 + cml2[i] ;
    }
//    mean1 = sum1 / CML_SIZE;
//    mean2 = sum2 / CML_SIZE;
    Mat a,b;
    a = Mat(1,CML_SIZE,CV_32FC1,cml1);
    b = Mat(1,CML_SIZE,CV_32FC1,cml2);
    Mat tmp_m1,tmp_m2,tmp_sig1,tmp_sig2;
    meanStdDev(a,tmp_m1,tmp_sig1);
    meanStdDev(b,tmp_m2,tmp_sig2);
    mean1 = tmp_m1.at<double>(0,0);
    mean2 = tmp_m2.at<double>(0,0);
    sigma1 = tmp_sig1.at<double>(0,0);
    sigma2 = tmp_sig2.at<double>(0,0);
    float coeff;
    coeff = 1/(float(CML_SIZE)*sigma1*sigma2);
//    ncc=coeff*(ncc_fft+N*mean1*mean2-mean1*SUM(b)-mean2*SUM(a));
    float ncc_fft;//ncc_0
    float ncc_2;
    float ncc_3;
    float ncc_4;
    ncc_2 = CML_SIZE*mean1*mean2;
    ncc_3 = mean1*sum2;
    ncc_4 = mean2*sum1;
    ncc_fft=0.0;
    for(i=0;i<CML_SIZE;i++){
        ncc_fft=ncc_fft + cml1[i]*cml2[i];
    }
    ncc=coeff*(ncc_fft+ncc_2-ncc_3-ncc_4);
    //printf("\n in NCC0\n");
//    printf("\t%f\t",ncc);
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
    printf("\ncos_angle %f\n",cos_angleij);
    angleij = acos(cos_angleij);
    return angleij;
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
        else return FALSE;
}

tuple NCC_value(float *Ci,float *Cj,int after_dft_size){
//  Ci,Cj,two-dem matrix
//  change to one-d array
    tuple ret;
    int i,j,k;
    float value_ini=-10.0;
    float value[after_dft_size][after_dft_size];
    float *p1;
    float *p2;
    //mpi here
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
//            p1 = Ci[i*after_dft_size];
//            p2 = Cj[j*after_dft_size];
            value[i][j] = NCC0(&Ci[i*after_dft_size],&Cj[j*after_dft_size],after_dft_size);
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
    printf("\n%d\t%d\t%f\n",ret.x,ret.y,value_ini);
    return ret;
}

void voting_algo(int *hist,float *p,int projection_i,int projection_j,int VA_length)
 {
    int projection_k;
    int i;
    for (i=0;i<VA_length;i++){
        ;//here
    }
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
    Mat_<float> dstf=I;
    Mat_<float>::iterator it = dstf.begin();
    Mat_<float>::iterator itend = dstf.end();
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

int main()
{
//	tmat m;
//	sinogram s;
    int CML_SIZE=140;
//    float mM[CML_SIZE][CML_SIZE];
	int i,j;
	int after_dft_size;
	after_dft_size=getOptimalDFTSize(CML_SIZE);
	//float k;

	struct _mrchead header;
	//initialize the assist matrix
//	cml_help_max(m);
   /* for (i=0;i<CML_NUM;i++){*/
		//for (j=0;j<CML_SIZE;j++){
			////std::cout<<m[i][j]<<std::endl;	
			//}
	/*}*/
	
	FILE *f;
	f=fopen("/home/qjq/1.mrc","rb");
	header=MrcProcess::readhead(f);
	int	head_col=header.nx;
	int head_row=header.ny;
	float *datap=new float[head_col*head_row];
	//float datat[head_col*head_row];
	//float datap[head_col*head_row];
	std::cout<<header.nx<<","<<header.ny<<std::endl;
	MrcProcess::readmrcdata(f,datap,header);
	//MrcProcess::readmrcdata(f,datat,header);
	//printf("\ndatat[0] %f",datat[0]);
//initialize the sinogram
   /* for (i=0;i<CML_NUM;i++)*/
	//{
		//for (j=0;j<CML_SIZE;j++){
			//s[i][j]=0.0;
		//}
	//}
//	float *pt=datap;
//	/*normalize matrix before sinogram*/
//    std::vector<std::vector<int> > vec(CML_SIZE,std::vector<int>(CML_SIZE));
    float *mM = new float[CML_SIZE*CML_SIZE];
    getsubmrc(datap,332,120,CML_SIZE,mM,header);
	/*cv::Mat xm=cv::Mat(head_row,head_col,CV_32FC1,datap);*/
	//cv::Mat xm2;
	//xm2=MrcProcess::Gauss_Equal_Norm(xm);
	//printf("\nbefore norm mM\t %f",mM[0][0]);
	//cv::Mat xm_origin=cv::Mat(CML_SIZE,CML_SIZE,CV_32FC1,mM);
	//cv::Mat xm3;
	//xm3=MrcProcess::Gauss_Equal_Norm(xm_origin);
	//printf("\nafter norm mM\t %f",mM[0][0]);
	//MrcProcess::showimagecpp(xm);
	//MrcProcess::showimagecpp(xm_origin);
	//MrcProcess::showimagecpp(xm2);
	//MrcProcess::showimagecpp(xm3);
   /* for (i=0;i<CML_SIZE;i++){*/
		//for (j=0;j<CML_SIZE;j++){
			//printf("%f,",mM[i][j]);
		//}
		//printf("\n");
	/*}*/
   /* printf("mM 0 0 %f\n",mM[0][0]);*/
	/*printf("datap the first %f\n",datap[0]);*/
	//printf("\ndatap %f\n",datap[0]);
   /* testp(datap);*/
    //("datap\n");
	//for (i=0;i<CML_SIZE;i++){
		//printf("%f,",datap[i]);
	/*}*/
	Mat mMim=Mat(CML_SIZE,CML_SIZE,CV_32FC1,mM);
	//float tma=mM[0][0];
//	Mat after_dft;

    //cafter_dft = cvCreateImage(cvSize (after_dft_size,after_dft_size),IPL_DEPTH_32F,1);
//    lin_polar_img = cvCreateImage(cvSize (after_dft_size,after_dft_size),IPL_DEPTH_32F,1);
    Mat after_dft_zero=imdft(mMim);
//    Mat_<float> after_dft=after_dft_zero;
//    float aft_dft[after_dft_size][after_dft_size];
//    Mat_<float>::iterator it = after_dft.begin();
//    Mat_<float>::iterator itend = after_dft.end();
//    while (it!=itend){
//        for (i=0;i<after_dft_size;i++){
//            for (j=0;j<after_dft_size;j++){
//                aft_dft[i][j]=*it;
//                ++it;
//            }
//        }
//    }
    float aft_dft[after_dft_size][after_dft_size];
//    while (it!=itend){
//        for(i=0;i<after_dft_size*after_dft_size;i++){
//            aft_dft[i]=*it;
//            ++it;
//        }

//    }
    Mat kkk=imdft(mMim);
//    Mat kkk=imread("/home/qjq/OCYiE.jpg",0);
//    Mat dkkk=imdft(kkk);
//    MrcProcess::showimagecpp(dkkk);
//    MrcProcess::showimagecpp(kkk);
    Mat dst(kkk.size(),kkk.type());
    IplImage ipl_afdft=kkk;
    IplImage ipl_dst=dst;
    cvLinearPolar( &ipl_afdft, &ipl_dst, cvPoint2D32f(kkk.cols/2,kkk.rows/2),kkk.cols/2,CV_INTER_CUBIC);
//    MrcProcess::showimagecpp(dst);
//    printf("\ndst.height %d %d",dst.cols,dst.rows);
    //linearPolar to two array;

    Mat_<float> dstf=dst;
    Mat_<float>::iterator it = dstf.begin();
    Mat_<float>::iterator itend = dstf.end();
        while (it!=itend){
            for (i=0;i<after_dft_size;i++){
                for (j=0;j<after_dft_size;j++){
                    aft_dft[i][j]=*it;                    
                    ++it;
                }
            }
        }
    float *a = new float[after_dft_size];
    float *b = new float[after_dft_size];
    for(i=0;i<after_dft_size;i++){
        a[i]=aft_dft[0][i];
    }
    for(i=0;i<after_dft_size;i++){
        b[i]=aft_dft[1][i];
    }
    float ncc_t;
    //11.21
    ncc_t=NCC0(aft_dft[0],aft_dft[3],after_dft_size);
//    float ncc_value[after_dft_size][after_dft_size];
//    for (i=0;i<after_dft_size;i++){
//        for (j=0;j<after_dft_size;j++){
//            ncc_value[i][j]=NCC0(aft_dft[i],aft_dft[j],after_dft_size);
//        }
//    }
//    printf("\nncc_t %f\n",ncc_t);
    tuple t;
    float *all_aft = new float[after_dft_size*after_dft_size];
    int k=0;
    for(i=0;i<after_dft_size;i++){
        for(j=0;j<after_dft_size;j++){
            all_aft[k]=aft_dft[i][j];
            k++;
        }
    }

    t = NCC_value(all_aft,all_aft,after_dft_size);
//    printf("\nNCC_value t %d\t%d\t\n",t.x,t.y);
    int *cml_matrix = new int[after_dft_size*after_dft_size];
//	createSinogram(s,m,mM);
//	printf("\nsinogram\n");
//	Mat sino=Mat(CML_NUM,CML_SIZE,CV_32FC1,s);
	//cv::Mat sino2;
	//sino=MrcProcess::Gauss_Equal_Norm(sino);
	//cv::Mat sino2=MrcProcess::mynorm(sino);
    //MrcProcess::showimagecpp(sino);
//	Mat sino2;
//	sino2=MrcProcess::mynorm(sino);
	//sino.convertTo(sino,CV_8UC1,1,0);//start here 11.5
//	MrcProcess::showimagecpp(sino2);
	//printf("\nafter norm sinogram\n");
	//for (i=0;i<CML_NUM;i++)
	//{
		//for (j=0;j<CML_SIZE;j++)
		//{
			//printf("%f,",s[i][j]);
		//}
		//printf("\n");
	//}
	//tuple (*pm)[60][CML_SIZE];
	//pm=&m;
//    std::vector<float*> sss;
//    sss.push_back(all_aft);
//    float* ssss[3];
//    *ssss[0]=all_aft;
//    t = NCC_value(ssss[0],ssss[0],after_dft_size);
//    int N;
//    float *all_data = new float[N*CML_SIZE*CML_SIZE];
//    float *all_data_dft = new float[N*after_dft_size*after_dft_size];

	delete[] datap;
    delete[] a;
    delete[] b;
    delete[] all_aft;
    delete[] mM;
    delete[] cml_matrix;
//    delete[] all_data;
//    delete[] all_data_dft;
	//testsino();
    //test 3 mrc,a,b,c;
    struct _mrchead mrca,mrcb,mrcc;
    FILE *fmrca,*fmrcb,*fmrcc;
    fmrca=fopen("/home/qjq/md_em/3d_0.000000_0.000000_0.000000.mrc","rb");
    fmrcb=fopen("/home/qjq/md_em/3d_0.000000_-45.000000_-45.000000.mrc","rb");
    fmrcc=fopen("/home/qjq/md_em/3d_-45.000000_-45.000000_0.000000.mrc","rb");
    mrca=MrcProcess::readhead(fmrca);
    mrcb=MrcProcess::readhead(fmrcb);
    mrcc=MrcProcess::readhead(fmrcc);
    float *data_a=new float[mrca.nx*mrca.ny];
    float *data_b=new float[mrcb.nx*mrcb.ny];
    float *data_c=new float[mrcc.nx*mrcc.ny];
    MrcProcess::readmrcdata(fmrca,data_a,mrca);
    MrcProcess::readmrcdata(fmrcb,data_b,mrcb);
    MrcProcess::readmrcdata(fmrcc,data_c,mrcc);
    int dft_size=getOptimalDFTSize(mrca.nx);
    Mat image_a=Mat(mrca.nx,mrca.nx,CV_32FC1,data_a);
    Mat image_b=Mat(mrca.nx,mrca.nx,CV_32FC1,data_b);
    Mat image_c=Mat(mrca.nx,mrca.nx,CV_32FC1,data_c);
    Mat afdft_a=imdft(image_a);
    Mat afdft_b=imdft(image_b);
    Mat afdft_c=imdft(image_c);
    MrcProcess::showimagecpp(afdft_a);
    MrcProcess::showimagecpp(afdft_b);
    MrcProcess::showimagecpp(afdft_c);
    Mat ldft_a(afdft_a.size(),afdft_a.type());
    Mat ldft_b(afdft_b.size(),afdft_b.type());
    Mat ldft_c(afdft_c.size(),afdft_c.type());
    linearpolar(afdft_a,ldft_a);
    linearpolar(afdft_b,ldft_b);
    linearpolar(afdft_c,ldft_c);
    MrcProcess::showimagecpp(ldft_a);
    MrcProcess::showimagecpp(ldft_b);
    MrcProcess::showimagecpp(ldft_c);
    float *mata = new float[dft_size*dft_size];
    float *matb = new float[dft_size*dft_size];
    float *matc = new float[dft_size*dft_size];
    image_to_mat(ldft_a,mata,dft_size);
    image_to_mat(ldft_b,matb,dft_size);
    image_to_mat(ldft_c,matc,dft_size);
    int cab,cba,cac,cca,cbc,ccb;
    tuple ab,ac,bc;
    ab=NCC_value(mata,matb,dft_size);
    bc=NCC_value(matb,matc,dft_size);
    ac=NCC_value(mata,matc,dft_size);
    cab=ab.x;
    cba=ab.y;
    cbc=bc.x;
    ccb=bc.y;
    cac=ac.x;
    cca=ac.y;
    float angle_ab=cal_angle(cab,cac,cba,cbc,cca,ccb,dft_size);
    float angle_ac=cal_angle(cac,cab,cca,ccb,cba,cbc,dft_size);
    float angle_bc=cal_angle(cbc,cba,ccb,cca,cab,cac,dft_size);
//    printf("\nncc_value ab\t%f\n",ab);
    printf("\nangle_ab \t%f\n",angle_ab);
//    printf("\nncc_value ab\t%f\n",ac);
    printf("\nangle_ac \t%f\n",angle_ac);
//    printf("\nncc_value ab\t%f\n",bc);
    printf("\nangle_bc \t%f\n",angle_bc);
    delete[] data_a;
    delete[] data_b;
    delete[] data_c;
    delete[] mata;
    delete[] matb;
    delete[] matc;

	return 0;
}
