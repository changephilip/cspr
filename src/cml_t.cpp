#include "mrcparser.h"
#include "mnistload.h"
#include "mrcprocess.h"
//#include "gsl/gsl_matrix.h"
#include <iostream>
#include <math.h>
#include <gsl/gsl_fft.h>

//#include "dft.cpp"
#define CML_INTE 1
#define CML_NUM 180
//int CML_NUM*int CML_LINE=180
#define CML_SIZE 140
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
typedef float sinogram[CML_NUM][CML_SIZE];	
typedef tuple tmat[CML_NUM][CML_SIZE];
typedef float mrcMat[CML_SIZE][CML_SIZE];
void cml_help_max(tmat chm){
	/*this function should be used to create line of matrix*/
	/*according to the matrix's value,we calculate the line of mrc matrix*/
	/*defaultly,we will have 60 lines in one projection,the size of line is CML_SIZE*/
	/*struct tuple matrix can't be returned*/
	/*so the code should be insert beside other code*/
	//struct tuple m[60][CML_SIZE];
	float theta=3.1415926/float(CML_NUM);
	printf("theta %f",theta);
	int i=0;
	int j=0;
	printf("\ncml_help_mat\n");
	//int cml_inte=180/CML_INTE;
	for(i=0;i<CML_NUM;i++){
		float k=tan(i*theta+0.0001);
		printf("k %f\n",k);
		if (k<1.0003 and k>0)
		{
			for (j=0;j<CML_SIZE;j++){
				chm[i][j].y=CML_SIZE-j;
				chm[i][j].x=int(k*(j)+(CML_SIZE+1)*(1-k)/2.0+0.5);
				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
				}
		}
		else if (k>1.0003 or k<-1.0003){
			for (j=CML_SIZE-1;j>=0;j--){
				chm[i][j].x=j;
				chm[i][j].y=CML_SIZE-int(((j)+(CML_SIZE+1)*(k-1)/2.0)/k+0.5);
				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
				}	
			}
		else {
			for (j=CML_SIZE-1;j>=0;j--){
				chm[i][j].y=CML_SIZE-j;
				chm[i][j].x=int(k*(j)+(CML_SIZE+1)*(1-k)/2.0+0.5);
				//std::cout<<k<<","<<i<<","<<j<<","<<chm[i][j].x<<","<<chm[i][j].y<<std::endl;
			}
		
		}
	}
	
}
void createSinogram(sinogram s,tmat &mi,mrcMat &mM)
{
	int i=0;
	int j=0;
	for (i=0;i<CML_NUM;i++)
	{
		for (j=0;j<CML_SIZE;j++){
			s[i][j]=mM[mi[i][j].x][mi[i][j].y];	
		}
	}
}

void createSinogramave(sinogram s,tmat &mi,mrcMat &mM)
{
	int i=0;
	int j=0;
	for (i=0;i<CML_NUM;i++)
	{
		for (j=0;j<CML_SIZE;j++){
			if ( mi[i][j].x*mi[i][j].y!=0 and mi[i][j].x!=CML_SIZE and mi[i][j].y!=CML_SIZE){
			
			s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/9.0	
			}
			else if ( mi[i][j].x==0 and mi[i][j].y!=0 and mi[i][j].y!=CML_SIZE){
			//no x-1
				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/6.0;
			}
			else if ( mi[i][j].x==CML_SIZE and mi[i][j].y!=0 and mi[i][j].y!=CML_SIZE){
			// no x+1
				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1])/6.0;	
			}
			else if ( mi[i][j].y==0 and mi[i][j].x!=0 and mi[i][j].x!=CML_SIZE){
			// no y-1
				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y+1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y]+mM[mi[i][j].x+1][mi[i][j].y+1])/9.0;	
			}
			else if ( mi[i][j].y==CML_SIZE and mi[i][j].x!=0 and mi[i][j].x!=CML_SIZE){
			// no y+1
				s[i][j]=(mM[mi[i][j].x][mi[i][j].y]+mM[mi[i][j].x-1][mi[i][j].y-1]+mM[mi[i][j].x-1][mi[i][j].y]+mM[mi[i][j].x][mi[i][j].y-1]+mM[mi[i][j].x][mi[i][j].y+1]+mM[mi[i][j].x+1][mi[i][j].y-1]+mM[mi[i][j].x+1][mi[i][j].y])/6.0;
			
			}
		}
	}
}




void getsubmrc(float *p,int x,int y,mrcMat s,const struct _mrchead header)
{
	printf("\ninside test p[0] %f\n",p[0]);
	if (x<=header.nx-CML_SIZE-1 and y<=header.ny-CML_SIZE-1)
	{
		int i=0;
		int j=0;
		for (j=0;j<CML_SIZE;j++){
			for (i=0;i<CML_SIZE;i++){
				
				s[j][i]=p[header.nx*(y+j)+x+i];
				//printf("%dsp",header.nx*(y+j)+x+i);
				//printf("\n");
			}
		}
	}
	else
	{
		printf("error");
	}
}
void testp(float *p){
	printf("\ntestp %f\n",p[0]);
}
void testsino(){
	tmat m;
	sinogram s;
	mrcMat mM;
	int i,j;
	cml_help_max(m);
	for (i=0;i<CML_SIZE;i++){
		for (j=0;j<CML_SIZE;j++){
			if (i+j==CML_SIZE or i+j==CML_SIZE-1 or i+j==CML_SIZE+1){
				mM[i][j]=50;
			}
			else mM[i][j]=0;
		}
	}
	for (i=0;i<CML_NUM;i++){
		for (j=0;j<CML_SIZE;j++){
			printf("(%d,%d)",m[i][j].x,m[i][j].y);
		}
		printf("\n");
	}
	createSinogram(s,m,mM);
	cv::Mat xm_origin=cv::Mat(CML_SIZE,CML_SIZE,CV_32FC1,mM);
	MrcProcess::showimagecpp(xm_origin);
	cv::Mat xms=cv::Mat(CML_NUM,CML_SIZE,CV_32FC1,s);
	MrcProcess::showimagecpp(xms);
}
int main()
{
	tmat m;	
	sinogram s;
	mrcMat mM; 
	int i,j;
	//float k;
	struct _mrchead header;
	//initialize the assist matrix
	cml_help_max(m);
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
	getsubmrc(datap,0,0,mM,header);	
	/*cv::Mat xm=cv::Mat(head_row,head_col,CV_32FC1,datap);*/
	//cv::Mat xm2;
	//xm2=MrcProcess::Gauss_Equal_Norm(xm);
	//printf("\nbefore norm mM\t %f",mM[0][0]);
	cv::Mat xm_origin=cv::Mat(CML_SIZE,CML_SIZE,CV_32FC1,mM);
	cv::Mat xm3;
	xm3=MrcProcess::Gauss_Equal_Norm(xm_origin);
	//printf("\nafter norm mM\t %f",mM[0][0]);
	//MrcProcess::showimagecpp(xm);
	//MrcProcess::showimagecpp(xm_origin);
	//MrcProcess::showimagecpp(xm2);
	MrcProcess::showimagecpp(xm3);
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
	//printf("datap\n");
	//for (i=0;i<CML_SIZE;i++){
		//printf("%f,",datap[i]);
	/*}*/
	createSinogram(s,m,mM);
	printf("\nsinogram\n");
	for (i=0;i<CML_NUM;i++)
	{
		for (j=0;j<CML_SIZE;j++)
		{
			printf("%f,",s[i][j]);
		}
		printf("\n");
	}
	cv::Mat sino=cv::Mat(CML_NUM,CML_SIZE,CV_8U,s);
	//cv::Mat sino2;
	//sino=MrcProcess::Gauss_Equal_Norm(sino);
	cv::Mat sino2=MrcProcess::mynorm(sino);
	MrcProcess::showimagecpp(sino2);
	printf("\nafter norm sinogram\n");
	for (i=0;i<CML_NUM;i++)
	{
		for (j=0;j<CML_SIZE;j++)
		{
			printf("%f,",s[i][j]);
		}
		printf("\n");
	}

	//tuple (*pm)[60][CML_SIZE];
	//pm=&m;
	delete[] datap;
	testsino();
	return 0;
}
