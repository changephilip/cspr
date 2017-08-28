#include "cml.h"
#include <time.h>
#include <sstream>
#include "mrcprocess.h"
float interpolate_bilinear(cv::Mat &mat_src,double ri,int rf,int rc,double ti,int tf,int tc){
  double inter_value=0.0;
  if (rf == rc && tc == tf){
    inter_value = mat_src.ptr<float>(rc)[tc];
  }
  else if (rf ==rc){
    inter_value = (ti - tf)*mat_src.ptr<float>(rf)[tc]+(tc - ti)*mat_src.ptr<float>(rf)[tf];
  }
  else if (tf == tc){
    inter_value = (ri -rf)*mat_src.ptr<float>(tc)[tf] + (rc - ri)*mat_src.ptr<float>(rf)[tf];
  }
  else{
    double inter_r1 = (ti - tf)*mat_src.ptr<float>(rf)[tc] + (tc - ti)*mat_src.ptr<float>(rf)[tf];
    double inter_r2 = (ti - tf)*mat_src.ptr<float>(rc)[tc] + (tc - ti)*mat_src.ptr<float>(rc)[tf];
    inter_value = (ri - rf)*inter_r2 + (rc-ri) * inter_r1;
  }
  return (float) inter_value;
}
bool cartesian_to_polar(cv::Mat &mat_c,cv::Mat &mat_p,int img_d){
  mat_p = cv::Mat::zeros(img_d,img_d,CV_32FC1);
  int line_len = mat_c.rows;
  int line_num = mat_c.cols;
  double delta_r = (2.0*line_len)/(img_d-1);
  double delta_t = 2.0*M_PI/line_num;
  double center_x = (img_d-1)/2.0;
  double center_y = (img_d-1)/2.0;

  for (int i=0;i<img_d;i++){
    for (int j=0;j<img_d;j++){
      double rx = j-center_x;
      double ry = center_y -i;
      double r = std::sqrt(rx*rx + ry*ry);
      if (r<=(img_d-1)/2.0){
        double ri = r*delta_r;
        int rf = (int)std::floor(ri);
        int rc = (int)std::ceil(ri);

        if (rf<0){
          rf=0;
        }
        if (rc>(line_len-1)){
          rc = line_len - 1;
        }
        double t = std::atan2(ry,rx);
        if (t<0){
          t = t + 2.0*M_PI;
        }

        double ti = t/delta_t;
        int tf= (int)std::floor(ti);
        int tc = (int)std::ceil(ti);

        if (tf<0){
          tf=0;
        }
        if (tc>(line_num -1 )){
          tc = line_num -1;
        }
        mat_p.ptr<float>(i)[j]=interpolate_bilinear(mat_c,ri,rf,rc,ti,tf,tc);
      }
    }
  }
  return true;
}
bool polar_to_cartesian(cv::Mat & mat_p,cv::Mat & mat_c,int rows_c,int cols_c){
  mat_c = cv::Mat::zeros(rows_c,cols_c,CV_32FC1);
  int polar_d=mat_p.cols;
  double polar_r = polar_d/2.0;
  double delta_r = polar_r/rows_c;
  double delta_t = 2.0*M_PI/cols_c;
  double center_polar_x = (polar_d -1)/2.0;
  double center_polar_y = (polar_d - 1)/2.0;
  for (int i=0;i<cols_c;i++){
    double theta_p = i*delta_t;
    double sin_theta = std::sin(theta_p);
    double cos_theta = std::cos(theta_p);
    for (int j=0;j<rows_c;j++){
      double temp_r = j *delta_r;
      int polar_x = (int)(center_polar_x + temp_r*cos_theta);
      int polar_y = (int)(center_polar_y - temp_r*sin_theta);
      mat_c.ptr<float>(j)[i] = mat_p.ptr<float>(polar_y)[polar_x];
    }
  }
  return true;
}
int main(int argc,char * argv[]){
    int cml_size=0;
    int i,j,k,l;
    int oc;
//    int dft_size;
    char *inname;
    char *outname;
    FILE *inmrc;
    FILE *outbin;
    double scale=cml_size/144.0;
    const int SIZE_COMP=128;

    while ((oc = getopt(argc,argv,"s:i:o:")) != -1)
    {
        switch(oc)
        {
            case 's':
            cml_size=atoi(optarg);
            break;
        case 'i':
            inname=optarg;
            break;
        case 'o':
            outname=optarg;
            break;
        }
    }
    if (cml_size==0 or !inname or !outname){
        printf("cml_size should be an int which is larger than 0\n");
        printf("or the input mrc file and output file are both needed\n");
        exit(EXIT_FAILURE);
    }



//    dft_size=cv::getOptimalDFTSize(cml_size);
//    int dft_size_pow=dft_size*dft_size;
    int cml_size_pow=cml_size*cml_size;

//    inmrc=fopen("/home/qjq/data/shiny-200-pf.mrcs","rb");
//    outbin=fopen("/home/qjq/data/qjq-200-data-posi","wb");
    inmrc=fopen(inname,"rb");
    outbin=fopen(outname,"wb");

    long filelength;
    fseek(inmrc,0,SEEK_END);
    filelength=ftell(inmrc);
    if (((filelength-1024)%(4*cml_size_pow))!=0){
        printf("your mrcs file is wrong!\n");
        printf("the filelength of mrcs file cna't be modded by float*cml_size^2\n");
        exit(EXIT_FAILURE);
    }
    int numItem=(filelength-1024)/(4*cml_size_pow);
    printf("%d images need to be processed\n",numItem);
    rewind(inmrc);

    fseek(inmrc,1024,SEEK_SET);


    for (i=0;i<numItem;i++){
        float tmp[cml_size_pow];

//        float tmpdft[dft_size_pow];

        fseek(inmrc,i*cml_size_pow*sizeof(tmp[0])+1024,SEEK_SET);
        fread(tmp,sizeof(tmp[0]),cml_size_pow,inmrc);

        cv::Mat image_mrc=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);


        cv::Mat image_dft=CML::imdft(image_mrc);
        cv::Mat lp_mrc(image_mrc.size(),image_mrc.type());
        CML::linearpolar(image_dft,lp_mrc);
        //cartesian_to_polar(image_dft,lp_mrc,cml_size);
        //polar_to_cartesian(image_dft,lp_mrc,cml_size,cml_size);
        if (i==0 or i==5 or i==43){
            MrcProcess::showimagecpp(image_mrc);
			//MrcProcess::showimagecpp(MrcProcess::mynorm(image_mrc));
            MrcProcess::showimagecpp(image_dft);
			//MrcProcess::showimagecpp(MrcProcess::mynorm(image_dft));
            MrcProcess::showimagecpp(lp_mrc);
			//MrcProcess::showimagecpp(MrcProcess::mynorm(lp_mrc));
        }
		
//        MrcProcess::showimagecpp(lpdft_mrc);
        float tmp_comp[SIZE_COMP*SIZE_COMP];
        cv::Size dsize=cv::Size(SIZE_COMP,SIZE_COMP);
        cv::Mat image_comp=cv::Mat(dsize,CV_32FC1);
        cv::resize(lp_mrc,image_comp,dsize);
//        cv::normalize(lpdft_mrc,lpdft_mrc,0,1,CV_MINMAX);
        CML::image_to_mat(image_comp,tmp_comp,SIZE_COMP);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }
        //cv::Size cutsize=cv::Size(SIZE_COMP-36,SIZE_COMP);
                cv::Size cutsize=cv::Size(SIZE_COMP-106,SIZE_COMP);
//		cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1);
		float tmp_cut[SIZE_COMP][SIZE_COMP-106];
		
		for (j=0;j<SIZE_COMP;j++){
			for (k=0;k<22;k++){
				tmp_cut[j][k]=lp_mrc.at<float>(j,k+1);
			}
		}
		cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1,tmp_cut);
		cv::resize(image_cut,image_comp,dsize);
		CML::image_to_mat(image_comp,tmp_comp,SIZE_COMP);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }
        if (i==0 or i==5 or i==43){
            MrcProcess::showimagecpp(image_cut);
            MrcProcess::showimagecpp(image_comp);
        }
        fwrite(tmp_comp,sizeof(tmp_comp[0]),SIZE_COMP*SIZE_COMP,outbin);

    }

    fclose(outbin);
    printf("write processed data to file successfully!\n");
    //test

    fseek(inmrc,1024,SEEK_SET);

    FILE *testbin;
//    testbin=fopen("/home/qjq/data/qjq-200-data-posi","rb");
    testbin=fopen(outname,"rb");

    for (i=0;i<numItem;i++){
        float tmp[cml_size_pow];
//        float tmpdft[dft_size_pow];

        fseek(inmrc,i*cml_size_pow*sizeof(tmp[0])+1024,SEEK_SET);
        fread(tmp,sizeof(tmp[0]),cml_size_pow,inmrc);

        cv::Mat image_mrc=cv::Mat(cml_size,cml_size,CV_32FC1,tmp);
        cv::Mat image_dft=CML::imdft(image_mrc);
        cv::Mat lp_mrc(image_mrc.size(),image_mrc.type());
        //cartesian_to_polar(image_dft,lp_mrc,cml_size);
        CML::linearpolar(image_dft,lp_mrc);
        //polar_to_cartesian(image_dft,lp_mrc,cml_size,cml_size);
//        MrcProcess::showimagecpp(lpdft_mrc);

//        cv::normalize(lpdft_mrc,lpdft_mrc,0,1,CV_MINMAX);
        float tmp_comp[SIZE_COMP*SIZE_COMP];
        cv::Size dsize=cv::Size(SIZE_COMP,SIZE_COMP);
        cv::Mat image_comp=cv::Mat(dsize,CV_32FC1);
        cv::resize(lp_mrc,image_comp,dsize);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }
		cv::Size cutsize=cv::Size(SIZE_COMP-106,SIZE_COMP);
//		cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1);
		float tmp_cut[SIZE_COMP][SIZE_COMP-106];
		
		for (j=0;j<SIZE_COMP;j++){
			for (k=0;k<22;k++){
				tmp_cut[j][k]=lp_mrc.at<float>(j,k+1);
			}
		}
		cv::Mat image_cut=cv::Mat(cutsize,CV_32FC1,tmp_cut);
		cv::resize(image_cut,image_comp,dsize);
		CML::image_to_mat(image_comp,tmp_comp,SIZE_COMP);
        l=0;
        for (j=0;j<SIZE_COMP;j++){
            for (k=0;k<SIZE_COMP;k++){
                tmp_comp[l]=lp_mrc.at<float>(j,k);
                l=l+1;
            }

        }


        float tmptest[SIZE_COMP*SIZE_COMP];
        fread(tmptest,sizeof(tmptest[0]),SIZE_COMP*SIZE_COMP,testbin);

        float sum_pow=0.0;
        for ( j=0;j<SIZE_COMP*SIZE_COMP;j++){
            sum_pow=sum_pow+(tmptest[j]-tmp_comp[j])*(tmptest[j]-tmp_comp[j]);
        }
        sum_pow=sqrt(sum_pow/SIZE_COMP*SIZE_COMP);
        if (sum_pow>0.0001){
            printf("i=%d\tsumpow=%f\n",i,sum_pow);
        }
        /*
        printf("i=%d\n",i);
        for ( j=0;j<dft_size*dft_size;j++){
            printf("%f\t%f\t%f\n",tmptest[j],tmpdft[j],tmptest[j]-tmpdft[j]);
        }
        printf("\n");
        */
//        for (int j=0;j<2*dft_size;j++){
//            printf("%f\t",);
//        }
//        printf("\n");


    }

    printf("Test is Done\nSee the screen log\n");

    fclose(testbin);

    fclose(inmrc);
    return 0;

}

