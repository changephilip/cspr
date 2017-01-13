#include "cml_nocv.h"
#include <time.h>
#include <sys/time.h>
//#include <array>>
#include <algorithm>
int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N=-1;
    int START=1;
	int version=-1;
    int hist_flag=0;
    char* filename;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:p:v:k:")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
//        case 'd':
//            directreaddisk_flag=1;
//            directdiskfile=optarg;
//            break;
        case 'n':
            N=atoi(optarg);
            break;
        case 'f':
            filename=optarg;
            break;
        case 'p':
            START=atoi(optarg);
            break;
		case 'v':
			version=atoi(optarg);
            break;
        case 'k':
            hist_flag=atoi(optarg);
            break;
        }
    }
    printf("cml_size\t%d\n",cml_size);
    printf("N\t%d\n",N);
    printf("00002\n");
    if (!filename) {
        printf("-f filename's abstract path is needed");
        exit(EXIT_FAILURE);
    }
    if (N==-1 or cml_size==0){
        printf("-n N and -s dft_cml_size are both needed,if N=0,then N=max\n");
        exit(EXIT_FAILURE);
    }
//    if (directreaddisk_flag==0){
//        printf("-d after_linear_dft_data on disk is needed\n");
//        exit(EXIT_FAILURE);
//    }

    int dft_size=cml_size;
    int dft_size_pow=dft_size*dft_size;
    FILE *f;
//    f=fopen("/home/qjq/data/qjq_200_data","rb");
    f=fopen(filename,"rb");
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);
    if (START-1<0 or (START-1+N)*dft_size_pow>filelength){
        printf("-p START can't L.E 0 or LARGER THAN filelength");
        exit(EXIT_FAILURE);
    }
    rewind(f);
    printf("length\t%ld\n",filelength);
    if (N==0){
        N=filelength/dft_size_pow;
    }
    if (N>filelength/dft_size_pow){
        printf("N can't be larger than the max numofItem\n");
        exit(EXIT_FAILURE);
    }
    printf("00003\n");

    int t_start,t_read_file,t_ncc_value,t_end;
    struct timeval tsBegin,tsEnd;
    t_start=time(NULL);



    int T,alpha_ij;
    T=60;
    int i,j,k;
    float sigma=180.0/float(T);
    //初始化cml矩阵
    int *cml_pair_matrix[N];
        for (i=0;i<N;i++){
            cml_pair_matrix[i]= new int[N];
        }
    printf("00004\n");

    //初始化数据集
    long malloc_size=N*dft_size_pow;
    float *lineardft_matrix=new float[malloc_size];
    fseek(f,(START-1)*dft_size_pow,SEEK_SET);
    fread(lineardft_matrix,sizeof(float),malloc_size,f);
    t_read_file=time(NULL);
    printf("%f\t%f\n",lineardft_matrix[0],lineardft_matrix[1]);
//计算cml_pair_matrix 旧方法,a cblas version 
if (version == 0){
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            if (i==j){
                cml_pair_matrix[i][j]=-1;
            }
            else {
                cmlncv_tuple tmp;
				tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                //tmp=CMLNCV::NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                cml_pair_matrix[i][j]=tmp.x;
                cml_pair_matrix[j][i]=tmp.y;
            }
        }
    }
}
//using a faster version
if (version == 1){
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            if (i==j){
                cml_pair_matrix[i][j]=-1;
            }
            else {
                cmlncv_tuple tmp;
				//tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
				tmp=CMLNCV::NCC_Q(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                cml_pair_matrix[i][j]=tmp.x;
                cml_pair_matrix[j][i]=tmp.y;
            }
        }
    }
}
//the fastest cpu version
if (version == -1){
//cml_pair_matrix 新方法
    float **total_nccq[N];
//    float ;
//    total_nccq = new float** [N];
    int postion;
    for (i=0;i<N;i++){
        total_nccq[i] = new float* [dft_size];
    }
    for (i=0;i<N;i++){
        for (j=0;j<dft_size;j++){
            total_nccq[i][j] = new float[4];
        }
    }
//    printf("000005\n");
    gettimeofday(&tsBegin,NULL);
    for (i=0;i<N;i++){
//        #pragma omp parallel for,can't openmp here
        for (j=0;j<dft_size;j++){
            postion=i*dft_size_pow+j*dft_size;
//            printf("000006\n");
            total_nccq[i][j][0] = cblas_sasum( dft_size, &lineardft_matrix[postion], 1);//sum
//            printf("000007\n");
            total_nccq[i][j][1] = total_nccq[i][j][0] / dft_size;//mean
//            printf("000008\n");
            total_nccq[i][j][2] = cblas_sdot( dft_size, &lineardft_matrix[postion], 1,&lineardft_matrix[postion],1);//dot
//            printf("000009\n");
            total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + dft_size*total_nccq[i][j][1]*total_nccq[i][j][1] - 2*total_nccq[i][j][0]*total_nccq[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
        }
    }
    gettimeofday(&tsEnd,NULL);
    printf("\t%ld\t\n",1000000L*(tsEnd.tv_sec-tsBegin.tv_sec)+tsEnd.tv_usec-tsBegin.tv_usec);
//    for (i=0;i<10000;i++){
    gettimeofday(&tsBegin,NULL);

    for (i=0;i<N;i++){
    #pragma omp parallel for
        for (j=0;j<N;j++){
            if (i==j){
                cml_pair_matrix[i][j]=-1;
            }
            else {
                cmlncv_tuple tmp;
//                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                tmp=CMLNCV::NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                cml_pair_matrix[i][j]=tmp.x;
                cml_pair_matrix[j][i]=tmp.y;
            }
        }
    }
        for (i=0;i<N;i++){
            for (j=0;j<dft_size;j++){
                delete[] total_nccq[i][j];
            }
        }
        for (i=0;i<N;i++){
            delete[] total_nccq[i];
        }

}
    gettimeofday(&tsEnd,NULL);
    printf("\t%ld\t\n",1000000L*(tsEnd.tv_sec-tsBegin.tv_sec)+tsEnd.tv_usec-tsBegin.tv_usec);

/*
    float tmp;
    gettimeofday(&tsBegin,NULL);
        #pragma omp parallel for
    for (i=0;i<N*N*dft_size_pow;i++){
        tmp=cblas_sdot( dft_size, &lineardft_matrix[0], 1,&lineardft_matrix[dft_size],1);
    }
    gettimeofday(&tsEnd,NULL);
    printf("\t%ld\t\n",1000000L*(tsEnd.tv_sec-tsBegin.tv_sec)+tsEnd.tv_usec-tsBegin.tv_usec);
*/
        t_ncc_value=time(NULL);
        printf("\ncml_pari_matrix 0\n");
        for (i=0;i<N;i++){
            printf("%d,",cml_pair_matrix[0][1]);
        }


        float *hist_peak =  new float[N*(N-1)/2];
        int *hist_index = new int[N*(N-1)/2];

        float half_pow_pi=sqrt(M_2_PI)*sigma;
//combine the voting and peak
        for (i=0;i<N;i++){
            for (j=i+1;j<N;j++){
                alpha_ij=((2*N-1-i)*i/2+j-(i+1));
                float tmp_voting[N];
                float tmp_hist[T]={0.0};
//                const int step=sizeof(tmp_hist)/sizeof(float);
//                std::array<float,60> tmp_hist={0.0};
    #pragma omp parallel for
                for (k=0;k<N;k++){

    //                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
                    if (k!=i and k!=j){
                        tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size);
                    }
                    else {
                        tmp_voting[k]=-10.0;
                    }
                }
                /*
                if(i==0 and j==2){
                    printf("\ntest tmp_voting alpha 0 2\n");
                    for (int q=0;q<N;q++){
                        printf("%f\t",tmp_voting[q]);
                    }
                   }
                */
                for (int m=0;m<N;m++){

                        float tmp=tmp_voting[m];
                        if (tmp!=-10.0 and tmp!=-9.0){
#pragma omp parallel for
                        for (int l=0;l<T;l++){
                            float alpha_t_alpha12=(180.0*l/float(T))-tmp;
                            tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/(4*sigma*sigma))/half_pow_pi;
                        }
                    }
                }
                /*
                if(i==0 and j==2){
                    printf("\ntest hist alpha 0 1\n");

                    for (int q=0;q<T;q++){
                        printf("%f\t",tmp_hist[q]);
                    }
                }
                */
//                for (int m=0;m<T;m++){
//                    int inital=0;
//                    float max_in
//                }
                hist_peak[alpha_ij]=CMLNCV::max_float(tmp_hist,T);
                hist_index[alpha_ij]=CMLNCV::max_float_index(tmp_hist,T);
//                hist_peak[alpha_ij]=*std::max_element(tmp_hist,tmp_hist+step);
//                hist_index[alpha_ij]=std::distance(tmp_hist,max_element(tmp_hist,tmp_hist+step));
            }
        }
        int time_voting;
        time_voting=time(NULL);
//计算每一个alphaij的voting序号
        int NumOfHighPeak=0;
        float threshold=N/10.0;
        for (i=0;i<N;i++){
            for (j=i+1;j<N;j++){
                int index = (2*N-1-i)*i/2+j-(i+1);
                if (hist_peak[index]>threshold) {
                    NumOfHighPeak = NumOfHighPeak +1;
                }
            }
        }
        printf("\nNumOfHighPeak\t%d\n",NumOfHighPeak);
        int *Local_SCHCP[NumOfHighPeak];
        for (i=0;i<NumOfHighPeak;i++){
            Local_SCHCP[i] = new int [N];
//            Local_SCHCP[i] = {0};//the last two ints contain the i and j of alphaij
            for (j=0;j<N;j++){
                Local_SCHCP[i][j]=0;
            }
        }

        printf("0000007\n");
        int P=0;
        for (i=0;i<N;i++){
            for (j=i+1;j<N;j++){
                int index = (2*N-1-i)*i/2+j-(i+1);
                if (hist_peak[index]>threshold) {
//                    printf("0000008\n");
                    std::vector<int> list_peak;
                    float angle_peak=hist_index[index]*sigma;
                    float tmp_voting[N];
//                    printf("0000009\n");
#pragma omp parallel for
                    for (k=0;k<N;k++){

        //                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
                        if (k!=i and k!=j){
                            tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size);
                        }
                        else {
                            tmp_voting[k]=-10.0;
                        }
                    }
//                    printf("00000316\n");
                    for (int m=0;m<N;m++){
                        if (tmp_voting[m]!=-10.0 and tmp_voting[m]!=-9.0){
                            if (fabs(tmp_voting[m]-angle_peak)<sigma){
                                list_peak.push_back(m);
                            }
                        }
                    }
//                    printf("00000323\n");
//                    Local_SCHCP[P][N]=i;
//                    Local_SCHCP[P][N+1]=j;
//                    printf("00000324\n");
//                    printf("\n%d,%d,",i,j);
                    for (auto q : list_peak){
//                        printf("%d\t",q);
                        Local_SCHCP[P][q] = 1;
                    }
//                    printf("00000330\n");
                    Local_SCHCP[P][i]= NumOfHighPeak;
                    Local_SCHCP[P][j]= NumOfHighPeak;
                    P = P+1;
//                    printf("\n");
                }
            }
        }
        int Result[N]={0};
        for (i=0;i<NumOfHighPeak;i++){
#pragma omp parallel for
            for (j=0;j<N;j++){
                Result[j]=Result[j]+Local_SCHCP[i][j];
            }
        }
        float Resultf[N];
#pragma omp parallel for
        for (i=0;i<N;i++){
            Resultf[i]=Result[i]/float(N);
        }


        for (i=0;i<NumOfHighPeak;i++){
            delete[] Local_SCHCP[i];
        }



        t_end=time(NULL);
        fclose(f);


//print the hist_index and hist_peak
if (hist_flag!=0){
        printf("alpha_ij\ti\tj\tindex\tpeak\n");
        for (i=0;i<N;i++){
            for(j=i+1;j<N;j++){
                int index=(2*N-1-i)*i/2+j-(i+1);
                printf("alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[index],hist_peak[index]);
            }
        }
}
//print the SCHCP
        printf("\n self-consistent high credible particles\n");
        for (i=0;i<N;i++){
            printf("%d\t%d\t%d\n",i+START-1,Result[i]/NumOfHighPeak,Result[i]%NumOfHighPeak);
        }
        printf("\n");
//        for (i=0;i<N;i++){
//            printf("%d\t",i);
//        }
        printf("\n");

//        printf("test ncc0 and bncc\n");
//        float ncc0=CMLNCV::NCC0(lineardft_matrix,&lineardft_matrix[200],dft_size);
//        float bncc=CMLNCV::BNCC(lineardft_matrix,&lineardft_matrix[200],dft_size);
//        printf("ncc0 %f\tbncc %f\n",ncc0,bncc);
        delete[] lineardft_matrix;
        delete[] hist_index;
        delete[] hist_peak;
        for (i=0;i<N;i++){
            delete[] cml_pair_matrix[i];
        }
//        delete[] cml_pair_matrix;
        //for (i=0;i<N;i++){
            //for (j=0;j<dft_size;j++){
                //delete[] total_nccq[i][j];
            //}
        //}
        //for (i=0;i<N;i++){
            //delete[] total_nccq[i];
        //}
//        delete[] total_nccq;
        printf("\n time read file\t%d",t_read_file-t_start);
        printf("\n time ncc value\t%d",t_ncc_value-t_read_file);
        printf("\n time voting\t%d",time_voting-t_ncc_value);
//        printf("\n time cblas1\t%d",t_cblas1-t_read_file);

        return 0;
}
