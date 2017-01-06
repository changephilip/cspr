#include "cml_cl.h"
#include <time.h>
//#include <array>>
#include <algorithm>
int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N=-1;
    char* filename;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:")) != -1)
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
    fread(lineardft_matrix,sizeof(float),malloc_size,f);
    t_read_file=time(NULL);
    printf("%f\t%f\n",lineardft_matrix[0],lineardft_matrix[1]);
//计算cml_pair_matrix
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            if (i==j){
                cml_pair_matrix[i][j]=-1;
            }
            else {
                cmlncv_tuple tmp;
                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                cml_pair_matrix[i][j]=tmp.x;
                cml_pair_matrix[j][i]=tmp.y;
            }
        }
    }
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
                if(i==0 and j==2){
                    printf("\ntest tmp_voting alpha 0 2\n");
                    for (int q=0;q<N;q++){
                        printf("%f\t",tmp_voting[q]);
                    }
                   }
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
                if(i==0 and j==2){
                    printf("\ntest hist alpha 0 1\n");

                    for (int q=0;q<T;q++){
                        printf("%f\t",tmp_hist[q]);
                    }
                }
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
        t_end=time(NULL);
        fclose(f);
        printf("alpha_ij\ti\tj\tindex\tpeak\n");
        for (i=0;i<N;i++){
            for(j=i+1;j<N;j++){
                int index=(2*N-1-i)*i/2+j-(i+1);
                printf("alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[index],hist_peak[index]);
            }
        }
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
        printf("\n time read file\t%d",t_read_file-t_start);
        printf("\n time ncc value\t%d",t_ncc_value-t_read_file);
        printf("\n time voting\t%d",t_end-t_ncc_value);
        return 0;
}

