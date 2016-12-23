#include "cml.h"
#include <time.h>
#include <sstream>
//need link mpi
int main(int argc,char **argv)
{
	
        int oc;                     /*选项字符 */
        //char *b_opt_arg;            /*选项参数字串 */


        int calculate_flag=0;
        int writetodisk_flag=0;
        int directreaddisk_flag=0;
        int star_flag=0;
        int cml_size=0;
        int N=-1;
        char* directdiskfile;
        char* outdiskfile;
        char* starfilename;

        while((oc = getopt(argc, argv, "s:r:d:w:cn:")) != -1)
        {
            switch(oc)
            {
                case 's':
                    cml_size=atoi(optarg);
                    break;
                case 'r':
                    star_flag==1;
                    starfilename=optarg;
                case 'd':
                    directreaddisk_flag=1;
                    directdiskfile=optarg;
                    break;
                case 'w':
                    writetodisk_flag=1;
                    outdiskfile= optarg;
                    break;
                case 'c':
                    calculate_flag=1;
                    break;
                case 'n':
                    N=atoi(optarg);
                 break;
            }
        }
        if (N==-1 or cml_size==0){
            printf("-n N and -s cml_size are both needed\n");
            exit(0);
        }
        if (stat_flag==0 or directreaddisk_flag==0){
            printf("-r .star file or -d after_linear_dft_data on disk is at least needed\n");
            exit(0);
        }

    int dft_size=cv::getOptimalDFTSize(cml_size);
    int dft_size_pow=dft_size*dft_size;


    int i,j,k;
    int t_start,t_read_file,t_ncc_value,t_end;
    t_start=time(NULL);




if (star_flag==1){
    string starfilename_string(starfilename);
    std::ifstream mainMrcStarFile(starfilename_string);
    fileNameToCoodinateList table=CML::CNNpicker(mainMrcStarFile);
    int numItem = CML::calMnistItem(table);
    printf("\nnumItem\t%d\n",numItem);
    if (N==0){
        N=numItem;
    }

}
if (directreaddisk_flag==1){

}

    long malloc_size_dft=N*dft_size*dft_size;
    printf("\nlineardft_matrix_length\t%ld\n",malloc_size_dft);
    float *lineardft_matrix=new float[malloc_size_dft];
    CML::cml_dftread(lineardft_matrix,table,cml_size,dft_size,N);
    printf("\nmalloc success\n");

    t_read_file=time(NULL);



if (calculate_flag==1){

    int T,alpha_ij;
    T=60;
    float sigma=180.0/T;


    int hist_row=N*(N-1)/2;


//初始化cml矩阵
    int *cml_pair_matrix[N];
    for (i=0;i<N;i++){
        cml_pair_matrix[i]= new int[N];
    }
//初始化对每一个alphaij的角度计算矩阵
    float *voting = new float[N*(N-1)*N/2];
//初始化hist矩阵的每个值为0.0
    float *hist = new float[N*(N-1)*T/2];

    for (i=0;i<hist_row*T;i++){
        hist[i]=0.0;
    }
//hist_peak矩阵的初始化，记录每一个alphaij的最大可能值
    float *hist_peak =  new float[N*(N-1)/2];
    int *hist_index = new int[N*(N-1)/2];
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            if (i==j){
                cml_pair_matrix[i][j]=-1;
            }
            else {
                cml_tuple tmp;
                tmp=CML::NCC_valuet(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                cml_pair_matrix[i][j]=tmp.x;
                cml_pair_matrix[j][i]=tmp.y;
            }
        }
    }
    t_ncc_value=time(NULL);

    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++){
            alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N;
#pragma omp parallel for
            for (k=0;k<N;k++){
                    int index=alpha_ij+k;
//                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
                if (k!=i and k!=j){
                    if (CML::voting_condition(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size)==TRUE) {
                        voting[index]=CML::cal_angle(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size);
                    }
                    else {
                        voting[index]=-9.0;
                    }
                }
                else {
                    voting[index]=-10.0;
                }
            }
        }
    }


    float half_pow_pi=sqrt(M_2_PI)*sigma;
    for (i=0;i<hist_row;i++){
        for (j=0;j<N;j++){
            float tmp=voting[i*N+j];
            if (tmp!=-10.0 and tmp!=-9.0){
                #pragma omp parallel for
//                {
                for(k=0;k<T;k++){
                    hist[i*T+k]=hist[i*T+k]+exp(-(180.0*k/T-tmp)*(180.0*k/T-tmp)/(4*sigma*sigma))/half_pow_pi;
                }
//                }
            }

        }
    }

    #pragma omp parallel for
    for (i=0;i<hist_row;i++){
//        peak=&hist[i*T];
        hist_peak[i]=CML::max_float(&hist[i*T],T);
        hist_index[i]=CML::max_float_index(&hist[i*T],T);
        }
//    free(peak);
    t_end=time(NULL);
//    for (i=0;i<N;i++){

//    }


//test med data
    /*
    printf("alpha_ij\ti\tj\tindex\tpeak\n");
    for (i=0;i<N;i++){
        for(j=i+1;j<N;j++){
            int index=(2*N-1-i)*i/2+j-(i+1);
            printf("alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[i],hist_peak[index]);
        }
    }
    printf("alpha_ij\t0\n");
    for(i=0;i<N;i++){
        printf("%d,",cml_pair_matrix[1][i]);
    }
    printf("voting[0]\n");
    for (i=0;i<N;i++){
        printf("%f,",voting[i]);
    }
    printf("voting[1]\n");
    for (i=0;i<N;i++){
        printf("%f,",voting[i+N]);
    }
    printf("\nhist_peak[1]\n");
    for(i=0;i<T;i++){
        printf("%f\t",hist[i]);
    }
    printf("hist_peak[2]\n");
    for(i=0;i<T;i++){
        printf("%f\t",hist[T+i]);
    }
    */
//    delete[] cml_matrix;
//    printf("\n time read file %f\n",double((t2-t1)/CLOCKS_PER_SEC));
//    printf(" time calc %f\n",double((t3-t2)/CLOCKS_PER_SEC));
    printf("\n time read file\t%d",t_read_file-t_start);
    printf("\n time ncc value\t%d",t_ncc_value-t_read_file);
    printf("\n time voting\t%d",t_end-t_ncc_value);

    #pragma omp parallel for
    for (i=0;i<N;i++){
        delete[] cml_pair_matrix[i];
    }
    float a[9]={0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    printf("test max_float\t%f\n",CML::max_float(&a[1],8));
    printf("test max_float_index\t%d\n",CML::max_float_index(&a[1],8));
//    delete cml_pair_matrix;
    }

}
