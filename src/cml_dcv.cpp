#include "cml.h"
#include <time.h>
#include <sstream>
//need link mpi
int main(int argc,char **argv)
{

    int N;
    N=atoi(argv[1]);
    printf("%d\n",N);

    int cml_size;
    int i,j,k;
    int t_start,t_read_file,t_ncc_value,t_end;
    int T,alpha_ij;
    T=60;
    float sigma=180.0/T;

    t_start=time(NULL);

    cml_size=140;
    int dft_size=cv::getOptimalDFTSize(cml_size);
    int dft_size_pow=dft_size*dft_size;


    int hist_row=N*(N-1)/2;


//初始化cml矩阵
    int *cml_pair_matrix[N];
    for (i=0;i<N;i++){
        cml_pair_matrix[i]= new int[N];
    }
//初始化对每一个alphaij的角度计算矩阵
    float *voting[hist_row];
    for (i=0;i<hist_row;i++){
        voting[i]=new float[N];
    }

//初始化hist矩阵的每个值为0.0
    float *hist[hist_row];
    for (i=0;i<hist_row;i++){
        hist[i]=new float[T];
    }

    for (i=0;i<hist_row;i++){
        for (j=0;j<T;j++){
            hist[i][j]=0.0;
        }
    }

//hist_peak矩阵的初始化，记录每一个alphaij的最大可能值
    float *hist_peak=new float[hist_row];
    int *hist_index = new int[hist_row];

    std::ifstream mainMrcStarFile("/home/qjq/particles4class2d.star");

    fileNameToCoodinateList table=CML::CNNpicker(mainMrcStarFile);
    int numItem = CML::calMnistItem(table);
    FILE *f;
    f=fopen("qjq_data","wb");
    printf("\nnumItem\t%d\n",numItem);
    long malloc_size_dft=numItem*dft_size*dft_size;
    printf("\nlineardft_matrix_length\t%ld\n",malloc_size_dft);
    float *lineardft_matrix=new float[malloc_size_dft];
    float *disk_dft_matrix=new float[malloc_size_dft];
    CML::cml_dftread(lineardft_matrix,table,cml_size,dft_size,N);

    printf("\nmalloc success\n");
    char sbuf[1960000];
    setvbuf(f,sbuf,_IOFBF,1960000);
    fwrite(lineardft_matrix,sizeof(float),malloc_size_dft,f);
    fseek(f,0,SEEK_END);
    long filelength;
    filelength=ftell(f);
    printf("qjq_data length %ld\n",filelength);
    fclose(f);
    f=fopen("qjq_data","rb");
    fseek(f,0,SEEK_SET);
    fread(disk_dft_matrix,sizeof(float),malloc_size_dft,f);
    for (i=0;i<malloc_size_dft;i++){
        if (lineardft_matrix[i]!=disk_dft_matrix[i]){
            printf("line %f \t disk %f\n",lineardft_matrix[i],disk_dft_matrix[i]);
            printf("i %d error not equal\n",i);
            exit(EXIT_FAILURE);
        }
    }
//    printf("lineardft_matrix \t%f\n",lineardft_matrix[0]);
//    printf("disk_matrix \t%f\n",disk_dft_matrix[0]);
    fclose(f);

    t_read_file=time(NULL);


        for (i=0;i<N;i++){
            for (j=0;j<N;j++){
                if (i==j){
                    cml_pair_matrix[i][j]=-1;
                }
                else {
                    cml_tuple tmp;
                    tmp=CML::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                    cml_pair_matrix[i][j]=tmp.x;
                    cml_pair_matrix[j][i]=tmp.y;
                }
            }
        }


        t_ncc_value=time(NULL);

        for (i=0;i<N;i++){
            for (j=i+1;j<N;j++){
                alpha_ij=(2*N-1-i)*i/2+j-(i+1);
#pragma omp parallel for
                for (k=0;k<N;k++){
                    if (k!=i and k!=j){
                        if (CML::voting_condition(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size)==TRUE) {
                            voting[alpha_ij][k]=CML::cal_angle(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size);
                            }
                        else {
                            voting[alpha_ij][k]=-9.0;
                            }
                        }
                    else {
                        voting[alpha_ij][k]=-10.0;
                    }
                }
            }
        }




        float half_pow_pi=sqrt(M_2_PI)*sigma;
        for (i=0;i<hist_row;i++){
            for (j=0;j<N;j++){
                float tmp=voting[i][j];
                if (tmp!=-10.0 and tmp!=-9.0){
                    #pragma omp parallel for

                    for(k=0;k<T;k++){
                        hist[i][k]=hist[i][k]+exp(-(180.0*k/T-tmp)*(180.0*k/T-tmp)/(4*sigma*sigma))/half_pow_pi;
                    }

                }

            }
        }

    //    float *peak;

        #pragma omp parallel for
        for (i=0;i<hist_row;i++){
    //        peak=&hist[i*T];
            hist_peak[i]=CML::max_float(hist[i],T);
            hist_index[i]=CML::max_float_index(hist[i],T);
            }
    //    free(peak);
        t_end=time(NULL);
    //    for (i=0;i<N;i++){

    //    }
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
            printf("%f,",voting[0][i]);
        }
        printf("voting[1]\n");
        for (i=0;i<N;i++){
            printf("%f,",voting[1][i]);
        }
        printf("\nhist_peak[1]\n");
        for(i=0;i<T;i++){
            printf("%f\t",hist[0][i]);
        }
        printf("\nhist_peak[2]\n");
        for(i=0;i<T;i++){
            printf("%f\t",hist[1][i]);
        }


        printf("\n time read file\t%d",t_read_file-t_start);
        printf("\n time ncc value\t%d",t_ncc_value-t_read_file);
        printf("\n time voting\t%d",t_end-t_ncc_value);


        delete[] lineardft_matrix;
        delete[] disk_dft_matrix;
        delete[] hist_peak;
        delete[] hist_index;
        #pragma omp parallel for
        for (i=0;i<N;i++){
            delete[] cml_pair_matrix[i];
        }
        #pragma omp parallel for
        for (i=0;i<hist_row;i++){
            delete[] voting[i];
            delete[] hist[i];
        }
        float a[9]={0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
        printf("test max_float\t%f\n",CML::max_float(&a[1],8));
        printf("test max_float_index\t%d\n",CML::max_float_index(&a[1],8));

    return 0;
    }
