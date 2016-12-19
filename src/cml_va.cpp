#include "cml.h"
#include <time.h>
#include <fftw3.h>
//need link mpi
int main(int argc,char **argv)
{
	
    int N;
    N=atoi(argv[1]);
    printf("%d",N);
    int cml_size;
    int i,j,k;
    int t_start,t_read_file,t_ncc_value,t_end;

    t_start=time(NULL);
//    N=50;
    cml_size=140;

    int dft_size=cv::getOptimalDFTSize(cml_size);

    int cml_pair_matrix[N][N];
    std::ifstream mainMrcStarFile("/home/qjq/particles4class2d.star");
    int dft_size_pow=dft_size*dft_size;
    fileNameToCoodinateList table=CML::CNNpicker(mainMrcStarFile);
    int numItem = CML::calMnistItem(table);
    printf("\nnumItem\t%d\n",numItem);
//    long malloc_size=numItem*cml_size*cml_size;
    long malloc_size_dft=N*dft_size*dft_size;
    printf("\nlineardft_matrix_length\t%ld\n",malloc_size_dft);
//    float *cml_matrix=new float[numItem*cml_size*cml_size];
    float *lineardft_matrix=new float[malloc_size_dft];
//    CML::cml_read(cml_matrix,table,cml_size);n
    CML::cml_dftread(lineardft_matrix,table,cml_size,dft_size,N);
    printf("\nmalloc success\n");
    t_read_file=time(NULL);


//    #pragma omp parallel
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
    float *voting = new float[N*(N-1)*(N-2)/2];
    int T,alpha_ij;
    T=60;
    float *hist = new float[N*(N-1)*T/2];
//    *hist = {0};
    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++){
#pragma omp parallel for
            for (k=0;k<N;k++){
                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*(N-2)+k;
                if (k!=i and k!=j){
                    if (CML::voting_condition(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size)==TRUE) {
                        voting[alpha_ij]=CML::cal_angle(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],dft_size);
                    }
                    else voting[alpha_ij]=-9.0;
                }
                else voting[alpha_ij]=-10.0;
            }

        }
    }
    int hist_row=N*(N-1)/2;
    for (i=0;i<hist_row*T;i++){
        hist[i]=0.0;
    }
    float sigma=180.0/T;
    for (i=0;i<hist_row;i++){
        for (j=0;j<N-2;j++){
            float tmp=voting[i*(N-2)+j];
            if (voting[i*(N-2)+j]!=-10.0 /*and voting[i*(N-2)+j]!=-9.0*/){
                #pragma omp parallel for
//                {
                for(k=0;k<T;k++){
                    hist[i*T+k]=hist[i*T+k]+1*exp(-(180.0*k/T-tmp)*(180.0*k/T-tmp)/4*sigma*sigma)/sqrt(M_2_PI*sigma*sigma);
                }
//                }
            }

        }
    }
    float *hist_peak =  new float[N*(N-1)/2];
    float *peak;
    int *hist_index = new int[N*(N-1)/2];

    for (i=0;i<hist_row;i++){
        peak=&hist[i*T];
        hist_peak[i]=CML::max_float(peak,T);
        hist_index[i]=CML::max_float_index(peak,T);
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
//    delete[] cml_matrix;
//    printf("\n time read file %f\n",double((t2-t1)/CLOCKS_PER_SEC));
//    printf(" time calc %f\n",double((t3-t2)/CLOCKS_PER_SEC));
    printf("\n time read file\t%d",t_read_file-t_start);
    printf("\n time ncc value\t%d",t_ncc_value-t_read_file);
    printf("\n time voting\t%d",t_end-t_ncc_value);
    delete[] lineardft_matrix;
    delete[] voting;
    delete[] hist;
    delete[] hist_peak;
    delete[] hist_index;

}
