//#include "cml_nocv.h"
#include "cml_cuda.h"
//#include "cml_cuda.cu"
#include <time.h>
#include <sys/time.h>
//#include <array>>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define L 140
#define L_power 19600

struct voted
{
    int index;
    int value;
};
typedef struct{
    int x;
    int y;
    float value;
} cml_retstruc;
bool comp(const voted &a,const voted &b)
{
    return a.value<b.value;
}
__global__ void parent_ncc_kernel(float *d_data,int *d_ctr,int *d_ctr_id1,int *d_ctr_id2,float *d_sum,float *d_mean,float *d_stdv,int N,cml_retstruc *S){
     //获取局部id
     //设置cublas环境，启动cublas_sgemm
     //设置局部变量C3,接受sgemmm的结果，估计为160K,调用子内核时，不能使用local memory,必须把C3分配在global memory
     //调整方案，不使用子内核调用，直接部署代码
    int globalThreadID=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    int imagea=d_ctr_id1[globalThreadID];
    int imageb=d_ctr_id2[globalThreadID];
//    int L_power=L*L;
    int i,j;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha=1.0;
    float beta=0.0;
    float C3[L_power];
    //cudaMalloc((void**)&C3,L*L*sizeof(float));
    cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,&d_data[L_power*imageb],N,&d_data[L_power*imagea],N,&beta,C3,N);
    cublasDestroy(handle);

    //help矩阵排列，直接在主机端排列好？还是kernel调用？
    for (i=0;i<L;i++){
        //image_1*L+i
        for (j=0;j<L;j++){
            //image_2*L+j
//            C[i*L+j]=(C[i*L+j]+L*help[image_1*3].y*help[image_2*3].y-xy-yx)/(N*z*z);
            C3[i*L+j]=(C3[i*L+j]+L*d_mean[imagea*L+i]*d_mean[imageb*L+j]-d_sum[imagea*L+i]*d_mean[imageb*L+j]-d_mean[imagea*L+i]*d_sum[imageb*L+j])/(N*d_stdv[imagea*L+i]*d_stdv[imageb*L+j]);
        }
    }

     //分配flambda网格，让C3就地接受结果
//     flambda();
     //C3作为参数，获得最大返回值参数，获得alpha_ij编号，并获得L中的序号。
    float max_value=C3[0];
    int max_index_i=0;
    int max_index_j=0;
    for (i=0;i<L;i++){
        for (j=0;j<L;j++){
        if (C3[i*L+j]>max_value){
            max_value=C3[i*L+j];
            max_index_i=i;
            max_index_j=j;
            }
        }
    }
    S[globalThreadID].value=max_value;
    S[globalThreadID].x=max_index_i;
    S[globalThreadID].y=max_index_j;

}

void wrapper_kernel(float *data,int N,int cml_size,float ***help,cml_retstruc *S){
    //wrapper_kernel前应该完成，数据打包成一个长数组
    //设置控制矩阵
    //读取数据接口，数值矩阵，，返回值矩阵，设置cuda环境，启动kernel
    //返回值矩阵，包括cml_matrix的value和坐标
    int control_size=N*(N-1)/2;
    int i,j;

//    int BLOCK_SIZE;//理论上没有上限
 //   int THREAD_PER_BLOCK;//(<512,根据显卡设备的cuda参数定)
    //配置控制矩阵，alpha_ij序数控制,ctr为alphaij序数
//    ctr = (int *)malloc(control_size);
    int *ctr;
    int *ctr_id1;
    int *ctr_id2;
    ctr= new int [control_size];
    ctr_id1 = new int [control_size];
    ctr_id2 = new int [control_size];
    for (i=0;i<control_size;i++){
        ctr[i]=i;
    }
    for (i=0;i<control_size;i++){
        for (j=i+1;j<control_size;j++){
//an error here ,the id is not eq i;should be modified later.
            ctr_id1[i] = i;
            ctr_id2[i] = j;
        }
    }
    //配置辅助矩阵help.拆分成三个数组，每个数组为N×L
    float *sum;
    float *mean;
    float *stdv;
    sum = new float [N*L];
    mean = new float [N*L];
    stdv = new float [N*L];
    for (i=0;i<N;i++){
        for (j=0;j<L;j++){
            sum[i*L+j]=help[i][j][0];
            mean[i*L+j]=help[i][j][1];
            stdv[i*L+j]=help[i][j][3];
        }
    }
    int *d_ctr;
    int *d_ctr_id1;
    int *d_ctr_id2;
    float *d_data;
    float *d_sum;
    float *d_mean;
    float *d_stdv;
    cml_retstruc *d_S;
    cudaMalloc((void **) &d_sum,sizeof(float)*N*L);
    cudaMalloc((void **) &d_mean,sizeof(float)*N*L);
    cudaMalloc((void **) &d_stdv,sizeof(float)*N*L);
    cudaMalloc((void **) &d_data,sizeof(float)*N*L*L);
    cudaMalloc((void **) &d_ctr,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id1,sizeof(int)*control_size);
    cudaMalloc((void **) &d_ctr_id2,sizeof(int)*control_size);

    cudaMalloc((void **) &d_S,sizeof(cml_retstruc)*control_size);

    cudaMemcpy(d_sum,sum,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,mean,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_stdv,stdv,N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr,ctr,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id1,ctr_id1,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctr_id2,ctr_id2,control_size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,data,N*L*L*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimGrid(control_size/500,1,1);
    dim3 dimBlock(500,1,1);
    parent_ncc_kernel<<<dimGrid,dimBlock>>>(d_data,d_ctr,d_ctr_id1,d_ctr_id2,d_sum,d_mean,d_stdv,N,d_S);
    cudaMemcpy(S,d_S,sizeof(cml_retstruc)*control_size,cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_stdv);
    cudaFree(d_ctr);
    cudaFree(d_ctr_id1);
    cudaFree(d_ctr_id2);
    cudaFree(d_S);
    delete[] sum;
    delete[] mean;
    delete[] stdv;
    delete[] ctr;
    delete[] ctr_id1;
    delete[] ctr_id2;
    //使用一个简单的kernel,不使用child kernel调用。
    //flambda需要的辅助矩阵，设置为线性格式

    //分配cuda内存，把数据矩阵、辅助矩阵存入

    //返回值矩阵，线性，分配内存

    //设置网格、线程参数，启动parent_ncc_kernel



}
int main(int argc ,char* argv[]){
    int oc;                     /*选项字符 */
    //char *b_opt_arg;            /*选项参数字串 */

//    int directreaddisk_flag=0;
    int cml_size=0;
    int N=-1;
    int N_noise=-1;
    int START=1;
    int version=-1;
    int hist_flag=0;
    int iteration=0;
    int debug_flag=0;
    int iteration_SIZE=0;
//    int noise_flag=0;
    char* filename;
    char* good_particle;
    char* logfile;
    char* noisefile;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:p:v:k:i:d:l:o:hN:")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
        case 'n':
            N=atoi(optarg);
            break;
        case 'N':
            noisefile=optarg;
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
        case 'i':
            iteration=1;
            iteration_SIZE=atoi(optarg);
            break;
        case 'd':
            debug_flag=1;
            break;
        case 'l':
            good_particle=optarg;
            break;
        case 'o':
            logfile=optarg;
            break;
        case 'h':
            printf("CML_NONPART is a program that picks high quality particles and throw out non-particles\n");
            printf("Author:\tQian jiaqiang ,Fudan Univesity,15210700078@fudan.edu.cn\tHuangQiang Lab\n");
            printf("-s the particle size after dft and linear polar\n");
            printf("-n the number of particles\n");
            printf("-N the file contains noise images\n");
            printf("-f the mrcs file which contains the particles data\n");
            printf("-p the start position of particles in mrcs file,default 1\n");
            //printf("-v the calculate method we used,default -1,needn't changed");
            printf("-i if you use -i,then we will try to throw out non-particles\n");
            printf("-l your output filename,which will contain the particles\n");
            printf("example\n");
            printf("cml_noise -s 144 -n 0 -N ~/data/noise.mrc -f ~/data/data.mrc -i -l ~/output/particles -o ~/output/log\n");
            exit(EXIT_SUCCESS);
            break;
        }
    }



    if (!filename) {
        printf("-f filename's abstract path is needed\n");
        exit(EXIT_FAILURE);
    }
    if (!good_particle){
        printf("-l output file name 's abstract path is needed\n");
    }
    if (N==-1 or cml_size==0){
        printf("-n N and -s dft_cml_size are both needed,if N=0,then N=max\n");
        exit(EXIT_FAILURE);
    }
    if (!logfile){
        printf("-o logfile is needed.\n");
        exit(EXIT_FAILURE);
    }
    if (!noisefile){
        printf("-N noise file is needed.\n");
        exit(EXIT_FAILURE);
    }
    FILE *OUTFILE;
    OUTFILE=fopen(logfile,"a+");
    fprintf(OUTFILE,"cml_size\t%d\n",cml_size);
    fprintf(OUTFILE,"N\t%d\n",N);

//全局参数设定
    int dft_size=cml_size;
    int dft_size_pow=dft_size*dft_size;
    int T;
    T=72;
    int i,j,k;
    float sigma=180.0/float(T);
    FILE *f;
    FILE *outputfile;
    FILE *fnoise;
    char mybuf[1024];
    char mybuf2[1024];

//    f=fopen("/home/qjq/data/qjq_200_data","rb");
    f=fopen(filename,"rb");
    fnoise=fopen(noisefile,"rb");

    outputfile=fopen(good_particle,"a+");
    setvbuf(outputfile,mybuf,_IOLBF,1024);
    setvbuf(OUTFILE,mybuf2,_IOLBF,1024);
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);
    if (START-1<0 or (START-1+N)*dft_size_pow>filelength){
        printf("-p START can't L.E 0 or GREATER THAN filelength");
        exit(EXIT_FAILURE);
    }
    rewind(f);
    fprintf(OUTFILE,"length\t%ld\n",filelength);
    if (N==0){
        N=filelength/dft_size_pow/sizeof(float);
    }
    if (N>filelength/dft_size_pow/sizeof(float)){
        fprintf(OUTFILE,"N can't be larger than the max numofItem\n");
        exit(EXIT_FAILURE);
    }

    long noisefilelength;
    fseek(fnoise,0,SEEK_END);
    noisefilelength=ftell(fnoise);
    rewind(fnoise);
    N_noise=noisefilelength/dft_size_pow/sizeof(float);
    int List_Noise[N_noise];
    for (i=0;i<N_noise;i++){
        List_Noise[i]=i;
    }

    int t_start,t_read_file,t_ncc_value,t_end,t_all_end,t_vote_1,t_vote_2,t_vote_3;
    struct timeval tsBegin,tsEnd;
    t_start=time(NULL);


//    std::vector<int> Global_good_particle;
//    int alpha_ij;
    int iteration_size;
    if (iteration==1){
        //可以选择读取所有粒子数据到硬盘，也可以选择每次单独读取，先选择每次单独读取，节约内存资源
        iteration_size=iteration_SIZE;
        int n_iteration;

        int last_iteration=N%iteration_size;
        if (last_iteration==0){
            n_iteration=N/iteration_size;
        }
        else {
            n_iteration=N/iteration_size+1;
        }
        fprintf(OUTFILE,"n_iteration \t%d\n",n_iteration);
        int control_struct[n_iteration][2];
        //control_struct[local_start][this iteration's particles num]切分数据集，每个数据集为iteration_size
        for (int t=0;t<n_iteration;t++){
            control_struct[t][0]=iteration_size*t;//记录这个划分的第一个粒子的位置，编号-1;
            control_struct[t][1]=iteration_size;//这个子集的粒子数量;
        }
        if (last_iteration!=0){
        control_struct[n_iteration-1][1]=last_iteration;
        }


        int control=0;
        for  (control=0;control<n_iteration;control++){//在每次control中，完成iteration_size的计算
            //初始化cml矩阵
            int local_N=control_struct[control][1];//这次control中的粒子数量
            int local_start=control_struct[control][0]*dft_size_pow;//文件中，这次control的粒子的起始位置
            int double_local_N=2*local_N;
//            int local_size_index=local_size*(local_size-1)/2;
            int *cml_pair_matrix[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i]= new int[double_local_N];
                }
            float *cml_pair_matrix_help[double_local_N];
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix_help[i] = new float [double_local_N];
                }
                //初始化数据集
            long malloc_size=double_local_N*dft_size_pow;
            float *lineardft_matrix=new float[malloc_size];
            fseek(f,local_start*sizeof(float),SEEK_SET);
            fread(lineardft_matrix,sizeof(float),local_N*dft_size_pow,f);
            //读入Noise文件，将Noise文件乱序，取和local_N等量的噪音图像
            std::random_shuffle(List_Noise,List_Noise+N_noise);
            for (i=0;i<local_N;i++){
                fseek(fnoise,List_Noise[i]*dft_size_pow*sizeof(float),SEEK_SET);
                fread(&lineardft_matrix[(local_N+i)*dft_size_pow],sizeof(float),dft_size_pow,fnoise);
            }
            fprintf(OUTFILE,"read finished\n");

            //维护混合数据List，记录粒子信息,(？不需要维护局部表）
//            int maintain_list[double_local_N];
            //局部粒子矩阵，不需要做粒子数据偏移，但对粒子的序号在输出时要做偏移
            //计算cml_pair_matrix 旧方法,a cblas version
            if (version == 0){
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
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
                for (i=0;i<local_N;i++){
                    for (j=0;j<local_N;j++){
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
            t_start=time(NULL);
            //the fastest cpu version
            if (version == -1){
            //cml_pair_matrix 新方法
                float **total_nccq[double_local_N];
            //    float ;
            //    total_nccq = new float** [N];
                int postion;
                for (i=0;i<double_local_N;i++){
                    total_nccq[i] = new float* [dft_size];
                }
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        total_nccq[i][j] = new float[4];
                    }
                }
            //    printf("000005\n");
//                gettimeofday(&tsBegin,NULL);
                for (i=0;i<double_local_N;i++){
            //        #pragma omp parallel for,can't openmp here
                    for (j=0;j<dft_size;j++){
                        postion=i*dft_size_pow+j*dft_size;
            //            printf("000006\n");
//                        total_nccq[i][j][0] = cblas_sasum( dft_size, &lineardft_matrix[postion], 1);//sum
                        total_nccq[i][j][0] = CMLNCV::MYSUM(dft_size,&lineardft_matrix[postion]);//sum
            //            printf("000007\n");
                        total_nccq[i][j][1] = total_nccq[i][j][0] / dft_size;//mean
            //            printf("000008\n");
                        total_nccq[i][j][2] = cblas_sdot( dft_size, &lineardft_matrix[postion], 1,&lineardft_matrix[postion],1);//dot
            //            printf("000009\n");
                        total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + dft_size*total_nccq[i][j][1]*total_nccq[i][j][1] - 2*total_nccq[i][j][0]*total_nccq[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
                    }
                }


//calculate ncc with cpu
                for (i=0;i<double_local_N;i++){
//                #pragma omp parallel for
                    for (j=i+1;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
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
//calculate ncc with gpu
		cml_retstruc *S;
                S = new cml_retstruc[double_local_N*(double_local_N-1)/2];
                wrapper_kernel(lineardft_matrix,double_local_N,dft_size,total_nccq,S);
                for (i=0;i<double_local_N;i++){
                    for (j=i+1;j<double_local_N;j++){
                        if (S[(2*double_local_N-1-i)*i/2+j-(i+1)].value>0.5){
                        cml_pair_matrix_help[i][j]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].x;
                        cml_pair_matrix_help[j][i]=S[(2*double_local_N-1-i)*i/2+j-(i+1)].y;
                        }
                    
                    else{
                        cml_pair_matrix_help[i][j]=-1;
                        cml_pair_matrix_help[j][i]=-1;
                    }}
                }
                for (i=0;i<double_local_N;i++){
                    cml_pair_matrix[i][i]=-1;
                    cml_pair_matrix_help[i][i]=-1;
                }
                //test GPU with cpu
                float diff=0.0f;
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<double_local_N;j++){
                        diff+=(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j])*(cml_pair_matrix[i][j]-cml_pair_matrix_help[i][j]);
                    }
                }
                diff=sqrt(diff/double_local_N/double_local_N);
                printf("diff between gpu_ncc with cpu_ncc\t%f\n",diff);
                //test
                /*
                for (i=0;i<double_local_N;i++){
//                #pragma omp parallel for
                    for (j=0;j<double_local_N;j++){
//                    for (j=i+1;j<double_local_N;j++){
                        if (i==j){
                            cml_pair_matrix[i][j]=-1;
                        }
                        else {
                            cmlncv_tuple tmp;
            //                tmp=CMLNCV::NCC_value(&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            tmp=CMLNCV::NCC_QT(total_nccq[i],total_nccq[j],&lineardft_matrix[i*dft_size_pow],&lineardft_matrix[j*dft_size_pow],dft_size);
                            if (cml_pair_matrix[i][j]!=tmp.x) printf("error\n");
                            if (cml_pair_matrix[j][i]!=tmp.y) printf("error\n");
                        }
                    }
                }
                */
                //NCC计算完成，所有common line被算出，释放计算辅助的数据存储矩阵
                for (i=0;i<double_local_N;i++){
                    for (j=0;j<dft_size;j++){
                        delete[] total_nccq[i][j];
                        }
                    }
                for (i=0;i<double_local_N;i++){
                        delete[] total_nccq[i];
                    }

            }
            fprintf(OUTFILE,"ncc cal finished\n");
            t_ncc_value=time(NULL);

            float *hist_peak =  new float[double_local_N*(double_local_N-1)/2];
            int *hist_index = new int[double_local_N*(double_local_N-1)/2];
            float half_pow_pi=sqrt(2*M_PI)*sigma;
//            float four_sigma_pow=4*sigma*sigma;
            float two_sigma_pow=2*sigma*sigma;
//            float alpha_t_alpha12;
            float cons=2*M_PI/dft_size;
            float Trecurse=180.0/T;
            //开始voting算法，先计算一遍voting，算出hist数组、hist_index数组

            //combine the voting and peak
//            for (i=0;i<local_N;i++){
//                        for (j=i+1;j<local_N;j++){
//                            //be sure ij,ji!=-1;

//                            alpha_ij=((2*local_N-1-i)*i/2+j-(i+1));
//                            float tmp_voting[local_N];
//                            float tmp_hist[72]={0.0};
//                            #pragma omp parallel for
//                            for (k=0;k<local_N;k++){
//                                //....#
//                //                alpha_ij=((2*N-1-i)*i/2+j-(i+1))*N+k;this is the error that caused difference between cml_dcv and cml_va
//                                if (k!=i and k!=j and cml_pair_matrix[i][j]>-1 and cml_pair_matrix[i][k]>-1 and cml_pair_matrix[j][i]>-1 and cml_pair_matrix[j][k]>-1 and cml_pair_matrix[k][i]>-1 and cml_pair_matrix[k][j]>-1) {
//                                    tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
//                                }
//                                else {
//                                    tmp_voting[k]=-10.0;
//                                }
//                            }

//                            for (int m=0;m<local_N;m++){

//                                    float tmp=tmp_voting[m];
//                                    if (tmp!=-10.0 and tmp!=-9.0){
//            #pragma omp parallel for
//                                    for (int l=0;l<T;l++){
//                                        float alpha_t_alpha12=(180.0*l/float(T))-tmp;
//                                        tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/(four_sigma_pow))/half_pow_pi;
//                                    }
//                                }
//                            }
//                            hist_peak[alpha_ij]=CMLNCV::max_float(tmp_hist,T);
//                           hist_index[alpha_ij]=CMLNCV::max_float_index(tmp_hist,T);

//                        }
//                    }


            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    long int alpha_ij=((2*double_local_N-1-i)*i/2+j-(i+1));
                    if (cml_pair_matrix[i][j]!=-1){
                        float tmp_voting[double_local_N];
                        float tmp_hist[72]={0.0};
#pragma omp parallel for
                        for (k=0;k<double_local_N;k++){
                            if (k!=i and k!=j){
                                if (cml_pair_matrix[i][k]!=-1 and cml_pair_matrix[j][k]!=-1){
                                    tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
                                }
                                else {tmp_voting[k]=-10.0;}
                            }
                            else {tmp_voting[k]=-10.0;}
                        }
                        for (int m=0;m<double_local_N;m++){
                            float tmp=tmp_voting[m];
                            if (tmp!=-10.0){
#pragma omp parallel for
                                for (int l=0;l<T;l++){
                                    float alpha_t_alpha12=Trecurse*l-tmp;
                                    tmp_hist[l]=tmp_hist[l]+exp(-1.0*alpha_t_alpha12*alpha_t_alpha12/two_sigma_pow)/half_pow_pi;
                                }
                            }
                        }
                        hist_peak[alpha_ij]=CMLNCV::max_float(tmp_hist,T);
                    }
                    else{
                    hist_peak[alpha_ij]=0.0;
                    }
                }
            }
            t_vote_1=time(NULL);
            if (hist_flag){
            fprintf(outputfile,"alpha_ij\ti\tj\thist_index\tpeak_value\n");
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index=(2*double_local_N-1-i)*i/2+j-(i+1);
                    fprintf(outputfile,"alpha_ij\t%d\t%d\t%d\t%f\n",i,j,hist_index[index],hist_peak[index]);
                }
            }
            }
            fprintf(OUTFILE,"\n311\n");
            //从CML_Pair中找出优秀粒子保留，等于剔除non-particle
            //计算每一个alphaij的voting序号
            int NumOfHighPeak=0;

            //固定threshold值，从右侧开始取第一个峰值,先固定threshold
            //threshold 为r/sqrt（local_N),r为2、4、8等。排序取local_N*threshold个为符合条件者。
//            float max_hist_peak=CMLNCV::max_float(hist_peak,local_N*(local_N-1)/2);
            int r=4;

            //复制一个hist_peak的备份;
            float *hist_peak_cp =  new float[double_local_N*(double_local_N-1)/2];
            long hist_peak_size=double_local_N*(double_local_N-1)/2;
            int threshold_top=floor((1-(r/sqrt(double_local_N)))*hist_peak_size);
            for (i=0;i<hist_peak_size;i++){
                hist_peak_cp[i]=hist_peak[i];
            }
            sort(hist_peak_cp,hist_peak_cp+hist_peak_size);
            fprintf(OUTFILE,"threshold_top %d\n",threshold_top);
            float threshold=hist_peak_cp[threshold_top-1];
            delete[] hist_peak_cp;
            t_vote_2=time(NULL);
            int Result_voted[double_local_N];
            for (i=0;i<double_local_N;i++){
                Result_voted[i]=0;
            }
            //找出Peak值较高的CML_pair
            for (i=0;i<double_local_N;i++){
                for (j=i+1;j<double_local_N;j++){
                    int index = (2*double_local_N-1-i)*i/2+j-(i+1);
                    if (hist_peak[index]>threshold) {
//                        NumOfHighPeak = NumOfHighPeak +1;
                        Result_voted[i]=Result_voted[i]+1;
                        Result_voted[j]=Result_voted[j]+1;
                        }
                    }
                }
            float mean_noise=0.0;
            for (i=local_N;i<double_local_N;i++){
                mean_noise+=Result_voted[i];

            }
            mean_noise=mean_noise/local_N;
            fprintf(OUTFILE,"mean_noise\t%f\n",mean_noise);
            //排序result_voted
            std::vector<voted> V;
            for (i=0;i<double_local_N;i++){
                voted s;
                s.index=i;
                s.value=Result_voted[i];
                V.push_back(s);
            }
            std::sort(V.begin(),V.end(),comp);
            //输出排序值和结果,分两个，noise排位的平均值;随着排位上升，noise的比例
            for (i=0;i<double_local_N;i++){
                fprintf(outputfile,"%d\t%d\t%d\n",i,V[i].index,V[i].value);
            }
            int counter=0;

            for (i=0;i<double_local_N;i++){
                if (V[i].index>=local_N){
                    counter=counter+1;
                }
                if ((i+1)%100==0){
                    fprintf(OUTFILE,"%d\tcounter %d\trate\t%f\n",i,counter,counter/float(i+1));
                }
            }

            //取出高可信粒子
            /*
            const int step=sizeof(Result_voted)/sizeof(int);
            float threshold_voted=(*std::max_element(Result_voted,Result_voted+step))*0.2;
            fprintf(OUTFILE,"threshold_voted is %f\n",threshold_voted);
//            float threshold_voting=*std::max_element(Result_voting,Reslut_voting+step)*0.8;
            //only voted particles are trusted.
            std::vector<int> local_good_particle;
            for (i=0;i<double_local_N;i++){
                if (Result_voted[i]>threshold_voted){
                    int good_voted_particle=control_struct[control][0]+i;
                    local_good_particle.push_back(good_voted_particle);
                }
            }



            if (local_good_particle.size()==0){
                fprintf(outputfile,"the local_good_particle num is 0\n");
            }
            for (auto m:local_good_particle){
                fprintf(outputfile,"%d\n",m);
            }
            */
            //放入全局Global_good_particle中
//            for (auto m:local_good_particle){
//                Global_good_particle.push_back(m);
//            }

            //销毁所有堆
            delete[] lineardft_matrix;
            delete[] hist_index;
            delete[] hist_peak;
            for (i=0;i<double_local_N;i++){
                delete[] cml_pair_matrix[i];
                delete[] cml_pair_matrix_help[i];
            }
            t_end=time(NULL);
            fprintf(OUTFILE,"ncc_time %d\n",t_ncc_value-t_start);
            fprintf(OUTFILE,"voting time %d\n",t_end-t_ncc_value);
            fprintf(OUTFILE,"only voting time %d\n",t_vote_1-t_ncc_value);
            fprintf(OUTFILE,"sort time %d\n",t_vote_2-t_vote_1);
            fprintf(OUTFILE,"%d/%d\tcompleted\n",control,n_iteration);

        }

}

        fclose(f);
        fclose(outputfile);
        fclose(fnoise);
        fclose(OUTFILE);






        t_all_end=time(NULL);
        printf("all time %dhour\n",(t_all_end-t_start)/3600);

        return 0;
}
