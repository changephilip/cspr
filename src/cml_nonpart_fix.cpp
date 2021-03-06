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
    int iteration=0;
    int debug_flag=0;
    char* filename;
    char* good_particle;
    char* logfile;
//    char* directdiskfile;
    printf("00001\n");
    while((oc = getopt(argc, argv, "s:n:f:p:v:k:id:l:o:h")) != -1)
    {
        switch(oc)
        {
        case 's':
            cml_size=atoi(optarg);
            break;
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
        case 'i':
            iteration=1;
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
            printf("-f the mrcs file which contains the particles data\n");
            printf("-p the start position of particles in mrcs file,default 1\n");
            //printf("-v the calculate method we used,default -1,needn't changed");
            printf("-i if you use -i,then we will try to throw out non-particles\n");
            printf("-l your output filename,which will contain the particles\n");
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
    char mybuf[1024];
    char mybuf2[1024];

//    f=fopen("/home/qjq/data/qjq_200_data","rb");
    f=fopen(filename,"rb");
    outputfile=fopen(good_particle,"a+");
    setvbuf(outputfile,mybuf,_IOLBF,1024);
    setvbuf(OUTFILE,mybuf2,_IOLBF,1024);
    long filelength;
    fseek(f,0,SEEK_END);
    filelength=ftell(f);
    if (START-1<0 or (START-1+N)*dft_size_pow>filelength){
        printf("-p START can't G.T 0 or GREATER THAN filelength");
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


    int t_start,t_read_file,t_ncc_value,t_end,t_all_end,t_vote_1,t_vote_2,t_vote_3;
    struct timeval tsBegin,tsEnd;
    t_start=time(NULL);


//    std::vector<int> Global_good_particle;
//    int alpha_ij;
    int iteration_size;
    if (iteration==1){
        //可以选择读取所有粒子数据到硬盘，也可以选择每次单独读取，先选择每次单独读取，节约内存资源
        iteration_size=1000;
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
        for (control=0;control<n_iteration;control++){//在每次control中，完成iteration_size的计算
            //初始化cml矩阵
            int local_N=control_struct[control][1];//这次control中的粒子数量
            int local_start=control_struct[control][0]*dft_size_pow;//文件中，这次control的粒子的起始位置
//            int local_size_index=local_size*(local_size-1)/2;
            int *cml_pair_matrix[local_N];
                for (i=0;i<local_N;i++){
                    cml_pair_matrix[i]= new int[local_N];
                }
                //初始化数据集
            long malloc_size=local_N*dft_size_pow;
            float *lineardft_matrix=new float[malloc_size];
            fseek(f,local_start*sizeof(float),SEEK_SET);
            fread(lineardft_matrix,sizeof(float),malloc_size,f);


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
                float **total_nccq[local_N];
            //    float ;
            //    total_nccq = new float** [N];
                int postion;
                for (i=0;i<local_N;i++){
                    total_nccq[i] = new float* [dft_size];
                }
                for (i=0;i<local_N;i++){
                    for (j=0;j<dft_size;j++){
                        total_nccq[i][j] = new float[4];
                    }
                }
            //    printf("000005\n");
//                gettimeofday(&tsBegin,NULL);
                for (i=0;i<local_N;i++){
            //        #pragma omp parallel for,can't openmp here
                    for (j=0;j<dft_size;j++){
                        postion=i*dft_size_pow+j*dft_size;
            //            printf("000006\n");
//                        total_nccq[i][j][0] = cblas_sasum( dft_size, &lineardft_matrix[postion], 1);//sum
                        total_nccq[i][j][0] = CMLNCV::MYSUM(dft_size,&lineardft_matrix[postion]);
            //            printf("000007\n");
                        total_nccq[i][j][1] = total_nccq[i][j][0] / dft_size;//mean
            //            printf("000008\n");
                        total_nccq[i][j][2] = cblas_sdot( dft_size, &lineardft_matrix[postion], 1,&lineardft_matrix[postion],1);//dot
            //            printf("000009\n");
                        total_nccq[i][j][3] = sqrt((total_nccq[i][j][2] + dft_size*total_nccq[i][j][1]*total_nccq[i][j][1] - 2*total_nccq[i][j][0]*total_nccq[i][j][1])/dft_size);//sigma=sqrt(dot+mean*mean*size-2*mean*sum)
                    }
                }

                for (i=0;i<local_N;i++){
                #pragma omp parallel for
                    for (j=0;j<local_N;j++){
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
                //NCC计算完成，所有common line被算出，释放计算辅助的数据存储矩阵
                for (i=0;i<local_N;i++){
                    for (j=0;j<dft_size;j++){
                        delete[] total_nccq[i][j];
                        }
                    }
                for (i=0;i<local_N;i++){
                        delete[] total_nccq[i];
                    }

            }

            t_ncc_value=time(NULL);

            float *hist_peak =  new float[local_N*(local_N-1)/2];
            int *hist_index = new int[local_N*(local_N-1)/2];
            float half_pow_pi=sqrt(M_PI*2)*sigma;
//            float four_sigma_pow=4*sigma*sigma;
            float two_sigma_pow=2*sigma*sigma;
//            float alpha_t_alpha12;
            float cons=M_PI*2/dft_size;
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


            for (i=0;i<local_N;i++){
                for (j=i+1;j<local_N;j++){
                    long int alpha_ij=((2*local_N-1-i)*i/2+j-(i+1));
                    if (cml_pair_matrix[i][j]!=-1){
                        float tmp_voting[local_N];
                        float tmp_hist[T]={0.0};
#pragma omp parallel for
                        for (k=0;k<local_N;k++){
                            if (k!=i and k!=j){
                                if (cml_pair_matrix[i][k]!=-1 and cml_pair_matrix[j][k]!=-1){
                                    tmp_voting[k]=CMLNCV::cvoting(cml_pair_matrix[i][j],cml_pair_matrix[i][k],cml_pair_matrix[j][i],cml_pair_matrix[j][k],cml_pair_matrix[k][i],cml_pair_matrix[k][j],cons);
                                }
                                else {tmp_voting[k]=-10.0;}
                            }
                            else {tmp_voting[k]=-10.0;}
                        }
                        for (int m=0;m<local_N;m++){
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
            for (i=0;i<local_N;i++){
                for (j=i+1;j<local_N;j++){
                    int index=(2*local_N-1-i)*i/2+j-(i+1);
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
            float *hist_peak_cp =  new float[local_N*(local_N-1)/2];
            long hist_peak_size=local_N*(local_N-1)/2;
            int threshold_top=floor((1-(r/sqrt(local_N)))*hist_peak_size);
            for (i=0;i<hist_peak_size;i++){
                hist_peak_cp[i]=hist_peak[i];
            }
            sort(hist_peak_cp,hist_peak_cp+hist_peak_size);
            fprintf(OUTFILE,"threshold_top %d\n",threshold_top);
            float threshold=hist_peak_cp[threshold_top-1];
            delete[] hist_peak_cp;
            t_vote_2=time(NULL);
            int Result_voted[local_N];
            for (i=0;i<local_N;i++){
                Result_voted[i]=0;
            }
            //找出Peak值较高的CML_pair
            for (i=0;i<local_N;i++){
                for (j=i+1;j<local_N;j++){
                    int index = (2*local_N-1-i)*i/2+j-(i+1);
                    if (hist_peak[index]>threshold) {
//                        NumOfHighPeak = NumOfHighPeak +1;
                        Result_voted[i]=Result_voted[i]+1;
                        Result_voted[j]=Result_voted[j]+1;
                        }
                    }
                }



            const int step=sizeof(Result_voted)/sizeof(int);
            float threshold_voted=(*std::max_element(Result_voted,Result_voted+step))*0.2;
            fprintf(OUTFILE,"threshold_voted is %f\n",threshold_voted);
//            float threshold_voting=*std::max_element(Result_voting,Reslut_voting+step)*0.8;
            //only voted particles are trusted.
            std::vector<int> local_good_particle;
            for (i=0;i<local_N;i++){
                if (Result_voted[i]>threshold_voted){
                    int good_voted_particle=control_struct[control][0]+i;
                    local_good_particle.push_back(good_voted_particle);
                }
            }
            t_end=time(NULL);
            if (local_good_particle.size()==0){
                fprintf(outputfile,"the local_good_particle num is 0\n");
            }
            for (auto m:local_good_particle){
                fprintf(outputfile,"%d\n",m);
            }
            //放入全局Global_good_particle中
//            for (auto m:local_good_particle){
//                Global_good_particle.push_back(m);
//            }

            //销毁所有堆
            delete[] lineardft_matrix;
            delete[] hist_index;
            delete[] hist_peak;
            for (i=0;i<local_N;i++){
                delete[] cml_pair_matrix[i];
            }
            fprintf(OUTFILE,"ncc_time %d\n",t_ncc_value-t_start);
            fprintf(OUTFILE,"voting time %d\n",t_end-t_ncc_value);
            fprintf(OUTFILE,"only voting time %d\n",t_vote_1-t_ncc_value);
            fprintf(OUTFILE,"sort time %d\n",t_vote_2-t_vote_1);
            fprintf(OUTFILE,"%d/%d\tcompleted\n",control,n_iteration);

        }

}

    fclose(f);
    fclose(outputfile);
    fclose(OUTFILE);






        t_all_end=time(NULL);
        printf("all time %dhour\n",(t_all_end-t_start)/3600);

        return 0;
}

