#include <iostream>
//#include <openmpi/mpi.h>
int main()
{
  #pragma omp parallel
  {
    std::cout << "Hello World!\n";
  }
}
