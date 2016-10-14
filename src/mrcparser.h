#ifndef MRCPARSER_H
#define MRCPARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include<algorithm>
#include <boost/algorithm/string.hpp>

/*
    class MrcParser
{

public:
    MrcParser();
};
*/
struct mrcStarHead{
    std::vector<std::string> mrcsHead;
    int positionOfMicrographName;
    int postionOfCoordinateX;
    int postionOfCoordinateY;
};
struct mrcStarData{
    std::vector<std::vector<std::string>> dataDocker;
    std::vector<std::string> MicrographNameList;
};
//typedef std::vector<std::vector<std::string>> dataDocker;
//typedef std::vector<std::string> fileNameList;

namespace mrcStarParser{


        struct mrcStarHead parserMrcsHead(std::ifstream &file);
        struct mrcStarData mrcStarDataRead(std::ifstream &file,struct mrcStarHead);
        void elimDups(std::vector<std::string> &words);

       /*
        struct mrcStarHead parserMrcsHead(FILE *f);
        struct mrcStarData mrcStarDataRead(FILE *fin,struct mrcStarHead);
        void elimDups(std::vector<std::string> &words);
        */
}

#endif // MRCPARSER_H
