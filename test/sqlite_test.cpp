
#include "sqlite3.h"
#include <iostream>
#include <sstream>

using namespace std;
sqlite3 * pDB;
int createTable()
{
    char* errMsg;
    std::string dropTab = "drop table test_tab;";
    string strSQL= "create table test_tab (f1 int, f2 long);";

    int res= sqlite3_exec(pDB , dropTab.c_str() , 0 , 0 , &errMsg);

    if (res != SQLITE_OK)
    {
        std::cout << "执行SQL 出错." << errMsg << std::endl;
        return -1;
    }
    res = sqlite3_exec(pDB , strSQL.c_str() ,0 ,0, &errMsg);

    if (res != SQLITE_OK)
    {
        std::cout << "执行创建table的SQL 出错." << errMsg << std::endl;
        return -1;
    }
    else
    {
        std::cout << "创建table的SQL成功执行."<< std::endl;
    }

    return 0;
}

int insert1()
{
    char* errMsg;

    int res = sqlite3_exec(pDB,"begin transaction;",0,0, &errMsg);

    for (int i= 1; i < 10; ++i)
    {
        std::stringstream strsql;
        strsql << "insert into test_tab  values(";
        strsql  << i << ","<< (i+10) << ");";
        std::string str = strsql.str();
        res = sqlite3_exec(pDB,str.c_str(),0,0, &errMsg);
        if (res != SQLITE_OK)
        {
            std::cout << "执行SQL 出错." << errMsg << std::endl;
            return -1;
        }
    }

    res = sqlite3_exec(pDB,"commit transaction;",0,0, &errMsg);

    std::cout << "SQL成功执行."<< std::endl;


    return 0;
}

static int callback(void *NotUsed, int argc, char **argv, char **azColName)
{

    for(int i = 0 ; i < argc ; i++)
    {
        std::cout << azColName[i] << " = " << (argv[i] ? argv[i] : "NULL") << ", " ;
    }

    std::cout<< "\n";
    return 0;
}

int select1()
{
    char* errMsg;
    string strSQL= "select * from test_tab;";

    int res = sqlite3_exec(pDB, strSQL.c_str(), callback , 0 , &errMsg);

    if (res != SQLITE_OK)
    {
        std::cout << "执行SQL 出错." << errMsg << std::endl;
        return -1;
    }
    else
    {
        std::cout << "SQL成功执行."<< std::endl;
    }

    return 0;
}

int main()
{
    int res = sqlite3_open("/home/pcchange/changephilip/sqlite_test/sql.db", &pDB);

    if( res ){
        std::cout << "Can't open database: "<< sqlite3_errmsg(pDB);
        sqlite3_close(pDB);
        return -1;
    }
    res = createTable();
    if (res != 0)
    {
        return 0;
    }
    res = insert1();
    if (res != 0)
    {
        return 0;
    }
    select1();

    return 0;
}
