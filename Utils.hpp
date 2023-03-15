#ifndef UTILS_H
#define UTILS_H

#include <ostream>
//#include "matrix.hpp"

//forward declarations of matrix.hpp in order to use utils.hpp 
namespace robotics{
    template<typename T,size_t ROWS, size_t COLS>
    class Matrix;

    template<typename T,size_t ROWS, size_t COLS>
    std::ostream& operator<<(std::ostream& stream, const Matrix<T,ROWS,COLS>& other){
        for(size_t i=0;i<ROWS;++i){
            for(size_t j=0;j<COLS;++j){
                stream << other.at(i,j) << " ";
            }
            stream << "\n";
        }
        return stream;
    }

}

#endif