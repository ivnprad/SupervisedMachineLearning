#ifndef SUPERVISEDLEARNING_H
#define SUPERVISEDLEARNING_H


#include "matrix.hpp"


/**
 * ROWS  Training Examples
 * COLS  Features
**/

using namespace robotics;

template<typename T, size_t ROWS, size_t COLS>
T compute_cost(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b){
    /*
        compute cost
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter
    Returns:
    cost (scalar): cost
    */
    T cost;
    Matrix<T,ROWS,1> matrix_error = X*w+b-y;

    for (size_t row_idx=0;row_idx<matrix_error.getRows();++row_idx){
        cost +=  std::pow(matrix_error.at(row_idx,0), 2);
    }
    return cost/(2*ROWS);
}


template<typename T, size_t ROWS, size_t COLS>
std::pair<T, robotics::Matrix<T, COLS, 1>> compute_gradient(robotics::Matrix<T,ROWS,COLS>& X,const robotics::Matrix<T,ROWS,1>& y,const robotics::Matrix<T,COLS,1>& w, T b){
    /*
        """
        Computes the gradient for linear regression
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter
        Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
        dj_db (scalar):
        """
    */
    
    robotics::Matrix<T,ROWS,1> matrix_error = X*w+b-y;
    robotics::Matrix<T, COLS, 1> DJ_DW = (( X.getTransposed() )* matrix_error)/static_cast<T>(ROWS);

    auto DJ_DB = ( ( robotics::Matrix<double,1,ROWS>(1) )* matrix_error ) / static_cast<T>(ROWS);

    std::pair<T, robotics::Matrix<T, COLS, 1>> gradientDescentParameters(DJ_DB.at(0,0), std::move(DJ_DW));

   return gradientDescentParameters;


}


template<typename T, size_t ROWS, size_t COLS, typename lambdaComputeCost, typename lambdaComputeGradient>
std::pair<T, Matrix<T, COLS, 1>>  gradient_descent(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b, T alpha, int num_iterations, lambdaComputeCost computeCostFunc, lambdaComputeGradient computeGradientFunc) {    
    /*
    """
        Performs batch gradient descent to learn w and b. Updates w and b by taking
        num_iters gradient steps with learning rate alpha
        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters
    b_in (scalar)
    cost_function
    gradient_function
    alpha (float)
    num_iters (int)
    : initial model parameter
    : function to compute cost
    : function to compute the gradient
    : Learning rate
    : number of iterations to run gradient descent
        Returns:
        w (ndarray (n,)) : Updated values of parameters
        b (scalar)       : Updated value of parameter
        """
    */
    std::vector<T> J_history;
    auto w_copy = w;
    T b_copy = b;

    for(size_t i=0; i<num_iterations;++i){
        //# Calculate the gradient and update the parameters
        auto lRP = computeGradientFunc(X,y,w_copy,b_copy);  // first dj_db, second dj_dw
        w_copy = w_copy - lRP.second*alpha;
        b_copy = b_copy - lRP.first*alpha;

        J_history.emplace_back(computeCostFunc(X,y,w_copy,b_copy));

        if (i%100==0){
            std::cout << " COST at " << i << " " << J_history.at(i) <<std::endl; 
        }
    }

    std::pair<T, Matrix<T, COLS, 1>> linearRegressionParameters(b_copy, std::move(w_copy));
    return linearRegressionParameters;

}

/******  FEATURE SCALING ************
 * to choose learning rate start with a small number and keep increasing 3X  
 */
template<typename T, size_t ROWS, size_t COLS>
Matrix<T,1,COLS> meanColumnwise(const Matrix<T,ROWS,COLS>& other) {
    Matrix<T,1,COLS> itsMean;
    for(size_t j=0;j<COLS;++j){
        T sum=0;
        for(size_t i=0;i<ROWS;++i){
                sum+= other.at(i,j);
        }
        itsMean.at(0,j) = sum/static_cast<T>(ROWS);
    }
    return itsMean;
}  

template<typename T, size_t ROWS, size_t COLS>
Matrix<T,1,COLS> standardDeviationColumnwise(const Matrix<T,ROWS,COLS>& other, const Matrix<T,1,COLS>& meanColumnwise_) {
    Matrix<T,1,COLS> itsStandardDeviation;
    for(size_t j=0;j<COLS;++j){
        T sum_of_squares=0;
        for(size_t i=0;i<ROWS;++i){
            sum_of_squares +=  std::pow(other.at(i,j)-meanColumnwise_.at(0,j),2);
        }
        itsStandardDeviation.at(0,j) =  std::sqrt(sum_of_squares/static_cast<T>(ROWS));
    }

    return itsStandardDeviation;
}  

template<typename T, size_t ROWS, size_t COLS>
Matrix<T,ROWS,COLS> getXNormalizedByZScore(const Matrix<T,ROWS,COLS>& other, const Matrix<T,1,COLS>& mean, const Matrix<T,1,COLS>& standardDeviation) 
    {
    Matrix<T,ROWS,COLS> Xnormalized;
    for(size_t i=0;i<ROWS;++i){
        for(size_t j=0;j<COLS;++j){
            Xnormalized.at(i,j) = (other.at(i,j)-mean.at(0,j))/(standardDeviation.at(0,j));
        }
    }
    return Xnormalized;
}  

template<typename T, size_t ROWS, size_t COLS>
std::tuple<Matrix<T,ROWS,COLS>, Matrix<T,1,COLS>, Matrix<T,1,COLS>> zscore_normalize_features(const Matrix<T,ROWS,COLS>& X){

    Matrix<T,1,COLS> mean = meanColumnwise(X);
    Matrix<T,1,COLS> stdDev = standardDeviationColumnwise(X,mean);
    Matrix<T,ROWS,COLS> Xnormalized = getXNormalizedByZScore(X,mean,stdDev);
    return std::make_tuple(Xnormalized, mean, stdDev);

}

#endif 
