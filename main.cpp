#include <iostream>
#include <iomanip>
#include <cmath>
#include "matrix.hpp"
#include <map>
#include <functional>
#include "data.hpp"
#include <tuple>
#include "SupervisedLearning.hpp"
#include "Classification.hpp"

using namespace robotics;

int main()
{
    /*******MATRIX OPERATIONS ************************/
    
    Matrix<double, 1, 1> jolin= {{1.0}};
    Matrix<int,1,1> singleMat({{1}});
    Matrix<int, 3, 3> mat2({{1, 2, 3},{4, 5, 6},{7, 8, 10}});
    mat2+3.0;
    std::cout << mat2<< std::endl;
    int det1 = mat2.getDeterminant();
    auto adjoint  = mat2.getAdjoint();
    std::cout << "determinant " <<  mat2.getDeterminant() << std::endl;
    std::cout << " adjugate " << mat2.getAdjugate() << std::endl;
    std::cout << " inverse " << mat2.getInverse() << std::endl;
    std::cout << " conver to double "<<  robotics::convert<double>(mat2)<< std::endl;


    /******** LINEAR REGRESSSION WITH MULTIPLE VARIABLES *******/

    Matrix<double, 3, 4> X_train({{2104, 5, 1, 45},{1416, 3, 2, 40},{852, 2, 1, 35}});
    Matrix<double, 3, 1> y_train({{460},{232}, {178} });
    Matrix<double, 4, 1> w_init({{0.39133535},{18.75376741},{-53.36032453},{-26.42131618}});

    double b_init = 785.1811367994083;
    double alpha = 5.0e-7;
    int iterations = 1000;

    auto computeCostFunc = [&](auto& X,auto& y, auto& w, auto& b){ return compute_cost(X,y,w,b); };     // Lambda is amazing!!!!
    auto computeGradientFunc = [&](auto& X,auto& y, auto& w, auto& b){ return compute_gradient(X,y,w,b); };

    Matrix<double, 4, 1> init_W({{0.0},{0.0},{0.0},{0.0}});
    double init_B = 0.0;

    auto gradientDescentParameters = gradient_descent(X_train,y_train,init_W,init_B,alpha,iterations, computeCostFunc,computeGradientFunc);

    std::cout << " gradientDescentParameters final w" << std::endl;
    std::cout << gradientDescentParameters.second << std::endl;
    std::cout << " gradientDescentParameters.second b " << std::endl;
    std::cout << gradientDescentParameters.first << std::endl;

    // Optional Lab_ Multiple linear regression PDF
    //_ Courseraexpected values 426.185,286.167 ,171.468
    std::cout << " prediction " << std::endl; 
    std::cout <<   X_train*gradientDescentParameters.second + gradientDescentParameters.first << std::endl;
    
    std::cout << " target value  " << std::endl;
    std::cout << y_train  << std::endl;

    gradientDescentParameters = gradient_descent(X_train_2,y_target_2.getTransposed(),init_W,init_B,9.9e-7,10, computeCostFunc,computeGradientFunc);
    gradientDescentParameters = gradient_descent(X_train_2,y_target_2.getTransposed(),init_W,init_B,9e-7,10, computeCostFunc,computeGradientFunc);
    gradientDescentParameters = gradient_descent(X_train_2,y_target_2.getTransposed(),init_W,init_B,1e-7,10,computeCostFunc,computeGradientFunc);

    /**********************FEATURE SCALING *****************/
    
    auto result = zscore_normalize_features(X_train_2);// Normalize matrix and get results in a tuple
    auto Xnormalized = std::get<0>(result); // Extract mean, standard deviation, and normalized matrix from the tuple
    auto stdDev = std::get<2>(result);
    auto mean = std::get<1>(result);
    std::cout << "Mean: " << mean << std::endl; // Print results
    std::cout << "Standard deviation: " << stdDev << std::endl;
    auto gDP = gradient_descent(Xnormalized,y_target_2.getTransposed(),init_W,init_B,0.1,1000,computeCostFunc,computeGradientFunc);

    Matrix<double, 1, 4> X_house({{1200, 3, 1, 40}});
    auto X_house_norm = (X_house-mean)/stdDev;
    std::cout << " X_house_norm " << std::endl;
    std::cout << X_house_norm << std::endl; // expected -0.530851 0.433809 -0.789272 0.0626957 
    auto x_house_predicted = X_house_norm*gDP.second+gDP.first;
    x_house_predicted*=1000;
    std::cout << " predicted price " << std::endl; // expected 318709
    std::cout << x_house_predicted<< std::endl;
    

    // LOGISTIC REGRESSION


    Matrix<double, 6, 2> X_train_lg({{0.5, 1.5},{1,1},{1.5, 0.5},{3, 0.5},{2, 2},{1, 2.5}});
    Matrix<double, 6, 1> y_train_lg({{0},{0}, {0},{1},{1},{1} });

    Matrix<double, 2, 1>  w_tmp({{1},{1}});
    //double b_tmp = -3;
    //double itsLoss = compute_loss(X_train_lg,y_train_lg,w_tmp,b_tmp);
    //double itsLoss_4 = compute_loss(X_train_lg,y_train_lg,w_tmp,-4.0);

    std::cout << "loss -3 " << classification::compute_loss(X_train_lg,y_train_lg,w_tmp,-3.0) << std::endl;
    std::cout << "loss -4 " << classification::compute_loss(X_train_lg,y_train_lg,w_tmp,-4.0) << std::endl;

// X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
// y_tmp = np.array([0, 0, 0, 1, 1, 1])
// w_tmp = np.array([2.,3.])
// b_tmp = 1.

    w_tmp = {{2.0},{3.0}};
    double b_tmp = 1;

    auto gradientValues = classification::compute_gradient(X_train_lg,y_train_lg,w_tmp,b_tmp);

    std::cout << " gradient dj_db" << std::endl;
    std::cout << gradientValues.first << std::endl;
    //std::cout << gradientValues.second << std::endl;

    std::cout << " gradient dj_dw" << std::endl;
    std::cout << gradientValues.second << std::endl;

    w_tmp = {{0.0},{0.0}};
    b_tmp = 0.0;
    double alpha_lg = 0.1;
    double iters = 10000;

    auto gradientValues_lg = classification::gradient_descent(X_train_lg,y_train_lg,w_tmp,b_tmp,alpha_lg,iters);

    std::cout << " gradientValues_lg " << std::endl;
    std::cout << gradientValues_lg.first << std::endl;
    //std::cout << gradientValues.second << std::endl;

    std::cout << " gradientValues_lg " << std::endl;
    std::cout << gradientValues_lg.second << std::endl; 

    // probability equal 0.5 = 1(1+e^(x*w+b))  BOUNDARY LINE
    // 1=e^(x*w+b)
    // X*w+b=0
    // for x1=0 ->  x2=b/w // the same for x2
    double x2= - gradientValues_lg.first/gradientValues_lg.second.at(1,0);
    double x1= - gradientValues_lg.first/gradientValues_lg.second.at(0,0);

    std::cout << " x1 "<<  x1 << std::endl;
    std::cout << " x2 "<<  x2 << std::endl;

    //std::cout << classification::sigmoidFunction(X_train_lg,gradientValues_lg.second,gradientValues_lg.first) << std::endl;

    /****REGULARIZATION***/
    std::cout << " regularization cost" << compute_cost_reg(X_train_reg,y_train_reg,w_reg,b_reg,lambda_reg) << std::endl;
    std::cout << " regularization cost" << classification::compute_loss_reg(X_train_reg,y_train_reg,w_reg,b_reg,lambda_reg) << std::endl;

    auto gradientValues_reg = compute_gradient_reg(X_reg,y_train_reg,w_reg_2,b_reg,lambda_reg);
    std::cout << " gradient dj_db" << std::endl;
    std::cout << gradientValues_reg.first << std::endl; // expected 0.6648774569425726
    std::cout << " gradient dj_dw" << std::endl;
    std::cout << gradientValues_reg.second << std::endl; // expected [0.29653214748822276, 0.4911679625918033, 0.21645877535865857]
    

    auto gradientValues_log_reg = classification::compute_gradient_reg(X_reg,y_train_reg,w_reg_2,b_reg,lambda_reg);
    std::cout << " gradient dj_db LOG" << std::endl;
    std::cout << gradientValues_log_reg.first << std::endl; // expected 0.6648774569425726
    std::cout << " gradient dj_dw LOG" << std::endl;
    std::cout << gradientValues_log_reg.second << std::endl; // expected [0.29653214748822276, 0.4911679625918033, 0.21645877535865857]
    


    return 0;  

}