#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "matrix.hpp"
using namespace robotics;


namespace classification{
    template<typename T, size_t ROWS, size_t COLS>
    Matrix<T,ROWS,COLS> sigmoid(const Matrix<T,ROWS,COLS>& other){
        Matrix<T,ROWS,COLS> sigmoid_;
        for(size_t i=0;i<ROWS;++i){
            for(size_t j=0;j<COLS;++j){
                    sigmoid_.at(i,j) = 1/(1+ exp(-other.at(i,j)));
            }
        } 
        return sigmoid_;
    }

    template<typename T, size_t ROWS, size_t COLS>
    Matrix<T,ROWS,1>  sigmoidFunction(const Matrix<T,ROWS,COLS>& other,const Matrix<T,COLS,1>& w, T b){
        return sigmoid(other*w+b);
    }

    template<typename T, size_t ROWS, size_t COLS>
    T compute_loss(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b){

        //𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖))
        //std::log()
        // CHECK values of y they must either 1 or 0 no other value is accepted

        T loss;
        Matrix<T,ROWS,1> sigmoidMatrix = sigmoidFunction(X,w,b); //#TODO here are two FORS optimize!

        for(size_t i=0;i<ROWS;++i){
            //} //#TODO check y is 1 or zero 
            loss += -y.at(i,0)*std::log(sigmoidMatrix.at(i,0))-(1-y.at(i,0))*(std::log(1-sigmoidMatrix.at(i,0)));

        } 
        loss/= static_cast<T>(ROWS);
        return loss;
    }

    template<typename T, size_t ROWS, size_t COLS>
    T compute_loss_reg(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b, T lambda=1.0){

        //𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖))
        //std::log()
        // CHECK values of y they must either 1 or 0 no other value is accepted

        T loss;
        Matrix<T,ROWS,1> sigmoidMatrix = sigmoidFunction(X,w,b); //#TODO here are two FORS optimize!

        for(size_t i=0;i<ROWS;++i){
            //} //#TODO check y is 1 or zero 
            loss += -y.at(i,0)*std::log(sigmoidMatrix.at(i,0))-(1-y.at(i,0))*(std::log(1-sigmoidMatrix.at(i,0)));

        } 
        loss/= static_cast<T>(ROWS); // only ROWS 

        Matrix<T,1,1> regularization_term = (w.getTransposed()*w)*(lambda)/(static_cast<T>(2*ROWS));


        return loss+regularization_term.at(0,0);
    }



    template<typename T, size_t ROWS, size_t COLS>
    std::pair<T, Matrix<T, COLS, 1>> compute_gradient(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b){
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

        Matrix<T,ROWS,1> sigmoidMatrix = sigmoidFunction(X,w,b); // #TODO too many foor loops maybe a way to optimize in only one loop or for each for parallelism
        Matrix<T,ROWS,1> matrix_error = sigmoidMatrix-y;
        Matrix<T, COLS, 1> DJ_DW = (( X.getTransposed() )* matrix_error)/static_cast<T>(ROWS);

        auto DJ_DB = ( ( Matrix<double,1,ROWS>(1) )* matrix_error ) / static_cast<T>(ROWS);

        std::pair<T, Matrix<T, COLS, 1>> gradientDescentParameters(DJ_DB.at(0,0), std::move(DJ_DW));

        return gradientDescentParameters;


    }   


    template<typename T, size_t ROWS, size_t COLS>
    std::pair<T, Matrix<T, COLS, 1>> compute_gradient_reg(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b, T lambda=1.0){

        T rows_T = static_cast<T>(ROWS);
        Matrix<T,ROWS,1> sigmoidMatrix = sigmoidFunction(X,w,b); // #TODO too many foor loops maybe a way to optimize in only one loop or for each for parallelism
        Matrix<T,ROWS,1> matrix_error = sigmoidMatrix-y;
        Matrix<T, COLS, 1> DJ_DW = (( X.getTransposed() )* matrix_error)/rows_T;

        DJ_DW += w*lambda/rows_T;

        auto DJ_DB = ( ( Matrix<double,1,ROWS>(1) )* matrix_error ) / rows_T;

        std::pair<T, Matrix<T, COLS, 1>> gradientDescentParameters(DJ_DB.at(0,0), std::move(DJ_DW));

        return gradientDescentParameters;

    }


    template<typename T, size_t ROWS, size_t COLS>
    std::pair<T, Matrix<T, COLS, 1>>  gradient_descent(Matrix<T,ROWS,COLS>& X,const Matrix<T,ROWS,1>& y,const Matrix<T,COLS,1>& w, T b, T alpha, int num_iterations) {    
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
            auto lRP = compute_gradient(X,y,w_copy,b_copy);  // first dj_db, second dj_dw
            w_copy = w_copy - lRP.second*alpha;
            b_copy = b_copy - lRP.first*alpha;

            J_history.emplace_back(compute_loss(X,y,w_copy,b_copy));

            if (i%1000==0){
                //std::cout << " COST at " << i << " " << J_history.at(i) <<std::endl; 
                //std::cout << " w_copy " << w_copy <<std::endl; 
            }

        }
        std::pair<T, Matrix<T, COLS, 1>> logisticParameters(b_copy, std::move(w_copy));
        return logisticParameters;

    }


}

#endif