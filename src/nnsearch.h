
#include <RcppArmadillo.h>


using namespace std;

arma::umat make_candidates(const arma::mat& w, const arma::uvec& indsort, 
                           unsigned int col,
                           double rho);

arma::field<arma::uvec> neighbor_search(const arma::mat& w, double rho);
  
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset, const arma::mat& w, double rho);

arma::field<arma::uvec> radialndag(const arma::mat& w, double rho);

arma::umat sparse_struct(const arma::field<arma::uvec>& dag, int nr);


// predict

arma::umat make_candidates_testset(const arma::mat& w, 
                                   const arma::uvec& indsort, arma::uvec& testindsort,
                                   unsigned int col,
                                   double rho);

arma::field<arma::uvec> neighbor_search_testset(const arma::mat& wtrain, 
                                                const arma::mat& wtest, double rho);

arma::uvec sort_test(const arma::mat wtrain, const arma::mat wtest);

arma::field<arma::uvec> dagbuild_from_nn_testset(const arma::field<arma::uvec>& Rset, 
                                                 const arma::mat& wtrain, const arma::mat& wtest);

arma::field<arma::uvec> radgpbuild_testset(const arma::mat& wtrain,
                                           const arma::mat& wtest, 
                                           double rho);