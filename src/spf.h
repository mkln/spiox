#include "dirichletlaplace.h"
using namespace std;


// sparse precision factor modeling, Chandra et al 2021+

class SparsePrecisionFactor {
public:
  arma::mat* Y; // n, d
  arma::mat U; // n, q
  arma::mat V; // n, d
  arma::vec Delta; // d x 1 -- diagonal of Delta matrix
  arma::mat Lambda; // d, q -- latent factor loadings
  
  // metadata
  unsigned int d;
  unsigned int n;
  unsigned int q;
  
  // prior params
  int a_delta; // shape for gamma prior of delta elements
  double b_delta; // rate for gamma prior of delta elements
  double a_dl; // dirichlet param for DL prior on elems of Lambda
  arma::mat S_Lambda; // element-by-element prior variance of Lambda
  
  // intermediate objects
  arma::mat P, Pchol, Pi;
  arma::mat Iq;
  
  // utilities
  void compute_P();
  void replace_Y(arma::mat* _Y){ Y = _Y; };
  
  // gibbs sampler functions
  void fc_sample_uv();
  void fc_sample_Lambda();
  void fc_sample_Lambda_seq();
  void fc_sample_Delta();
  void fc_sample_dl();
  
  // set starting values
  void Lambda_start(const arma::mat& L){
    Lambda = L;
  }
  void Delta_start(const arma::mat& D){
    Delta = D;
  }
  
  // constructor
  SparsePrecisionFactor(){};
  
  SparsePrecisionFactor(arma::mat* _Y, unsigned int _q,
                        double _a_delta, double _b_delta, double _a_dl) :
    Y(_Y),
    q(_q) ,
    a_delta(_a_delta),
    b_delta(_b_delta),
    a_dl(_a_dl) {
    n = Y->n_rows;
    d = Y->n_cols;
    Iq = arma::eye(q,q);
    
    S_Lambda = arma::ones(d, q);
    
  }
};
