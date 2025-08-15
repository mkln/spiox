// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <limits>


// --------- helpers ---------

static inline std::vector<double> sorted_unique(const arma::vec& x) {
  arma::vec u = arma::unique(x);
  std::vector<double> v(u.begin(), u.end());
  std::sort(v.begin(), v.end());
  return v;
}

static inline arma::uvec indexify_col(const arma::vec& x, const std::vector<double>& uniq_sorted) {
  arma::uvec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double val = x[i];
    auto it = std::lower_bound(uniq_sorted.begin(), uniq_sorted.end(), val);
    if (it == uniq_sorted.end() || std::abs(*it - val) > 0.0) {
      Rcpp::stop("Non-gridded / mismatched value encountered.");
    }
    out[i] = static_cast<arma::uword>(std::distance(uniq_sorted.begin(), it));
  }
  return out;
}

static inline std::vector<arma::uword> grid_strides(const std::vector<arma::uword>& dims) {
  const arma::uword d = dims.size();
  std::vector<arma::uword> s(d, 1);
  for (arma::uword k = 1; k < d; ++k) s[k] = s[k - 1] * dims[k - 1];
  return s;
}

// generate all offsets in [-R..R]^d \ {0}, ordered by (dist^2, lex)
static inline std::vector< std::vector<int> >
  make_stencil(int d, int R) {
    std::vector< std::vector<int> > offs;
    offs.reserve(std::max(1, (int)std::pow(2*R+1, d) - 1));
    
    std::vector<int> v(d, -R), mins(d, -R), maxs(d, R);
    auto dist2 = [&](const std::vector<int>& a){
      long long s = 0; for (int k = 0; k < d; ++k) s += 1LL*a[k]*a[k]; return s;
    };
    auto lex_less = [&](const std::vector<int>& a, const std::vector<int>& b){
      for (int k = 0; k < d; ++k) { if (a[k] < b[k]) return true; if (a[k] > b[k]) return false; }
      return false;
    };
    
    while (true) {
      bool all_zero = true;
      for (int k = 0; k < d; ++k) if (v[k] != 0) { all_zero = false; break; }
      if (!all_zero) offs.push_back(v);
      int k = 0;
      for (; k < d; ++k) {
        v[k]++;
        if (v[k] <= maxs[k]) break;
        v[k] = mins[k];
      }
      if (k == d) break;
    }
    std::sort(offs.begin(), offs.end(),
              [&](const std::vector<int>& a, const std::vector<int>& b){
                auto da = dist2(a), db = dist2(b);
                if (da < db) return true;
                if (da > db) return false;
                return lex_less(a, b);
              });
    return offs;
  }

static inline arma::uword lin_from_multi(const std::vector<arma::uword>& strides, const arma::Row<arma::uword>& row) {
  arma::uword s = 0;
  for (arma::uword k = 0; k < row.n_cols; ++k) s += row[k] * strides[k];
  return s;
}

struct UwordHash {
  std::size_t operator()(const arma::uword& x) const noexcept {
    return std::hash<unsigned long long>()(static_cast<unsigned long long>(x));
  }
};


// [[Rcpp::export]]
arma::field<arma::uvec> dag_for_gridded_cols(const arma::mat& coords, int m = 20) {
 if (coords.n_rows == 0 || coords.n_cols == 0) Rcpp::stop("coords must be non-empty");
 if (m < 0) Rcpp::stop("m must be >= 0");
 
 const arma::uword N = coords.n_rows;
 const arma::uword d = coords.n_cols;
 
 // ---- 1) Grid indexing (0-based) ----

 std::vector< std::vector<double> > levels(d);
 std::vector<arma::uword> dims(d);
 arma::Mat<arma::uword> idx(N, d);
 
 for (arma::uword k = 0; k < d; ++k) {
   levels[k] = sorted_unique(coords.col(k));
   if (levels[k].empty()) Rcpp::stop("Empty grid in dimension %u", (unsigned)k);
   dims[k] = static_cast<arma::uword>(levels[k].size());
   idx.col(k) = indexify_col(coords.col(k), levels[k]);
 }
 std::vector<arma::uword> strides = grid_strides(dims);
 
 std::vector<arma::uword> lin(N);
 for (arma::uword i = 0; i < N; ++i) lin[i] = lin_from_multi(strides, idx.row(i));
 
 std::vector<arma::uword> order_idx(N);
 std::iota(order_idx.begin(), order_idx.end(), 0u);
 std::sort(order_idx.begin(), order_idx.end(),
           [&](arma::uword a, arma::uword b){ return lin[a] < lin[b]; });
 
 // If full grid: use flat vector lookup; else use hash map
 const arma::uword M =
   std::accumulate(dims.begin(), dims.end(), (arma::uword)1, std::multiplies<arma::uword>());
 bool full_grid = (M == N);
 
 std::vector<arma::uword> lin2id_vec;
 std::unordered_map<arma::uword, arma::uword, UwordHash> lin2id_map;
 
 if (full_grid) {
   lin2id_vec.assign(M, std::numeric_limits<arma::uword>::max());
   for (arma::uword r = 0; r < N; ++r) {
     arma::uword i = order_idx[r];
     lin2id_vec[ lin[i] ] = i;
   }
 } else {
   lin2id_map.reserve(N * 2);
   for (arma::uword r = 0; r < N; ++r) {
     arma::uword i = order_idx[r];
     lin2id_map.emplace(lin[i], i);
   }
 }
 
 int R = 0;
 auto pow_int = [](int a, int b){ int p = 1; for (int i=0;i<b;++i) p*=a; return p; };
 while ( ((pow_int(R+1, (int)d) - 1) < std::max(1, m)) && R < 100000 ) ++R;
 std::vector< std::vector<int> > S = make_stencil((int)d, R);
 
 arma::field<arma::uvec> parents(N);
 std::vector<arma::uword> cand_ids; cand_ids.reserve(m);
 std::vector<arma::uword> xj(d);
 
 for (arma::uword rr = 0; rr < N; ++rr) {
   arma::uword i = order_idx[rr];
   const arma::uword lin_i = lin[i];
   
   cand_ids.clear();
   
   // try offsets in S
   const arma::Row<arma::uword> xi = idx.row(i);
   for (const auto& off : S) {
     bool inside = true;
     for (arma::uword k = 0; k < d; ++k) {
       long long v = (long long)xi[k] + (long long)off[k];
       if (v < 0 || v >= (long long)dims[k]) { inside = false; break; }
       xj[k] = (arma::uword)v;
     }
     if (!inside) continue;
     
     arma::uword lin_j = 0;
     for (arma::uword k = 0; k < d; ++k) lin_j += xj[k] * strides[k];
     if (lin_j >= lin_i) continue;
     
     arma::uword j_id;
     if (full_grid) {
       j_id = lin2id_vec[lin_j];
       if (j_id == std::numeric_limits<arma::uword>::max()) continue; // hole (sparse)
     } else {
       auto it = lin2id_map.find(lin_j);
       if (it == lin2id_map.end()) continue;
       j_id = it->second;
     }
     
     cand_ids.push_back(j_id);
     if ((int)cand_ids.size() == m) break;
   }
   
   if (cand_ids.empty()) {
     parents(i) = arma::uvec();
   } else {
     if ((int)cand_ids.size() > m) cand_ids.resize(m);
     arma::uvec par(cand_ids.size());
     for (size_t t = 0; t < cand_ids.size(); ++t) par[t] = cand_ids[t];
     parents(i) = par;
   }
 }

 return parents;
}

