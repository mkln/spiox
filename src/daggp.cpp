#include "daggp.h"
#include <thread>

static inline double logistic(double a) {
  if (a >= 0) { double z = std::exp(-a); return 1.0 / (1.0 + z); }
  else        { double z = std::exp(a);  return z / (1.0 + z); }
}

static inline double logit(double p) {
  const double eps = 1e-12;
  const double pp = std::min(1.0 - eps, std::max(eps, p));
  return std::log(pp) - std::log(1.0 - pp);
}

DagGP::DagGP(
  const arma::mat& coords_in,
  const arma::vec& theta_in,
  const arma::field<arma::uvec>& custom_dag,
  int dag_opts,
  int covariance_matern,
  bool use_Ci_in,
  int num_threads_in){
  
  coords = coords_in;
  theta = theta_in;
  nr = coords.n_rows;
  
  use_Ci = use_Ci_in; 
  
  dag = custom_dag;
  
  oneuv = arma::ones<arma::uvec>(1);
  
  matern = covariance_matern; // 0 pexp or 1 matern or 2 wave or 3 sqexpcovariates
  //Rcpp::Rcout << "DagGP covariance choice " << matern << endl;
  //thread safe stuff
  n_threads = num_threads_in;
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  bessel_ws = (double *) R_alloc(n_threads*bessel_ws_inc, sizeof(double));
  
  //
  mblanket = arma::field<arma::uvec>(nr);
  hrows = arma::field<arma::vec>(nr);
  ax = arma::field<arma::uvec>(nr);
  
  if(dag_opts != 0){
    if(dag_opts > 0){
      // pruning the dag for efficiency
      // = trim edges to see if there are ties in the parent sets
      max_prune = dag_opts;
      
      cache_map = arma::zeros<arma::umat>(nr, 2);
      prune_dag_cache();
    } else {
      // gridded coords so full cache
      max_prune = -1;
      
      cache_map = arma::zeros<arma::umat>(nr, 2);
      build_grid_exemplars();
    }
  } else {
    max_prune = 0;
    dag_cache = dag;                           
    cache_map.set_size(nr, 2);
    for (unsigned int i = 0; i < nr; i++) {
      cache_map(i, 0) = i;                     
      cache_map(i, 1) = i;                     
    }
  }
  
  
  compute_comps();
  initialize_H();
  
  arma::wall_clock timer;
  
  timer.tic(); 
  color_from_mblanket();
  double n_secs = timer.toc(); // Stop timer and get seconds
  //Rcpp::Rcout << "Compute time 1: " << n_secs << " seconds" << std::endl;
  
}


void DagGP::prune_dag_cache() {
  // this function simplifies the dag by removing at most max_prune edges from each node
  // to possibly find that multiple nodes have the same parent set
  // when parent sets are the same, we can reduce the number of matrix inversions
  
  const int n = dag.n_elem;
  
  int m = 0;
  for(int i=0; i<dag.n_elem; i++){
    m = dag(i).n_elem > m ? dag(i).n_elem : m;
  }
  
  int min_support = 1;
  // only prune for dags with more than 10 neighbors
  int min_size = m > 10? std::max(m - max_prune, 10) : m; 
  
  //Rcpp::Rcout << "min_size: " << min_size << endl;
  
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
  
  // 1) Canonicalize parent sets P[i] (sorted & unique; as std::vector<int>)
  std::vector< std::vector<int> > P(n);
  long long edges_before = 0;
  for (int i = 0; i < n; ++i) {
    const arma::uvec& vi = dag(i);
    std::vector<int> tmp; tmp.reserve(vi.n_elem);
    for (size_t k = 0; k < vi.n_elem; ++k) {
      int val = (int)vi[k];
      //if (val > 0) 
      tmp.push_back(val);
    }
    P[i] = set_unique_sorted(tmp);
    edges_before += (long long)P[i].size();
  }
  
  // 2) Build postings (parent -> nodes), and prune rare parents (< min_support)
  Postings postings;
  postings.reserve(n * 2);
  for (int i = 0; i < n; ++i) {
    for (int p : P[i]) postings[p].push_back(i); // store nodes as 0-based
  }
  for (auto &kv : postings) {
    auto &vec = kv.second;
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
  }
  { // drop rare parents
    std::vector<int> to_drop; to_drop.reserve(postings.size());
    for (const auto &kv : postings)
      if ((int)kv.second.size() < min_support) to_drop.push_back(kv.first);
      for (int p : to_drop) postings.erase(p);
  }
  // Drop rare parents from each node's set
  for (int i = 0; i < n; ++i) {
    std::vector<int> kept; kept.reserve(P[i].size());
    for (int p : P[i]) if (postings.find(p) != postings.end()) kept.push_back(p);
    P[i].swap(kept);
  }
  
  // If nothing left after pruning, return identity mapping with originals
  bool any_left = false;
  for (int i = 0; i < n; ++i) { if (!P[i].empty()) { any_left = true; break; } }
  if (!any_left) {
    // unique sets = originals; exemplar id = node id
    dag_cache = arma::field<arma::uvec>(n);
    for (int i = 0; i < n; ++i) dag_cache(i) = arma::zeros<arma::uvec>(0); // empty after dropping all
    
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  // 3) Build candidate node pairs (i<j) by shared parents (parallelized)
  std::vector< std::pair<int,int> > pairs; 
  
  {
    // gather parent ids to iterate in parallel
    std::vector<int> parent_ids; parent_ids.reserve(postings.size());
    for (const auto &kv : postings) parent_ids.push_back(kv.first);
    
#ifdef _OPENMP
    int T = (n_threads > 0 ? n_threads : omp_get_max_threads());
#else
    int T = 1;
#endif
    std::vector< std::vector< std::pair<int,int> > > buffers(T);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int idx = 0; idx < (int)parent_ids.size(); ++idx) {
#ifdef _OPENMP
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      const int p = parent_ids[idx];
      const std::vector<int>& nodes = postings[p];
      const int s = (int)nodes.size();
      if (s < 2) continue;
      auto &buf = buffers[tid];
      for (int a = 0; a < s; ++a) {
        for (int b = a + 1; b < s; ++b) {
          buf.emplace_back(nodes[a], nodes[b]);
        }
      }
    }
    // merge & deduplicate
    size_t total = 0;
    for (auto &b : buffers) total += b.size();
    pairs.reserve(total);
    for (auto &b : buffers) {
      pairs.insert(pairs.end(), b.begin(), b.end());
      std::vector< std::pair<int,int> >().swap(b);
    }
  }
  
  if (pairs.empty()) {
    // No shared-parent pairs -> cores = originals
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  
  // 4) Candidate subsets = intersections P[i] ∩ P[j]; keep |S| >= min_size (parallel)
#ifdef _OPENMP
  int T_c = (n_threads > 0 ? n_threads : omp_get_max_threads());
#else
  int T_c = 1;
#endif
  std::vector< std::vector< std::vector<int> > > cands_local(T_c);
  
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int k = 0; k < (int)pairs.size(); ++k) {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    const int i = pairs[k].first;
    const int j = pairs[k].second;
    const auto S = intersect_sorted(P[i], P[j]);
    if ((int)S.size() >= min_size) cands_local[tid].push_back(S);
  }
  
  // Merge & deduplicate candidates by key
  std::vector< std::vector<int> > Cands;
  {
    std::unordered_set<std::string> seen;
    for (int t = 0; t < T_c; ++t) {
      for (auto &S : cands_local[t]) {
        std::string key = key_of_vec(S);
        if (!key.empty() && seen.insert(key).second) {
          Cands.push_back(std::move(S));
        }
      }
      std::vector< std::vector<int> >().swap(cands_local[t]);
    }
  }
  
  if (Cands.empty()) {
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  // 5) Compute supports for candidates (parallel), keep those >= min_support
  std::vector<int> support(Cands.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < (int)Cands.size(); ++c) {
    support[c] = support_size_of_subset(Cands[c], postings, min_support);
  }
  
  std::vector< std::vector<int> > C_keep;
  std::vector<int> supp_keep;
  std::vector<std::string> keys_keep;
  C_keep.reserve(Cands.size());
  supp_keep.reserve(Cands.size());
  keys_keep.reserve(Cands.size());
  
  for (size_t c = 0; c < Cands.size(); ++c) {
    if (support[c] >= min_support) {
      C_keep.push_back(std::move(Cands[c]));
      supp_keep.push_back(support[c]);
      keys_keep.push_back(key_of_vec(C_keep.back()));
    }
  }
  
  if (C_keep.empty()) {
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }

  }
  
  // 6) For each frequent candidate, get the nodes that contain it (parallel)
  std::vector< std::vector<int> > cand_nodes(C_keep.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < (int)C_keep.size(); ++c) {
    cand_nodes[c] = nodes_having_subset(C_keep[c], postings); 
  }
  
  // 7) Per-node best core (size -> support -> lexicographic key)
  struct BestCore {
    int size = -1;
    int support = -1;
    std::string key;
    int cand_index = -1;
  };
  std::vector<BestCore> best(n);
  
  for (int c = 0; c < (int)C_keep.size(); ++c) {
    const int sz = (int)C_keep[c].size();
    const int sp = supp_keep[c];
    const std::string &ky = keys_keep[c];
    const std::vector<int> &nodes = cand_nodes[c];
    for (int node1b : nodes) {
      const int i = node1b;
      auto &cur = best[i];
      bool better = false;
      if (sz > cur.size) better = true;
      else if (sz == cur.size && sp > cur.support) better = true;
      else if (sz == cur.size && sp == cur.support && (cur.key.empty() || ky < cur.key)) better = true;
      if (better) {
        cur.size = sz; cur.support = sp; cur.key = ky; cur.cand_index = c;
      }
    }
  }
  
  // Chosen cores (fallback to original P[i] if none assigned)
  std::vector< std::vector<int> > cores(n);
  for (int i = 0; i < n; ++i) {
    if (best[i].cand_index >= 0) cores[i] = C_keep[best[i].cand_index];
    else cores[i] = P[i];
  }
  
  // 8) Deduplicate cores to exemplar ids
  std::unordered_map<std::string, int> core_to_id;
  core_to_id.reserve(n * 2);
  std::vector< std::vector<int> > exemplars; exemplars.reserve(n);
  
  for (int i = 0; i < n; ++i) {
    std::string k = key_of_vec(cores[i]);
    auto it = core_to_id.find(k);
    int gid;
    if (it == core_to_id.end()) {
      gid = (int)exemplars.size(); // 0-based group id
      core_to_id.emplace(k, gid);
      exemplars.push_back(cores[i]);
    } else {
      gid = it->second;
    }
    cache_map(i, 0) = i;
    cache_map(i, 1) = gid;
  }
  
  // Convert exemplars to R list and compute edges_after as total size

  dag_cache = arma::field<arma::uvec>(exemplars.size());
  long long edges_after = 0;
  for (size_t k = 0; k < exemplars.size(); ++k) {
    dag_cache(k) = arma::conv_to<arma::uvec>::from(exemplars[k]);
    edges_after += (long long)exemplars[k].size();
  }
}

void DagGP::build_grid_exemplars() {
  // this function computes exemplar sets (parents, i) for gridded data
  // and assuming isotropic covariance
  // in this case the relationships between locations replicates across the domain
  // so we can massively reduce the number of matrix operations
  // this function is best paired with a neighbor search on an ordering that
  // makes it more likely to find many ties.
  
  const arma::uword N = coords.n_rows;
  const arma::uword d = coords.n_cols;
  if (N != dag.n_elem) Rcpp::stop("coords / dag size mismatch");
  
  arma::field<arma::uvec> dagplus = dag;
  if(max_prune < 0){
    // gridded coords, so cache the child too
    for(unsigned int i=0; i<nr; i++){
      dagplus(i) = arma::join_vert(dag(i), oneuv*i);
    }
  }
  
  
  // 1) grid index per dimension (0-based)
  arma::Mat<arma::uword> idx(N, d);
  for (arma::uword k = 0; k < d; ++k) {
    auto levels = sorted_unique_col(coords.col(k));
    idx.col(k) = indexify_col_exact(coords.col(k), levels);
  }
  
  // 2) hash by ordered offset pattern
  std::unordered_map<std::string, int> key2gid;
  key2gid.reserve(N * 2);
  
  cache_map.set_size(N, 2);
  std::vector< arma::uvec > exemplar_sets; exemplar_sets.reserve(N);
  
  for (arma::uword i = 0; i < N; ++i) {
    const arma::uvec& Pi = dagplus(i);           // 0-based parents for node i
    std::string key = key_from_offsets(idx, i, Pi);
    auto it = key2gid.find(key);
    int gid;
    if (it == key2gid.end()) {
      gid = (int)exemplar_sets.size();
      key2gid.emplace(std::move(key), gid);
      exemplar_sets.push_back(Pi);            // store this node's parents as exemplar
    } else {
      gid = it->second;
    }
    cache_map(i, 0) = i;                      // node id (0-based)
    cache_map(i, 1) = gid;                    // exemplar id (0-based)
  }
  
  // 3) materialize dag_cache from exemplars
  dag_cache.set_size(exemplar_sets.size());
  for (size_t g = 0; g < exemplar_sets.size(); ++g) {
    dag_cache(g) = exemplar_sets[g];
  }
  
  const arma::uword G = exemplar_sets.size();
  dag_cache.set_size(G);
  child_cache = arma::zeros<arma::uvec>(G);  // arma::uvec
  
  for (arma::uword g = 0; g < G; ++g) {
    if(max_prune < 0){
      const arma::uvec& v = exemplar_sets[g];      // expects [parents..., child]
      const arma::uword n = v.n_elem;
      
      if (n == 0) {
        dag_cache(g)   = arma::uvec();             // empty parents
        child_cache(g) = arma::uword(-1);          // sentinel (or choose 0)
        continue;
      }
      
      // parents = first n-1 elements (copy); handles n==1 by making empty uvec
      if (n > 1) dag_cache(g) = v.head(n - 1);
      else       dag_cache(g) = arma::uvec();      // no parents
      // child = last element
      child_cache(g) = v[n - 1];                   // or v.back()
    } else {
      dag_cache(g) = exemplar_sets[g];
    }
  }
}

void DagGP::color_from_mblanket(){
  
  arma::uword n_nodes = mblanket.n_elem;
  
  // Stores the color assigned to each node (-1 = unassigned)
  // This allows O(1) lookup to see a neighbors color.
  std::vector<int> node_colors(n_nodes, -1);
  
  // Intermediate storage: groups[c] is the list of nodes with color c
  std::vector<std::vector<arma::uword>> groups;
  
  for (arma::uword i = 0; i < n_nodes; ++i) {
    
    // 1. Identify which colors are forbidden for this node
    // We assume the number of colors wont exceed the current max + 1
    std::vector<bool> forbidden(groups.size() + 1, false);
    
    const arma::uvec& neighbors = mblanket(i);
    for (arma::uword neighbor_idx : neighbors) {
      // Safety check for index and check if neighbor is already colored
      if (neighbor_idx < n_nodes && node_colors[neighbor_idx] != -1) {
        int neighbor_c = node_colors[neighbor_idx];
        
        // If neighbors color is within our current range, mark it
        if (neighbor_c < forbidden.size()) {
          forbidden[neighbor_c] = true;
        }
      }
    }
    
    // 2. Pick the first color that is NOT forbidden
    int chosen_color = 0;
    while (chosen_color < forbidden.size() && forbidden[chosen_color]) {
      chosen_color++;
    }
    
    // 3. Assign the color
    node_colors[i] = chosen_color;
    
    // 4. Add to the corresponding group
    // If we picked a new color we havent used before, resize the groups vector
    if (chosen_color >= groups.size()) {
      groups.resize(chosen_color + 1);
    }
    groups[chosen_color].push_back(i);
  }
  
  // 5. Convert std::vector to arma::field
  colors = arma::field<arma::uvec>(groups.size());
  for (size_t c = 0; c < groups.size(); ++c) {
    colors(c) = arma::uvec(groups[c]);
  }
  
}

void DagGP::compute_comps(bool update_H){
  // this function avoids building H since H is always used to multiply a matrix A
  sqrtR = arma::zeros(nr);
  h = arma::field<arma::vec>(nr);
  arma::vec logdetvec(nr);
  
  arma::uvec errors = arma::zeros<arma::uvec>(nr);
  
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed;
  
  tstart = std::chrono::steady_clock::now();
  const int G = static_cast<int>(dag_cache.n_elem);
  arma::field<arma::mat> Pinv_cache(G);
  arma::field<arma::mat> h_cache(G);
  arma::vec CPC_cache(G);

  bool cache_error = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) 
#endif
  for (int g = 0; g < G; ++g) {
    arma::mat Pg;
    try {
      const arma::uvec& pxg = dag_cache(g);          // 0-based indices
      
      if (pxg.n_elem == 0u) { Pinv_cache(g).reset(); continue; }
      
      Pg  = Correlationf(coords, pxg, pxg, theta, bessel_ws, matern, /*same=*/true);
      Pinv_cache(g) = arma::inv_sympd(Pg);
      
      if(max_prune < 0){
        const arma::uvec& ixg = oneuv*child_cache(g);
        arma::mat CPt = Correlationf(coords, pxg, ixg, theta, bessel_ws, matern, false);
        h_cache(g) = Pinv_cache(g) * CPt;
        CPC_cache(g) = arma::conv_to<double>::from(CPt.t() * h_cache(g));
      }
    } catch (...) {
      cache_error = true;
    }
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  
  if(cache_error){
    Rcpp::stop("Failed to compute cached elements.\n");
  }
  
  //Rcpp::Rcout << "First loop: " << timed << endl; 
  
  tstart = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    try {
      arma::uvec ix = oneuv * i;                 
      const int gid = static_cast<int>(cache_map(i,1));   
      
      arma::uvec px;
      if(max_prune < 0){
        // gridded, look up original dag
        // because dag_cache has the exemplar parents
        px = dag(i);
      } else {
        px = dag_cache(gid);   
      }   
      
      ax(i) = arma::join_vert(ix, px);
      
      arma::mat CC  = Correlationf(coords, ix, ix, theta, bessel_ws, matern, true);
      
      arma::mat CPC;
      arma::vec hi;
      
      if (px.n_elem == 0u) {
        hi.set_size(0);
        sqrtR(i) = std::sqrt( arma::conv_to<double>::from( CC ) );
      } else {
        //const arma::mat& Pinv = Pinv_cache(gid);   
        // gridded data
        if(max_prune < 0){
          hi = h_cache(gid);   
          CPC = CPC_cache(gid);
        } else {
          arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, matern, false);
          hi = Pinv_cache(gid) * CPt;             
          CPC = CPt.t() * hi;
        }
        sqrtR(i) = std::sqrt( arma::conv_to<double>::from( CC - CPC ) );
      }
      
      h(i) = hi;
      
      arma::vec mhR = (px.n_elem ? -hi / sqrtR(i) : arma::vec());
      arma::vec rowoneR = arma::ones<arma::vec>(1) / sqrtR(i);
      
      hrows(i) = (px.n_elem ? arma::join_vert(rowoneR, mhR) : rowoneR);
      logdetvec(i) = -std::log(sqrtR(i));
      
      if(update_H){
        H(i,i) = 1.0/sqrtR(i);
        for (arma::uword j = 0; j < px.n_elem; ++j) {
          H(i, px(j)) = -hi(j) / sqrtR(i);        
        }
      }
    } catch (...) {
      errors(i) = 1;
    }
  }
  tend = std::chrono::steady_clock::now();
  
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  
  //Rcpp::Rcout << "Second loop: " << timed << endl; 
  
  if(arma::any(errors)){
    Rcpp::stop("Failure in building sparse DAG GP. Check coordinates/values of theta.");
  }
  
  if(use_Ci){ Ci = H.t() * H; }
  precision_logdeterminant = 2 * arma::accu(logdetvec);
}


double DagGP::logdens(const arma::vec& x){
  double loggausscore = arma::conv_to<double>::from( x.t() * H.t() * H * x );
  return 0.5 * ( precision_logdeterminant - loggausscore );
}

double DagGP::logdens_fast(const arma::vec& x) const {
  // log p(x) = 0.5 * ( logdet(Q) - x^T Q x ), Q = H^T H
  // x^T Q x = ||H x||^2, compute u = Hx via hrows/ax
  arma::vec u(nr, arma::fill::zeros);
  for (arma::uword i = 0; i < nr; ++i) {
    u(i) = arma::dot(hrows(i), x.elem(ax(i)));
  }
  const double quad = arma::dot(u, u);
  return 0.5 * (precision_logdeterminant - quad);
}

void DagGP::update_theta(const arma::vec& newtheta, bool update_H){
  theta = newtheta;
  compute_comps(update_H);
}

void DagGP::initialize_H(){
  // building sparse H and Ci
  unsigned int H_n_elem = nr;
  for(int i=0; i<nr; i++){
    if(max_prune < 0){
      // gridded, look up original dag
      // because dag_cache has the exemplar parents
      H_n_elem += dag(i).n_elem;
    } else {
      int u = cache_map(i, 1);                 
      H_n_elem += dag_cache(u).n_elem;      
    }
    
       
  }
  arma::umat H_locs(2, H_n_elem);
  arma::vec H_values(H_n_elem);
  unsigned int ix=0;
  for(int i=0; i<nr; i++){
    H_locs(0, ix) = i;
    H_locs(1, ix) = i;
    H_values(ix) = 1.0/sqrtR(i);
    ix ++;
    int u;
    arma::uvec daghere;
    if(max_prune < 0){
      u = i;
      daghere = dag(u);
    } else {
      u = cache_map(i,1);
      daghere = dag_cache(u);
    }
    
    for(unsigned int j=0; j<daghere.n_elem; j++){
      H_locs(0, ix) = i;
      H_locs(1, ix) = daghere(j);
      H_values(ix) =  -h(i)(j)/sqrtR(i);
      ix ++;
    }
  }
  H = arma::sp_mat(H_locs, H_values, nr, nr);
  
  // compute precision just to get the markov blanket
  arma::sp_mat Ci_temp = H.t() * H;
  
  // comp markov blanket
  for(int i=0; i<nr; i++){
    int nonzeros_in_col = 0;
    for (arma::sp_mat::const_iterator it = Ci_temp.begin_col(i); it != Ci_temp.end_col(i); ++it) {
      nonzeros_in_col ++;
    }
    mblanket(i) = arma::zeros<arma::uvec>(nonzeros_in_col-1); // not the diagonal
    int ix = 0;
    for (arma::sp_mat::const_iterator it = Ci_temp.begin_col(i); it != Ci_temp.end_col(i); ++it) {
      if(it.row() != i){ 
        mblanket(i)(ix) = it.row();
        ix ++;
      }
    }
  }
  
  if(use_Ci){
    Ci = Ci_temp;
  }
}

arma::mat DagGP::H_solve_A(const arma::mat& A, bool use_spmat){
  if(!use_spmat){
    // solve using rows of H stored in hrows and indexes in ax
    arma::mat Y = A; 
    for (arma::uword i = 0; i < nr; ++i) {
      double diag_val = 0.0;
      bool diag_found = false;
      const arma::vec& row_vals = hrows(i);
      const arma::uvec& row_cols = ax(i);
      for (arma::uword idx = 0; idx < row_vals.n_elem; ++idx) {
        arma::uword k = row_cols(idx);
        double val = row_vals(idx);
        if (k < i) {
          Y.row(i) -= val * Y.row(k);
        } else if (k == i) {
          diag_val = val;
          diag_found = true;
        }
      }
      Y.row(i) /= diag_val;
    }
    return Y;
  } else {
    // solve using H matrix 
    arma::mat Y = A; 
    for (arma::uword j = 0; j < nr; ++j) {
      arma::sp_mat::const_col_iterator it = H.begin_col(j);
      arma::sp_mat::const_col_iterator it_end = H.end_col(j);
      double diag_val = 0.0;
      for (arma::sp_mat::const_col_iterator diag_it = it; diag_it != it_end; ++diag_it) {
        if (diag_it.row() == j) {
          diag_val = *diag_it;
          break;
        }
      }
      Y.row(j) /= diag_val;
      for (; it != it_end; ++it) {
        arma::uword i = it.row();
        if (i > j) { 
          Y.row(i) -= (*it) * Y.row(j);
        }
      }
    }
    return Y;
  }
}

arma::mat DagGP::H_times_A(const arma::mat& A, bool use_spmat){
  // this function avoids building H since H is always used to multiply a matrix A
  if(!use_spmat){
    arma::mat result = arma::zeros(nr, A.n_cols);
    // calculate components for H and Ci
    for(unsigned int j=0; j<A.n_cols; j++){
      arma::vec Aj = A.col(j);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for(int i=0; i<nr; i++){
        result(i, j) = arma::accu(hrows(i).t() * Aj(ax(i)));
      }
    }
    return result;
  } else {
    return H * A;
  }
}


arma::mat DagGP::Corr_export(const arma::mat& these_coords, 
                             const arma::uvec& ix, const arma::uvec& jx, 
                             int matern, bool same){
  return Correlationf(these_coords, ix, jx, theta, bessel_ws, matern, same);
}



arma::vec DagGP::grad_theta_loglik_matern_halfint(const arma::vec& y) {
  if (matern != 1) Rcpp::stop("grad_theta_loglik_matern_halfint: matern must be 1 (Matérn).");
  if (theta.n_elem < 4) Rcpp::stop("theta must be (phi, sigmasq, nu, alpha).");
  
  const double nu = theta(2);
  const bool ok_nu =
    (std::abs(nu - 0.5) < 1e-12) ||
    (std::abs(nu - 1.5) < 1e-12) ||
    (std::abs(nu - 2.5) < 1e-12);
  if (!ok_nu) Rcpp::stop("Only nu in {0.5,1.5,2.5} supported in this gradient.");
  
  const arma::uword ptheta = theta.n_elem; // should be 4
  arma::vec grad(ptheta, arma::fill::zeros);
  
  // ---- 1. CACHING PHASE ----
  const int G = static_cast<int>(dag_cache.n_elem);
  arma::field<arma::mat> Pinv_cache(G);
  
  // These are ONLY allocated and used if the data is gridded
  arma::field<arma::vec> h_cache;
  arma::vec R_cache;
  arma::field<std::vector<arma::vec>> dhi_cache;
  arma::field<std::vector<double>> dR_cache;
  
  if (max_prune < 0) {
    h_cache.set_size(G);
    R_cache.set_size(G);
    dhi_cache.set_size(G);
    dR_cache.set_size(G);
  }
  
  bool cache_error = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for (int g = 0; g < G; ++g) {
    try {
      const arma::uvec& pxg = dag_cache(g);
      if (pxg.n_elem == 0u) { 
        Pinv_cache(g).reset(); 
        continue; 
      }
      
      // Base covariance and inverse (Computed for ALL data types)
      arma::mat Pg = Correlationf(coords, pxg, pxg, theta, bessel_ws, matern, /*same=*/true);
      arma::mat Pinv = arma::inv_sympd(Pg);
      Pinv_cache(g) = Pinv;
      
      // Fully gridded cache: Precompute gradient components to avoid O(N) matrix ops
      if (max_prune < 0) {
        std::vector<arma::mat> dP = Correlationf_grad(coords, pxg, pxg, theta, bessel_ws, matern, true);
        std::vector<arma::mat> dPinv(ptheta);
        for (arma::uword t = 0; t < ptheta; ++t) {
          dPinv[t] = -Pinv * dP[t] * Pinv;
        }
        
        arma::uvec ixg = oneuv * child_cache(g);
        
        arma::mat CPt = Correlationf(coords, pxg, ixg, theta, bessel_ws, matern, false);
        std::vector<arma::mat> dCPt = Correlationf_grad(coords, pxg, ixg, theta, bessel_ws, matern, false);
        
        double CCs = arma::as_scalar(Correlationf(coords, ixg, ixg, theta, bessel_ws, matern, true));
        std::vector<arma::mat> dCC = Correlationf_grad(coords, ixg, ixg, theta, bessel_ws, matern, true);
        
        arma::vec hg = Pinv * CPt;
        h_cache(g) = hg;
        R_cache(g) = CCs - arma::as_scalar(CPt.t() * hg);
        
        std::vector<arma::vec> dhig(ptheta);
        std::vector<double> dRg(ptheta);
        for (arma::uword t = 0; t < ptheta; ++t) {
          dhig[t] = dPinv[t] * CPt + Pinv * dCPt[t];
          double dCPC = arma::as_scalar(dCPt[t].t() * hg + CPt.t() * dhig[t]);
          dRg[t] = arma::as_scalar(dCC[t]) - dCPC;
        }
        dhi_cache(g) = dhig;
        dR_cache(g) = dRg;
      }
    } catch (...) {
      cache_error = true;
    }
  }
  if (cache_error) Rcpp::stop("Failed to compute Pinv and gradient caches.");
  
  // ---- 2. MAIN LOOP ----
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  arma::vec glocal(ptheta, arma::fill::zeros);
  
#ifdef _OPENMP
#pragma omp for
#endif
  for (int i = 0; i < nr; ++i) {
    try {
      arma::uvec ix = oneuv * i;
      const int gid = static_cast<int>(cache_map(i,1));
      
      arma::uvec px = (max_prune < 0) ? dag(i) : dag_cache(gid);
      const arma::uword m = px.n_elem;
      
      // --- Root Nodes ---
      if (m == 0u) {
        const double CCs = arma::as_scalar(Correlationf(coords, ix, ix, theta, bessel_ws, matern, true));
        std::vector<arma::mat> dCC = Correlationf_grad(coords, ix, ix, theta, bessel_ws, matern, true);
        const double sqrtR = std::sqrt(CCs);
        const double invS = 1.0 / sqrtR;
        const double ui = invS * y(i);
        for (arma::uword t = 0; t < ptheta; ++t) {
          const double dR = arma::as_scalar(dCC[t]);
          glocal(t) += -(0.5 / CCs) * dR - ui * (-0.5 * dR / (CCs * sqrtR) * y(i));
        }
        continue;
      }
      
      double R, sqrtR, invS, numer;
      std::vector<arma::vec> dhi(ptheta);
      std::vector<double> dR(ptheta);
      
      if (max_prune < 0) {
        // --- GRIDDED DATA (O(1) Vector Lookups) ---
        const arma::vec& hi = h_cache(gid);
        R = R_cache(gid);
        dhi = dhi_cache(gid);
        dR = dR_cache(gid);
        numer = y(i) - arma::as_scalar(hi.t() * y.elem(px));
        
      } else {
        // --- NON-GRIDDED DATA (Compute on the fly, zero OMP memory contention) ---
        const arma::mat& Pinv = Pinv_cache(gid);
        
        arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, matern, false);
        std::vector<arma::mat> dCPt = Correlationf_grad(coords, px, ix, theta, bessel_ws, matern, false);
        std::vector<arma::mat> dP = Correlationf_grad(coords, px, px, theta, bessel_ws, matern, true);
        
        double CCs = arma::as_scalar(Correlationf(coords, ix, ix, theta, bessel_ws, matern, true));
        std::vector<arma::mat> dCC = Correlationf_grad(coords, ix, ix, theta, bessel_ws, matern, true);
        
        arma::vec hi = Pinv * CPt;
        R = CCs - arma::as_scalar(CPt.t() * hi);
        numer = y(i) - arma::as_scalar(hi.t() * y.elem(px));
        
        for (arma::uword t = 0; t < ptheta; ++t) {
          arma::mat dPinv_t = -Pinv * dP[t] * Pinv;
          dhi[t] = dPinv_t * CPt + Pinv * dCPt[t];
          double dCPC = arma::as_scalar(dCPt[t].t() * hi + CPt.t() * dhi[t]);
          dR[t] = arma::as_scalar(dCC[t]) - dCPC;
        }
      }
      
      // Final likelihood assembly 
      sqrtR = std::sqrt(R);
      invS = 1.0 / sqrtR;
      const double ui = numer * invS;
      
      for (arma::uword t = 0; t < ptheta; ++t) {
        const double dlogdet = -(0.5 / R) * dR[t];
        const double dnumer = -arma::as_scalar(dhi[t].t() * y.elem(px)); 
        const double dinvS = -0.5 * dR[t] / (R * sqrtR);
        const double dui = dnumer * invS + numer * dinvS;
        
        glocal(t) += dlogdet - ui * dui;
      }
    } catch (...) {
      Rcpp::stop("Failure in grad_theta_loglik_matern_halfint().");
    }
  }
#ifdef _OPENMP
#pragma omp critical
#endif
{ grad += glocal; }
}

return grad;
}



//[[Rcpp::export]]
Rcpp::List daggp_build(const arma::mat& coords, const arma::field<arma::uvec>& dag,
                       double phi, double sigmasq, double nu, double alpha,
                       int matern=1, int num_threads=1, int dag_opts=0){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = alpha;
  
  //Rcpp::Rcout << "Building DAG-GP model\n";
  DagGP adag(coords, theta, dag, dag_opts, matern, false, num_threads);
  
  arma::sp_mat Ci = adag.H.t() * adag.H;
  //Rcpp::Rcout << "Done. Returning. \n";
  
  return Rcpp::List::create(
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant,
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("dag_cache") = adag.dag_cache,
    Rcpp::Named("cache_map") = adag.cache_map,
    Rcpp::Named("markov_blankets") = adag.mblanket
  );
}


//[[Rcpp::export]]
Rcpp::List daggpXd_build(const arma::mat& cx,
                         const arma::field<arma::uvec>& dag,
                         const arma::mat& M_s,          
                         const arma::mat& Theta_x,      
                         const double alpha,            
                         int num_threads = 1,
                         int dag_opts = 0){
  
  // M_s // 2x2 SPD (spatial metric)
  // Theta_x // D_x x k
  // alpha // in [0,1]
  
  const arma::uword D = cx.n_cols;
  if (D < 2) Rcpp::stop("daggpXd_build: cx must have at least 2 spatial columns.");
  const arma::uword D_x = D - 2;
  
  // ---- spatial SPD metric -> Theta_s (length 3) ----
  if (M_s.n_rows != 2 || M_s.n_cols != 2) {
    Rcpp::stop("daggpXd_build: M_s must be 2x2.");
  }
  // basic SPD check (will throw if not SPD)
  arma::mat Ls;
  try {
    Ls = arma::chol(M_s, "lower"); // M_s = Ls * Ls^T
  } catch (...) {
    Rcpp::stop("daggpXd_build: M_s must be symmetric positive definite (chol failed).");
  }
  
  // Our Theta_s parameterization corresponds to A_s lower-tri:
  // [ a11  0  ]
  // [ a21 a22 ]
  // where A_s A_s^T is the spatial metric.
  // So set Theta_s from entries of Ls (already lower-tri).
  arma::vec Theta_s(3);
  Theta_s(0) = Ls(0,0); // a11
  Theta_s(1) = Ls(1,0); // a21
  Theta_s(2) = Ls(1,1); // a22
  
  // ---- feature block checks ----
  if (Theta_x.n_rows != D_x) {
    Rcpp::stop("daggpXd_build: Theta_x.n_rows must equal ncol(cx)-2.");
  }
  if (Theta_x.n_cols < 1) {
    Rcpp::stop("daggpXd_build: Theta_x must have at least 1 column (k>=1).");
  }
  if (!(alpha >= 0.0 && alpha <= 1.0)) {
    Rcpp::stop("daggpXd_build: alpha must be in [0,1].");
  }
  
  // ---- pack theta to match Correlationf(matern=3): [Theta_s, vec(Theta_x), alpha] ----
  arma::vec theta(3 + D_x * Theta_x.n_cols + 1);
  theta.subvec(0, 2) = Theta_s;
  theta.subvec(3, 3 + D_x * Theta_x.n_cols - 1) = arma::vectorise(Theta_x);
  theta(theta.n_elem - 1) = alpha;
  
  const int matern = 3;
  
  DagGP adag(cx, theta, dag, dag_opts, matern, /*use_Ci_in=*/false, num_threads);
  
  arma::sp_mat Ci = adag.H.t() * adag.H;
  
  return Rcpp::List::create(
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant,
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("dag_cache") = adag.dag_cache,
    Rcpp::Named("cache_map") = adag.cache_map,
    Rcpp::Named("markov_blankets") = adag.mblanket,
    Rcpp::Named("Theta_s") = Theta_s,    
    Rcpp::Named("A_s") = Ls              
  );
}

/// FS

Rcpp::List DagGP::score_and_opg_matern_halfint(const arma::vec& y) {
  if (matern != 1) Rcpp::stop("score_and_opg: matern must be 1 (Matérn).");
  if (theta.n_elem < 4) Rcpp::stop("theta must be (phi, sigmasq, nu, alpha).");
  
  const double nu = theta(2);
  const bool ok_nu =
    (std::abs(nu - 0.5) < 1e-12) ||
    (std::abs(nu - 1.5) < 1e-12) ||
    (std::abs(nu - 2.5) < 1e-12);
  if (!ok_nu) Rcpp::stop("Only nu in {0.5,1.5,2.5} supported.");
  
  // unconstrained params u = (logphi, logsig2, logit(alpha))
  const double phi     = theta(0);
  const double sig2    = theta(1);
  const double alpha   = theta(3);
  
  const double u0 = std::log(std::max(phi,  1e-12));
  const double u1 = std::log(std::max(sig2, 1e-12));
  const double u2 = logit(alpha);
  
  // compute comps for current theta, but no sparse rebuild
  // (caller should have update_theta already; safe to ensure it)
  // update_theta(theta, false);
  
  // 0) loglik (fast path)
  const double ll = logdens_fast(y);
  
  // 1) Cache Pinv (geometry-based)
  const bool grid_mode = (max_prune < 0);
  const int G = static_cast<int>(dag_cache.n_elem);
  
  arma::field<arma::mat> Pinv_cache(G);
  
  // gridded extra caches (same idea as your code)
  arma::field<arma::vec> h_cache(G);
  arma::vec R_cache(G, arma::fill::zeros);
  arma::field<std::array<arma::vec,4>> dhi_cache(G);
  arma::field<std::array<double,4>> dR_cache(G);
  
  bool cache_error = false;
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for (int g = 0; g < G; ++g) {
    try {
      const arma::uvec& pxg = dag_cache(g);
      if (pxg.n_elem == 0u) { Pinv_cache(g).reset(); continue; }
      
      arma::mat Pg = Correlationf(coords, pxg, pxg, theta, bessel_ws, matern, true);
      arma::mat Pinv = arma::inv_sympd(Pg);
      Pinv_cache(g) = Pinv;
      
      if (grid_mode) {
        arma::uvec ixg = oneuv * child_cache(g);
        
        arma::mat CPt = Correlationf(coords, pxg, ixg, theta, bessel_ws, matern, false);
        std::vector<arma::mat> dCPt = Correlationf_grad(coords, pxg, ixg, theta, bessel_ws, matern, false);
        
        const double CCs = arma::as_scalar(Correlationf(coords, ixg, ixg, theta, bessel_ws, matern, true));
        std::vector<arma::mat> dCC = Correlationf_grad(coords, ixg, ixg, theta, bessel_ws, matern, true);
        
        arma::vec hg = Pinv * CPt;
        h_cache(g) = hg;
        
        const double CPCs = arma::as_scalar(CPt.t() * hg);
        const double Rg = CCs - CPCs;
        R_cache(g) = Rg;
        
        // need dPinv on the fly here (gridded still cheap because G small)
        std::vector<arma::mat> dP = Correlationf_grad(coords, pxg, pxg, theta, bessel_ws, matern, true);
        
        for (arma::uword t = 0; t < 4; ++t) {
          arma::mat dPinv = -Pinv * dP[t] * Pinv;
          arma::vec dhig  = dPinv * CPt + Pinv * dCPt[t];
          dhi_cache(g)[t] = dhig;
          
          const double dCPC = arma::as_scalar(dCPt[t].t() * hg + CPt.t() * dhig);
          dR_cache(g)[t] = arma::as_scalar(dCC[t]) - dCPC;
        }
      }
    } catch (...) {
      cache_error = true;
    }
  }
  if (cache_error) Rcpp::stop("Failed to compute caches in score_and_opg.");
  
  // 2) Main loop: accumulate score_u and OPG info (3x3)
  arma::vec score_u(3, arma::fill::zeros);
  arma::mat info_u(3, 3, arma::fill::zeros);
  
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  arma::vec score_local(3, arma::fill::zeros);
  arma::mat info_local(3, 3, arma::fill::zeros);
  
  // per-thread workspaces to reduce allocations
  arma::mat CPt;
  std::array<arma::mat,4> dCPt;
  std::array<double,4> dR;
  std::array<double,4> dCCs;
  arma::vec hi;
  std::array<arma::vec,4> dhi;
  arma::vec ypx;
  
#ifdef _OPENMP
#pragma omp for
#endif
  for (int i = 0; i < nr; ++i) {
    arma::vec gtheta_i(4, arma::fill::zeros); // per-node contribution to grad wrt theta
    
    try {
      const int gid = static_cast<int>(cache_map(i,1));
      arma::uvec px = grid_mode ? dag(i) : dag_cache(gid);
      const arma::uword m = px.n_elem;
      
      if (m == 0u) {
        // Root node
        arma::uvec ix = oneuv * i;
        const double CCs = arma::as_scalar(Correlationf(coords, ix, ix, theta, bessel_ws, matern, true));
        std::vector<arma::mat> dCC = Correlationf_grad(coords, ix, ix, theta, bessel_ws, matern, true);
        
        const double R = CCs;
        const double sqrtR = std::sqrt(R);
        const double invS  = 1.0 / sqrtR;
        const double ui    = invS * y(i);
        
        for (arma::uword t = 0; t < 4; ++t) {
          const double dRt = arma::as_scalar(dCC[t]);
          const double dlogdet = -(0.5 / R) * dRt;
          const double dinvS   = -0.5 * dRt / (R * sqrtR);
          const double dui     = dinvS * y(i);
          gtheta_i(t) = dlogdet - ui * dui;
        }
        
      } else {
        // non-root
        ypx = y.elem(px);
        const arma::mat& Pinv = Pinv_cache(gid);
        
        double R, numer;
        if (grid_mode) {
          hi    = h_cache(gid);
          R     = R_cache(gid);
          numer = y(i) - arma::dot(hi, ypx);
          for (arma::uword t=0; t<4; ++t) { dhi[t] = dhi_cache(gid)[t]; dR[t] = dR_cache(gid)[t]; }
        } else {
          arma::uvec ix = oneuv * i;
          
          CPt = Correlationf(coords, px, ix, theta, bessel_ws, matern, false);
          std::vector<arma::mat> dCPt_v = Correlationf_grad(coords, px, ix, theta, bessel_ws, matern, false);
          for (arma::uword t=0; t<4; ++t) dCPt[t] = dCPt_v[t];
          
          const double CCs = arma::as_scalar(Correlationf(coords, ix, ix, theta, bessel_ws, matern, true));
          std::vector<arma::mat> dCC = Correlationf_grad(coords, ix, ix, theta, bessel_ws, matern, true);
          for (arma::uword t=0; t<4; ++t) dCCs[t] = arma::as_scalar(dCC[t]);
          
          hi = Pinv * CPt;
          const double CPCs = arma::as_scalar(CPt.t() * hi);
          R = CCs - CPCs;
          numer = y(i) - arma::dot(hi, ypx);
          
          std::vector<arma::mat> dP = Correlationf_grad(coords, px, px, theta, bessel_ws, matern, true);
          
          for (arma::uword t=0; t<4; ++t) {
            arma::mat dPinv = -Pinv * dP[t] * Pinv;
            dhi[t] = dPinv * CPt + Pinv * dCPt[t];
            
            const double dCPC = arma::as_scalar(dCPt[t].t() * hi + CPt.t() * dhi[t]);
            dR[t] = dCCs[t] - dCPC;
          }
        }
        
        const double sqrtR = std::sqrt(R);
        const double invS  = 1.0 / sqrtR;
        const double ui    = numer * invS;
        
        for (arma::uword t=0; t<4; ++t) {
          const double dlogdet = -(0.5 / R) * dR[t];
          const double dnumer  = -arma::dot(dhi[t], ypx);
          const double dinvS   = -0.5 * dR[t] / (R * sqrtR);
          const double dui     = dnumer * invS + numer * dinvS;
          gtheta_i(t) = dlogdet - ui * dui;
        }
      }
      
      // ---- transform per-node score to u-space (3) ----
      arma::vec gu_i(3, arma::fill::zeros);
      gu_i(0) = gtheta_i(0) * phi;                 // d/d logphi
      gu_i(1) = gtheta_i(1) * sig2;                // d/d logsig2
      gu_i(2) = gtheta_i(3) * (alpha * (1.0-alpha)); // d/d logit(alpha)
      
      // accumulate
      score_local += gu_i;
      info_local  += gu_i * gu_i.t();
      
    } catch (...) {
      Rcpp::stop("Failure in score_and_opg main loop.");
    }
  }
  
#ifdef _OPENMP
#pragma omp critical
#endif
{
  score_u += score_local;
  info_u  += info_local;
}
}

return Rcpp::List::create(
  Rcpp::Named("loglik")  = ll,
  Rcpp::Named("score_u") = score_u,
  Rcpp::Named("info_u")  = info_u,
  Rcpp::Named("u")       = arma::vec({u0,u1,u2})
);
}

Rcpp::List DagGP::fit_fisher_matern_halfint(const arma::vec& y,
                                            int n_iter,
                                            double step0,
                                            double lambda_lphi,
                                            double lambda_lsig,
                                            double lambda_a,
                                            bool verbose) {
  if (theta.n_elem < 4) Rcpp::stop("theta must be (phi, sigmasq, nu, alpha).");
  
  const double nu = theta(2);
  const bool ok_nu =
    (std::abs(nu - 0.5) < 1e-12) ||
    (std::abs(nu - 1.5) < 1e-12) ||
    (std::abs(nu - 2.5) < 1e-12);
  if (!ok_nu) Rcpp::stop("Only nu in {0.5,1.5,2.5} supported.");
  
  auto logistic = [](double a) {
    if (a >= 0) { double z = std::exp(-a); return 1.0 / (1.0 + z); }
    else        { double z = std::exp(a);  return z / (1.0 + z); }
  };
  auto logit = [](double p) {
    const double eps = 1e-12;
    const double pp = std::min(1.0 - eps, std::max(eps, p));
    return std::log(pp) - std::log(1.0 - pp);
  };
  
  // ---- stopping rule parameters ----
  const double tol_obj_rel = 1e-7;  // relative objective improvement
  const double tol_step    = 1e-6;  // ||Δu||
  const double tol_grad    = 1e-4;  // ||score_u|| (after penalty)
  const int    patience    = 5;     // consecutive small-improvement iters
  
  // initialize unconstrained u = (logphi, logsig2, logit(alpha))
  arma::vec u(3);
  u(0) = std::log(std::max(theta(0), 1e-12));
  u(1) = std::log(std::max(theta(1), 1e-12));
  u(2) = logit(theta(3));
  
  // penalty center = initial u (prevents implicit shrink-to phi=1, sig2=1, alpha=0.5)
  arma::vec u_center = u;
  
  auto set_theta_from_u = [&](const arma::vec& uu) {
    arma::vec th = theta;
    th(0) = std::exp(uu(0));         // phi
    th(1) = std::exp(uu(1));         // sigmasq
    th(2) = nu;                      // fixed
    th(3) = logistic(uu(2));         // alpha in (0,1)
    update_theta(th, /*update_H=*/false);
  };
  
  auto penalty = [&](const arma::vec& uu) {
    return 0.5 * lambda_lphi * std::pow(uu(0) - u_center(0), 2.0)
    + 0.5 * lambda_lsig * std::pow(uu(1) - u_center(1), 2.0)
    + 0.5 * lambda_a    * std::pow(uu(2) - u_center(2), 2.0);
  };
  
  // initial objective
  set_theta_from_u(u);
  double ll0  = logdens_fast(y);
  double obj0 = ll0 - penalty(u);
  
  // history
  arma::vec obj_hist(n_iter, arma::fill::zeros);
  arma::mat u_hist(3, n_iter, arma::fill::zeros);
  arma::mat theta_hist(4, n_iter, arma::fill::zeros);
  arma::vec gradnorm_hist(n_iter, arma::fill::zeros);
  arma::vec stepnorm_hist(n_iter, arma::fill::zeros);
  arma::ivec ls_hist(n_iter, arma::fill::zeros);
  
  double obj_prev = obj0;
  int no_improve = 0;
  int it_used = 0;
  
  for (int it = 0; it < n_iter; ++it) {
    // keep model state aligned
    set_theta_from_u(u);
    
    // score + OPG info (must be implemented)
    Rcpp::List S = score_and_opg_matern_halfint(y);
    arma::vec score_u = Rcpp::as<arma::vec>(S["score_u"]); // length 3
    arma::mat info_u  = Rcpp::as<arma::mat>(S["info_u"]);  // 3x3
    
    // penalty gradient in u-space
    arma::vec gpen(3, arma::fill::zeros);
    gpen(0) = lambda_lphi * (u(0) - u_center(0));
    gpen(1) = lambda_lsig * (u(1) - u_center(1));
    gpen(2) = lambda_a    * (u(2) - u_center(2));
    score_u -= gpen;
    
    // penalty Hessian + ridge
    info_u(0,0) += lambda_lphi;
    info_u(1,1) += lambda_lsig;
    info_u(2,2) += lambda_a;
    const double ridge = 1e-8;
    info_u(0,0) += ridge; info_u(1,1) += ridge; info_u(2,2) += ridge;
    
    const double gradnorm = arma::norm(score_u, 2);
    
    // Fisher scoring step
    arma::vec delta = arma::solve(info_u, score_u, arma::solve_opts::fast);
    
    // line search (backtracking) on objective
    arma::vec u_old = u;
    double step = step0;
    arma::vec u_new = u;
    double obj_new = obj0;
    int ls_used = 0;
    
    for (; ls_used < 20; ++ls_used) {
      u_new = u_old + step * delta;
      
      set_theta_from_u(u_new);
      const double ll_new = logdens_fast(y);
      obj_new = ll_new - penalty(u_new);
      
      if (obj_new >= obj0) break;
      step *= 0.5;
    }
    
    // if line search fails, stop (you can loosen this if you want)
    if (ls_used == 20) {
      if (verbose) Rcpp::Rcout << "Stopping: line search failed.\n";
      break;
    }
    
    // accept
    u = u_new;
    obj0 = obj_new;
    
    const double stepnorm = arma::norm(u - u_old, 2);
    
    // store
    obj_hist(it) = obj0;
    u_hist.col(it) = u;
    
    arma::vec th(4);
    th(0) = std::exp(u(0));
    th(1) = std::exp(u(1));
    th(2) = nu;
    th(3) = logistic(u(2));
    theta_hist.col(it) = th;
    
    gradnorm_hist(it) = gradnorm;
    stepnorm_hist(it) = stepnorm;
    ls_hist(it) = ls_used;
    
    it_used = it + 1;
    
    if (verbose && (it % 10 == 0)) {
      Rcpp::Rcout << "it=" << it
                  << " obj=" << obj0
                  << " grad=" << gradnorm
                  << " step=" << stepnorm
                  << " ls=" << ls_used
                  << " phi=" << th(0)
                  << " sig2=" << th(1)
                  << " alpha=" << th(3)
                  << "\n";
    }
    
    // ---- stopping rules ----
    if (gradnorm < tol_grad) {
      if (verbose) Rcpp::Rcout << "Stopping: grad norm < tol_grad.\n";
      break;
    }
    
    if (stepnorm < tol_step) {
      if (verbose) Rcpp::Rcout << "Stopping: step norm < tol_step.\n";
      break;
    }
    
    const double rel = std::abs(obj0 - obj_prev) / (1.0 + std::abs(obj_prev));
    if (rel < tol_obj_rel) {
      no_improve++;
      if (no_improve >= patience) {
        if (verbose) Rcpp::Rcout << "Stopping: objective improvement below tol (patience).\n";
        break;
      }
    } else {
      no_improve = 0;
    }
    obj_prev = obj0;
  }
  
  // truncate histories
  arma::vec obj_hist2 = obj_hist.head(it_used);
  arma::mat u_hist2 = u_hist.cols(0, std::max(0, it_used - 1)).t();        // it_used x 3
  arma::mat theta_hist2 = theta_hist.cols(0, std::max(0, it_used - 1)).t(); // it_used x 4
  arma::vec grad_hist2 = gradnorm_hist.head(it_used);
  arma::vec step_hist2 = stepnorm_hist.head(it_used);
  arma::ivec ls_hist2 = ls_hist.head(it_used);
  
  // final theta
  arma::vec theta_hat(4);
  theta_hat(0) = std::exp(u(0));
  theta_hat(1) = std::exp(u(1));
  theta_hat(2) = nu;
  theta_hat(3) = logistic(u(2));
  update_theta(theta_hat, /*update_H=*/false);
  
  return Rcpp::List::create(
    Rcpp::Named("theta_hat") = theta_hat,
    Rcpp::Named("u_hat")     = u,
    Rcpp::Named("n_iter")    = it_used,
    Rcpp::Named("obj_hist")  = obj_hist2,
    Rcpp::Named("u_hist")    = u_hist2,
    Rcpp::Named("theta_hist")= theta_hist2,
    Rcpp::Named("gradnorm")  = grad_hist2,
    Rcpp::Named("stepnorm")  = step_hist2,
    Rcpp::Named("ls_used")   = ls_hist2
  );
}

// [[Rcpp::export]]
Rcpp::List daggp_fit_matern_halfint_fisher(const arma::mat& coords,
                                           const arma::field<arma::uvec>& dag,
                                           const arma::vec& y,
                                           double phi, double sigmasq, double nu, double alpha,
                                           int num_threads = 1,
                                           int dag_opts = 0,
                                           int n_iter = 100,
                                           double step0 = 1.0,
                                           double lambda_lphi = 1e-2,
                                           double lambda_lsig = 1e-2,
                                           double lambda_a    = 1e-2,
                                           bool verbose = false) {
  // theta = (phi, sigmasq, nu, alpha)
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = alpha;
  
  // covariance_matern = 1 -> Matérn (your code)
  const int covariance_matern = 1;
  
  // build model
  DagGP model(coords, theta, dag, dag_opts, covariance_matern, /*use_Ci_in=*/false, num_threads);
  
  // fit using Fisher scoring / OPG routine you added to DagGP
  Rcpp::List fit = model.fit_fisher_matern_halfint(
    y, n_iter, step0, lambda_lphi, lambda_lsig, lambda_a, verbose
  );
  
  // return fitted params + some useful diagnostics
  arma::vec theta_hat = fit["theta_hat"];
  
  return Rcpp::List::create(
    Rcpp::Named("theta_hat") = theta_hat,
    Rcpp::Named("phi_hat")   = theta_hat(0),
    Rcpp::Named("sigmasq_hat") = theta_hat(1),
    Rcpp::Named("nu")        = theta_hat(2),
    Rcpp::Named("alpha_hat") = theta_hat(3),
    Rcpp::Named("obj_hist")  = fit["obj_hist"]
  );
}

/// ADAM
Rcpp::List DagGP::adam_fit_matern_halfint(const arma::vec& y,
                                          int n_iter,
                                          double lr,
                                          double beta1,
                                          double beta2,
                                          double eps,
                                          double lambda_lphi,
                                          double lambda_lsig,
                                          double lambda_a,
                                          bool verbose) {
  if (matern != 1) Rcpp::stop("adam_fit_matern_halfint: matern must be 1 (Matérn).");
  if (theta.n_elem < 4) Rcpp::stop("theta must be (phi, sigmasq, nu, alpha).");
  
  const double nu = theta(2);
  const bool ok_nu =
    (std::abs(nu - 0.5) < 1e-12) ||
    (std::abs(nu - 1.5) < 1e-12) ||
    (std::abs(nu - 2.5) < 1e-12);
  if (!ok_nu) Rcpp::stop("adam_fit_matern_halfint: only nu in {0.5,1.5,2.5} supported.");
  
  // Unconstrained params:
  // u0 = log(phi), u1 = log(sigmasq), u2 = logit(alpha)
  arma::vec u(3);
  u(0) = std::log(std::max(theta(0), 1e-12));
  u(1) = std::log(std::max(theta(1), 1e-12));
  u(2) = logit(theta(3));
  
  arma::vec m(3, arma::fill::zeros);
  arma::vec v(3, arma::fill::zeros);
  
  arma::vec obj_hist(n_iter, arma::fill::zeros);
  
  // --- Early Stopping Setup ---
  double prev_obj = -1e300;
  int patience_counter = 0;
  const int patience_limit = 15; // Number of iterations to wait before giving up
  const double tol = 1e-4;       // Minimum required change in the objective function
  
  for (int it = 0; it < n_iter; ++it) {
    // map back to constrained theta
    const double phi     = std::exp(u(0));
    const double sigmasq = std::exp(u(1));
    const double alpha   = logistic(u(2));
    
    arma::vec th = theta;
    th(0) = phi;
    th(1) = sigmasq;
    th(2) = nu;     // fixed
    th(3) = alpha;
    
    // update model comps (no need to rebuild sparse H)
    update_theta(th, /*update_H=*/false);
    
    // loglik (fast)
    const double ll = logdens_fast(y);
    
    // penalties on unconstrained params
    const double pen = 0.5 * (lambda_lphi * u(0)*u(0) +
                              lambda_lsig * u(1)*u(1) +
                              lambda_a    * u(2)*u(2));
    const double obj = ll - pen; // maximize
    obj_hist(it) = obj;
    
    // --- Exit Rule Logic ---
    if (it > 0) {
      double delta = std::abs(obj - prev_obj);
      if (delta < tol) {
        patience_counter++;
      } else {
        patience_counter = 0; // Reset if we take a meaningful step
      }
      
      if (patience_counter >= patience_limit) {
        if (verbose) {
          Rcpp::Rcout << "Early stopping triggered at iteration " << it 
                      << " (Objective changed by < " << tol 
                      << " for " << patience_limit << " consecutive iterations).\n";
        }
        obj_hist = obj_hist.head(it + 1); // Truncate the history array so we don't return trailing zeros
        break;
      }
    }
    prev_obj = obj;
    // -----------------------
    
    // gradient wrt constrained theta
    arma::vec gth = grad_theta_loglik_matern_halfint(y); // length 4
    
    // chain rule to unconstrained u
    arma::vec gu(3, arma::fill::zeros);
    
    // d/d log(phi) = d/dphi * phi
    gu(0) = gth(0) * phi;
    
    // d/d log(sigmasq) = d/dsigmasq * sigmasq
    gu(1) = gth(1) * sigmasq;
    
    // d/d logit(alpha) = d/dalpha * alpha(1-alpha)
    gu(2) = gth(3) * (alpha * (1.0 - alpha));
    
    // subtract penalty gradient (since obj = ll - 0.5*lambda*u^2)
    gu(0) -= lambda_lphi * u(0);
    gu(1) -= lambda_lsig * u(1);
    gu(2) -= lambda_a    * u(2);
    
    // Adam update (ascent)
    m = beta1 * m + (1.0 - beta1) * gu;
    v = beta2 * v + (1.0 - beta2) * (gu % gu);
    
    const double b1t = 1.0 - std::pow(beta1, it + 1.0);
    const double b2t = 1.0 - std::pow(beta2, it + 1.0);
    
    arma::vec mhat = m / b1t;
    arma::vec vhat = v / b2t;
    
    u += lr * (mhat / (arma::sqrt(vhat) + eps));
    
    if (verbose && ((it % 25) == 0)) {
      Rcpp::Rcout << "it=" << it
                  << " obj=" << obj
                  << " ll=" << ll
                  << " phi=" << phi
                  << " sigmasq=" << sigmasq
                  << " alpha=" << alpha
                  << "\n";
    }
  }
  
  // final theta
  arma::vec theta_hat = theta;
  theta_hat(0) = std::exp(u(0));
  theta_hat(1) = std::exp(u(1));
  theta_hat(2) = nu;
  theta_hat(3) = logistic(u(2));
  
  // update model to final theta
  update_theta(theta_hat, /*update_H=*/false);
  
  return Rcpp::List::create(
    Rcpp::Named("theta_hat") = theta_hat,
    Rcpp::Named("u_hat")     = u,
    Rcpp::Named("obj_hist")  = obj_hist
  );
}

//[[Rcpp::export]]
Rcpp::List daggp_fit_matern_halfint(const arma::mat& coords,
                                    const arma::field<arma::uvec>& dag,
                                    const arma::vec& y,
                                    double phi, double sigmasq, double nu, double alpha,
                                    int num_threads=1, int dag_opts=0,
                                    int n_iter=200, double lr=0.02,
                                    double lambda_lphi=1e-2, double lambda_lsig=1e-2, double lambda_a=1e-2,
                                    bool verbose=false) {
  
  arma::vec th(4);
  th(0) = phi;
  th(1) = sigmasq;
  th(2) = nu;
  th(3) = alpha;
  
  DagGP model(coords, th, dag, dag_opts, /*matern=*/1, /*use_Ci=*/false, num_threads);
  return model.adam_fit_matern_halfint(y, n_iter, lr, 0.9, 0.999, 1e-8,
                                       lambda_lphi, lambda_lsig, lambda_a, verbose);
}




/*

using EigenFromArma = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
 
arma::sp_mat sparse_convert(const Eigen::SparseMatrix<double>& XE){
 unsigned int num_cols = XE.cols();
 unsigned int num_rows = XE.rows();
 
 std::vector<unsigned int> rowind_vect(
 XE.innerIndexPtr(), XE.innerIndexPtr() + XE.nonZeros());
 std::vector<unsigned int> colptr_vect(
 XE.outerIndexPtr(), XE.outerIndexPtr() + XE.outerSize() + 1);
 std::vector<double> values_vect(
 XE.valuePtr(), XE.valuePtr() + XE.nonZeros());
 
 arma::vec values(values_vect.data(), values_vect.size(), false);
 arma::uvec rowind(rowind_vect.data(), rowind_vect.size(), false);
 arma::uvec colptr(colptr_vect.data(), colptr_vect.size(), false);
 
 arma::sp_mat result(rowind, colptr, values, num_rows, num_cols);
 
 return result; 
} */