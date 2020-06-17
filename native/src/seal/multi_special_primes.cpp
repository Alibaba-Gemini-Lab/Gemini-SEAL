#include "seal/util/common.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.h"

struct MulModShoup {
  uint64_t cnst, p;
  uint64_t cnst_shoup;

  explicit MulModShoup(uint64_t cnst, uint64_t p) : cnst(cnst), p(p) {
	uint64_t cnst_128[2]{0, cnst};
	uint64_t shoup[2];
	util::divide_uint128_inplace(cnst_128, p, shoup);
	cnst_shoup = shoup[0]; // cnst_shoup = cnst * 2^64 / p
  }

  inline uint64_t operator()(uint64_t x) const {
	unsigned long long hw64;
	multiply_uint64_hw64(x, cnst_shoup, &hw64);
	uint64_t q = static_cast<uint64_t>(hw64) * p;
	uint64_t t = (x * cnst - q);
	return t - ((p & -static_cast<uint64_t>(t < p)) ^ p);
  }
};

struct FMAU128 {
  // acc += op0 * op1
  static inline void apply(uint64_t*acc, uint64_t op0, uint64_t op1) {
	multiply_accumulate_uint64<1>(&op0, &op1, acc);
  }
};

/**
 * Faster cnst * (op0 - op1) mod p  for fixed constant cnst.
 *        cnst * (op0 + op1) mod p  for fixed constant cnst.
 */
struct PolyMulConstant {
  explicit PolyMulConstant(uint64_t cnst, seal::Modulus const &mod) : mod(mod), shoup(cnst, mod.value()) {
  }

  template <class Func>
	void apply(const uint64_t *op0, const uint64_t *op1, uint64_t *dst, size_t degree) {
	  if (!op0 || !op1 || !dst || degree == 0)
		return;

	  Func functor(shoup.p);

	  for (size_t l = 0; l < degree; ++l, ++op0, ++op1) {
		*dst++ = shoup(functor(*op0, *op1));
	  }
	}

  // dst <- cnt * (op0 - op1)
  void poly_sub(const uint64_t *op0, const uint64_t *op1, uint64_t *dst, size_t degree) {
	struct Sub {
	  const uint64_t p;
	  Sub(uint64_t p) : p(p) {}

	  inline uint64_t operator()(uint64_t a, uint64_t b) {
		return a - b + p;
	  }
	};
	apply<Sub>(op0, op1, dst, degree);
  }

  // dst <- cnt * (op0 + op1)
  void poly_add(const uint64_t *op0, const uint64_t *op1, uint64_t *dst, size_t degree) {
	struct Add {
	  const uint64_t p;
	  Add (uint64_t p) : p(p) {}

	  inline uint64_t operator()(uint64_t a, uint64_t b) {
		return a + b;
	  }
	};

	apply<Add>(op0, op1, dst, degree);
  }

  seal::Modulus const& mod;
  MulModShoup shoup;
};

void modup_to_single_rns(const uint64_t *in_poly, 
						 uint64_t *dst_poly,
						 const size_t degree,
						 const std::vector<size_t> &in_poly_rns_indices,
						 const size_t dst_rns_index,
						 const std::vector<Modulus> &key_rns,
						 MemoryPool &pool) {
  const size_t n_all_rns = key_rns.size();
  if (!in_poly || !dst_poly || dst_rns_index >= n_all_rns || in_poly_rns_indices.empty()) {
	throw std::invalid_argument("modup_to_single_rns: invalid_argument");
  }

  for (size_t idx : in_poly_rns_indices) {
	if (idx >= n_all_rns)
	  throw std::invalid_argument("modup_to_single_rns: invalid_argument");
	if (idx == dst_rns_index)
	  throw std::invalid_argument("modup_to_single_rns: invalid_argument");
  }

  if (in_poly_rns_indices.size() == 1) {
	if (key_rns[in_poly_rns_indices[0]].value() <= key_rns[dst_rns_index].value()) {
	  std::memcpy(dst_poly, in_poly, sizeof(uint64_t) * degree);
	} else {
	  auto const& rns = key_rns[dst_rns_index];
	  std::transform(in_poly, in_poly + degree, dst_poly, 
					 [rns](uint64_t v) -> uint64_t {
					 return barrett_reduce_63(v, rns);
					 });
	}
  } else {
	std::vector<uint64_t> inv_punch_prod;
	std::vector<uint64_t> punch_prod;

	for (size_t punch_idx : in_poly_rns_indices) {
	  uint64_t inv_prod {1}, prod {1};
	  for (size_t rns_idx : in_poly_rns_indices) {
		if (punch_idx == rns_idx) continue;
		prod = multiply_uint_mod(prod, key_rns[rns_idx].value(), key_rns[dst_rns_index]);
		inv_prod = multiply_uint_mod(inv_prod, key_rns[rns_idx].value(), key_rns[punch_idx]);
	  }

	  punch_prod.push_back(prod);
	  if (!util::try_invert_uint_mod(inv_prod, key_rns[punch_idx], inv_prod)) {
		throw std::runtime_error("modup_to_single_rns: inv_mod fail");
	  }
	  inv_punch_prod.push_back(inv_prod);
	}

	Pointer<uint64_t> accum{ allocate_zero_poly(degree * 2, 1, pool) };
	for (size_t i = 0; i < in_poly_rns_indices.size(); ++i) {
	  const size_t rns_idx = in_poly_rns_indices[i];
	  const uint64_t *in_poly_ptr = in_poly + i * degree;

	  uint64_t *accum_ptr = accum.get();
	  FMAU128 fma;
	  MulModShoup mulmod(inv_punch_prod.at(i), key_rns[rns_idx].value());

	  for (size_t d = 0; d < degree; ++d, accum_ptr += 2) {
		// Use Shoup's trick to accelerate (* cnst mod pi) for fixed constant.
		fma.apply(accum_ptr, mulmod(*in_poly_ptr++), punch_prod.at(i));
	  }
	}

	uint64_t *accum_ptr = accum.get();
	for (size_t d = 0; d < degree; ++d, accum_ptr += 2) {
	  *dst_poly++ = barrett_reduce_128(accum_ptr, key_rns[dst_rns_index]);
	}
  }
}

/// [src_poly]_{p_{i}, p_{i+1}, ...} -> [dst_poly]_{p_0, p1, p_2, ..., q_0, q_1, ...}
void modup_rns(const uint64_t *src_poly, 
			   uint64_t *dst_poly,
			   const size_t degree,
			   const size_t n_ct_rns, 
			   const size_t n_sp_rns, 
			   const size_t src_bundle_index,
			   std::vector<Modulus> const& key_modulus,
			   MemoryPool &pool) {
  const size_t n_bundles = (n_ct_rns + n_sp_rns - 1) / n_sp_rns;
  if (src_bundle_index >= n_bundles) {
	std::invalid_argument("modup_rns: src_bundle_index out of bound");
  }

  size_t rns0 = src_bundle_index * n_sp_rns;
  size_t rns1 = std::min(rns0 + n_sp_rns, n_ct_rns);
  std::vector<size_t> src_rns_indices(rns1 - rns0);
  std::iota(src_rns_indices.begin(), src_rns_indices.end(), rns0);

  for (size_t bundle_idx = 0; bundle_idx < n_bundles; ++bundle_idx) {
	if (bundle_idx != src_bundle_index) {
	  size_t dst_rns0 = bundle_idx * n_sp_rns;
	  size_t dst_rns1 = std::min(dst_rns0 + n_sp_rns, n_ct_rns);
	  uint64_t *dst_ptr = dst_poly + dst_rns0 * degree;
	  for (size_t dst_rns_idx = dst_rns0; dst_rns_idx < dst_rns1; ++dst_rns_idx, dst_ptr += degree) {
		modup_to_single_rns(src_poly, dst_ptr, degree, src_rns_indices, dst_rns_idx, key_modulus, pool);
	  }
	}
  }

  const size_t sp0_index = key_modulus.size() - n_sp_rns;
  for (size_t k = 0; k < n_sp_rns; ++k) {
	uint64_t *dst_ptr = dst_poly + (n_ct_rns + k) * degree;
	modup_to_single_rns(src_poly, dst_ptr, degree, src_rns_indices, sp0_index + k, key_modulus, pool);
  }
}

std::vector<uint64_t> puncture_products_of_special_rns(std::vector<Modulus> const& key_rns,
														   size_t n_special_rns,
														   size_t normal_rns_index)
{
  if (n_special_rns > key_rns.size() || normal_rns_index >= key_rns.size()
	  || (normal_rns_index + n_special_rns >= key_rns.size())) {
	throw std::invalid_argument("puncture_products_of_special_rns : invalid_argument");
  }

  const size_t sp0_index = key_rns.size() - n_special_rns;
  std::vector<uint64_t> puncture_products(n_special_rns);
  for (size_t i = 0; i < n_special_rns; ++i) {
	uint64_t prod{1};
	for (size_t j = 0; j < n_special_rns; ++j) {
	  if (i == j) continue; // puncture
	  prod = multiply_uint_mod(prod, key_rns[sp0_index + j].value(), key_rns[normal_rns_index]);
	}
    puncture_products[i] = prod;
  }

  return puncture_products;
}

std::vector<uint64_t> inv_puncture_products_of_special_rns(std::vector<Modulus> const& key_rns,
														   size_t n_special_rns)
{
  if (n_special_rns > key_rns.size()) {
	throw std::invalid_argument("inv_puncture_products_of_special_rns: invalid_argument");
  }

  const size_t sp0_index = key_rns.size() - n_special_rns;
  std::vector<uint64_t> puncture_products(n_special_rns);
  for (size_t i = 0; i < n_special_rns; ++i) {
	uint64_t prod{1};
	for (size_t j = 0; j < n_special_rns; ++j) {
	  if (i == j) continue; // puncture
	  prod = multiply_uint_mod(prod, key_rns[sp0_index + j].value(), key_rns[sp0_index + i]);
	}

	uint64_t inv_prod;
	if (!util::try_invert_uint_mod(prod, key_rns[sp0_index + i], inv_prod)) {
	  throw std::runtime_error("inv_puncture_products_of_special_rns: inv_mod fail");
	}

	puncture_products[i] = inv_prod;
  }

  return puncture_products;
}
/// Rescale the special rns part of poly.
/// Require: the special rns part is in the power-basis form.
void rescale_special_rns_inplace(uint64_t *poly, bool is_ckks,
								 const size_t degree, const size_t n_ct_rns, const size_t n_sp_rns,
								 std::vector<Modulus> const& key_rns, 
								 const NTTTables *small_ntt_tables,
								 MemoryPool &pool)
{
  const size_t sp_rns_index0 = key_rns.size() - n_sp_rns;
  std::vector<uint64_t> inv_hat_pj_pj = inv_puncture_products_of_special_rns(key_rns, n_sp_rns);
  std::vector<std::vector<uint64_t>> hat_pj_qi(n_ct_rns);
  for (size_t i = 0; i < n_ct_rns; ++i) {
	hat_pj_qi[i] = puncture_products_of_special_rns(key_rns, n_sp_rns, i);
  }

  Pointer<uint64_t> temp_poly{ allocate_uint(degree, pool) };
  Pointer<uint64_t> lazy_mult{ allocate_uint(2 * degree, pool) };

  for (size_t i = 0; i < n_ct_rns; ++i) {
	// Step 1: \sum_j { ([ct]_{pj} * \hat{pj}^{-1} mod pj) * (-\hat{pj} mod qi) }
    if (n_sp_rns > 1) {
      std::memset(lazy_mult.get(), 0, sizeof(uint64_t) * 2 * degree);

      for (size_t j = 0; j < n_sp_rns; ++j) {
        uint64_t *lazy_mult_ptr = lazy_mult.get();
        const uint64_t *ct_ptr = poly + (n_ct_rns + j) * degree;
        // Optimization 1) multiplication \hat{pj}^{-1} uses Shoup's trick for accleration.
        // Optimization 2) multiplication -\hat{pj} mod qi uses lazy reduction.
        MulModShoup mulmod(inv_hat_pj_pj[j], key_rns[sp_rns_index0 + j].value());
        FMAU128 fma;
        const uint64_t hat_pj_qi_ = hat_pj_qi.at(i).at(j);
        for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
          fma.apply(lazy_mult_ptr, mulmod(*ct_ptr++), hat_pj_qi_);
        }
      }

      // Step 2: lazy reduction
      uint64_t *lazy_mult_ptr = lazy_mult.get();
      for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
        temp_poly[l] = barrett_reduce_128(lazy_mult_ptr, key_rns[i]);
      }

    } else {
      // For case that n_sp_rns = 1, \hat{pj} = 1, so we just omit the multiplication
      const uint64_t *ct_ptr = poly + n_ct_rns * degree;

      if (key_rns[sp_rns_index0].value() > key_rns[i].value()) {
        modulo_poly_coeffs_63(ct_ptr, degree, key_rns[i], temp_poly.get());
      } else {
        set_uint(ct_ptr, degree, temp_poly.get());
      }

      const uint64_t qk_half = key_rns[sp_rns_index0].value() >> 1;
      const uint64_t fix = key_rns[i].value() - barrett_reduce_63(qk_half, key_rns[i].value());
      SEAL_ITERATE(temp_poly.get(), degree, [fix](uint64_t &K) { K += fix; });
    }

    // Step 3: convert to proper form.
    uint64_t qi_lazy = key_rns[i].value() << 1;
    if (is_ckks) {
      // [0, 4p)
      ntt_negacyclic_harvey_lazy(temp_poly.get(), small_ntt_tables[i]);
#if SEAL_USER_MOD_BIT_COUNT_MAX > 60
      // Reduce from [0, 4qi) to [0, 2qi)
      SEAL_ITERATE(temp_poly.get(), degree, [qi_lazy](uint64_t &K) {
                   K -= (qi_lazy & static_cast<uint64_t>(-static_cast<int64_t>(K >= qi_lazy)));
                   });
#else
      // Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
      qi_lazy = key_rns[i].value() << 2;
#endif
    } else {
      inverse_ntt_negacyclic_harvey_lazy(poly + i * degree, small_ntt_tables[i]);
    }

    // Step 4: [P^{-1}]_{q_i} * ([c]_{q_i} + [c']_{q_i}) mod q_i
    uint64_t P_qi{1};
    for (size_t j = 0; j < n_sp_rns; ++j) {
      P_qi = multiply_uint_mod(P_qi, key_rns[sp_rns_index0 + j].value(), key_rns[i]);
    }

    uint64_t invP_qi;
    if (!util::try_invert_uint_mod(P_qi, key_rns[i], invP_qi)) {
      throw std::runtime_error("rescale_special_rns_inplace: inv_mod fail");
    }

    SEAL_ITERATE(iter(poly + i * degree, temp_poly.get()), 
                 degree, [qi_lazy](auto K) { get<0>(K) += qi_lazy - get<1>(K); });
    multiply_poly_scalar_coeffmod(poly + i * degree, degree, invP_qi, key_rns[i], poly + i * degree);
  }
}

void Evaluator::switch_key_inplace(Ciphertext &encrypted, ConstRNSIter target_iter, const KSwitchKeys &kswitch_keys, size_t kswitch_keys_index, MemoryPoolHandle pool)
{
  auto parms_id = encrypted.parms_id();
  auto &context_data = *context_->get_context_data(parms_id);
  auto &parms = context_data.parms();
  auto &key_context_data = *context_->key_context_data();
  auto &key_parms = key_context_data.parms();
  auto &key_modulus = key_parms.coeff_modulus();
  auto small_ntt_tables = key_context_data.small_ntt_tables();
  auto scheme = parms.scheme();

  // Verify parameters.
  const size_t degree  = parms.poly_modulus_degree();
  const size_t n_ct_rns = parms.coeff_modulus().size();
  const size_t n_ct_all_rns = context_->get_context_data(context_->first_parms_id())->parms().coeff_modulus().size();
  const size_t n_total_rns = key_modulus.size();
  const size_t n_special_rns = n_total_rns - n_ct_all_rns;
  const size_t n_bundles = (n_ct_rns + n_special_rns - 1) / n_special_rns;
  const bool is_ckks = scheme == scheme_type::CKKS;

  // Prepare input
  auto &key_vector = kswitch_keys.data()[kswitch_keys_index];

  // Check only the used component in KSwitchKeys.
  for (auto &each_key : key_vector)
  {
    if (!is_metadata_valid_for(each_key, context_) ||
        !is_buffer_valid(each_key))
    {
      throw invalid_argument("kswitch_keys is not valid for encryption parameters");
    }
  }

  Pointer<uint64_t> lazy_mult[2] { allocate_zero_poly(degree, 2 * (n_ct_rns + n_special_rns), pool),
    allocate_zero_poly(degree, 2 * (n_ct_rns + n_special_rns), pool) };

  Pointer<uint64_t> poly_ext_rns { allocate_poly(degree, n_ct_rns + n_special_rns, pool) };

  for (size_t src_bundle_idx = 0; src_bundle_idx < n_bundles; ++src_bundle_idx) {
    // Step 1: convert the current bundle to power-basis
    const size_t rns0 = src_bundle_idx * n_special_rns;
    const size_t rns1 = std::min(rns0 + n_special_rns, n_ct_rns);

    RNSIter dst_iter(poly_ext_rns.get(), degree);
    SEAL_ITERATE(iter(dst_iter + rns0, target_iter + rns0, small_ntt_tables + rns0), rns1 - rns0,
                 [degree, is_ckks](auto I) {
                 set_uint(get<1>(I), degree, get<0>(I));
                 if (is_ckks) inverse_ntt_negacyclic_harvey(get<0>(I), get<2>(I));
                 });

    // Step 2: modulus up, including all normal primes and special primes.
    modup_rns(poly_ext_rns.get() + rns0 * degree, poly_ext_rns.get(),
              degree, n_ct_rns, n_special_rns, src_bundle_idx, key_modulus, pool);

    // Step 3: Inner Product using lazy reduction.
    // Note, the number of rns should be less than 256 to prevent 128bit overflow.
    for (size_t k = 0; k < n_ct_rns + n_special_rns; ++k) {
      bool is_spcl_rns = k >= n_ct_rns;
      size_t rns_idx = is_spcl_rns ? n_ct_all_rns + k - n_ct_rns : k;
      const uint64_t *ct_ptr = nullptr;
      if (is_ckks && (k >= rns0 && k < rns1)) {
        ct_ptr = *(target_iter + k);
      } else {
        ntt_negacyclic_harvey_lazy(poly_ext_rns.get() + k * degree, small_ntt_tables[rns_idx]);
        ct_ptr = poly_ext_rns.get() + k * degree;
      }

      for (size_t l : {0, 1}) {
        const uint64_t *key_ptr = key_vector[src_bundle_idx].data().data(l) + rns_idx * degree;
        auto lazy_mult_ptr = lazy_mult[l].get() + k * (degree << 1);
        FMAU128 fma;
        auto _ct_ptr = ct_ptr;
        for (size_t d = 0; d < degree; ++d, lazy_mult_ptr += 2) {
          fma.apply(lazy_mult_ptr, *_ct_ptr++, *key_ptr++);
        }
      }
    } // k-loop
  } // j-loop

  for (size_t b : {0, 1}) {
    // Step 4: Lazy reduction
    uint64_t *cmult_evk_ptr = poly_ext_rns.get();
    for (size_t k = 0; k < n_ct_rns + n_special_rns; ++k) {
      bool is_spcl_rns = k >= n_ct_rns;
      size_t rns_idx   = is_spcl_rns? n_ct_all_rns + k - n_ct_rns : k;

      auto lazy_mult_ptr = lazy_mult[b].get() + k * (degree << 1);
      uint64_t *dst_ptr = cmult_evk_ptr;
      for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
        *dst_ptr++ = barrett_reduce_128(lazy_mult_ptr, key_modulus[rns_idx]);
      }

      if (is_spcl_rns) {
        // Backward to power-basis for mod-down. Note that, we only convert the special prime part.
        // The following bext.mod_down() operation will take care of the normal part.
        inverse_ntt_negacyclic_harvey_lazy(cmult_evk_ptr, small_ntt_tables[rns_idx]);
        if (n_special_rns == 1) {
          // Add (p-1)/2 to change from flooring to rounding.
          const uint64_t qk = key_modulus[rns_idx].value();
          const uint64_t qk_half = qk >> 1;
          const auto& modulus = key_modulus[rns_idx];
          SEAL_ITERATE(cmult_evk_ptr, degree, [qk_half, &modulus](uint64_t &J) {
                       J = barrett_reduce_63(J + qk_half, modulus);
                       });
        }
      }
      cmult_evk_ptr += degree;
    }

    cmult_evk_ptr = poly_ext_rns.get();
    // Step 5: Rescale Down then Add
    rescale_special_rns_inplace(cmult_evk_ptr, is_ckks, degree, n_ct_rns, 
                                n_special_rns, key_modulus, small_ntt_tables, pool);

    uint64_t *encrypted_ptr = encrypted.data(b);
    for (size_t i = 0; i < n_ct_rns; i++, encrypted_ptr += degree, cmult_evk_ptr += degree) {
      add_poly_coeffmod(cmult_evk_ptr, encrypted_ptr, degree, key_modulus[i], encrypted_ptr);
    }
  }
}
