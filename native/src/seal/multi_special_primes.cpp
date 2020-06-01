namespace seal {
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
std::vector<uint64_t> neg_puncture_products_of_special_rns(std::vector<Modulus> const& key_rns,
														   size_t n_special_rns,
														   size_t normal_rns_index)
{
	if (n_special_rns > key_rns.size() || normal_rns_index >= key_rns.size()
		|| (normal_rns_index + n_special_rns >= key_rns.size())) {
		throw std::invalid_argument("neg_puncture_products_of_special_rns : invalid_argument");
	}

	const size_t sp0_index = key_rns.size() - n_special_rns;
	std::vector<uint64_t> neg_puncture_products(n_special_rns);
	for (size_t i = 0; i < n_special_rns; ++i) {
		uint64_t prod{1};
		for (size_t j = 0; j < n_special_rns; ++j) {
			if (i == j) continue; // puncture
			prod = multiply_uint_mod(prod, key_rns[sp0_index + j].value(), key_rns[normal_rns_index]);
		}
		neg_puncture_products[i] = negate_uint_mod(prod, key_rns[normal_rns_index]);
	}

	return neg_puncture_products;
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
	std::vector<std::vector<uint64_t>> neg_hat_pj_qi(n_ct_rns);
	for (size_t i = 0; i < n_ct_rns; ++i) {
		neg_hat_pj_qi[i] = neg_puncture_products_of_special_rns(key_rns, n_sp_rns, i);
	}

	Pointer<uint64_t> lazy_mult{ allocate_uint(2 * degree, pool) };
	Pointer<uint64_t> temp_poly{ allocate_uint(degree, pool) };

	for (size_t i = 0; i < n_ct_rns; ++i) {
		std::memset(lazy_mult.get(), 0, sizeof(uint64_t) * 2 * degree);
		// Step 1: \sum_j { ([ct]_{pj} * \hat{pj}^{-1} mod pj) * (-\hat{pj} mod qi) }
		for (size_t j = 0; j < n_sp_rns; ++j) {
			uint64_t *lazy_mult_ptr = lazy_mult.get();
			const uint64_t *ct_ptr = poly + (n_ct_rns + j) * degree;
			if (n_sp_rns > 1) {
				// Optimization 1) multiplication \hat{pj}^{-1} uses Shoup's trick for accleration.
				// Optimization 2) multiplication -\hat{pj} mod qi uses lazy reduction.
				MulModShoup mulmod(inv_hat_pj_pj[j], key_rns[sp_rns_index0 + j].value());
				FMAU128 fma;
				const uint64_t neg_hat_pj_qi_ = neg_hat_pj_qi.at(i).at(j);
				for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
					fma.apply(lazy_mult_ptr, mulmod(*ct_ptr++), neg_hat_pj_qi_);
				}
			} else {
				// For case that n_sp_rns = 1, \hat{pj} = 1, so we just omit the multiplication
				for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
					uint64_t v = negate_uint_mod(barrett_reduce_63(*ct_ptr++, key_rns[sp_rns_index0]), key_rns[sp_rns_index0]);
                    unsigned long long qword[2] = {v, 0};
					add_uint128(lazy_mult_ptr, qword, lazy_mult_ptr);
				}
			}
		}

		// Step 2: lazy reduction
		uint64_t *lazy_mult_ptr = lazy_mult.get();
		for (size_t l = 0; l < degree; l++, lazy_mult_ptr += 2) {
			temp_poly[l] = barrett_reduce_128(lazy_mult_ptr, key_rns[i]);
		}

		// Step 3: convert to proper form.
		if (is_ckks) {
			ntt_negacyclic_harvey_lazy(temp_poly.get(), small_ntt_tables[i]);
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

		PolyMulConstant polyMulConstant(invP_qi, key_rns[i]);
		polyMulConstant.poly_add(poly + i * degree, temp_poly.get(), poly + i * degree, degree);
	}
}
}
