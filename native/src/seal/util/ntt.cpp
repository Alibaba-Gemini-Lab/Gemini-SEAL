// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/modulus.h"
#include "seal/util/defines.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarith.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include <algorithm>

using namespace std;

namespace seal
{
    namespace util
    {
        // (x * 2^64) / p
        static uint64_t shoupify(uint64_t x, uint64_t p) {
            uint64_t cnst_128[2]{0, x};
            uint64_t shoup[2];
            seal::util::divide_uint128_inplace(cnst_128, p, shoup);
            return shoup[0];
        }

        NTTTables::NTTTables(int coeff_count_power, const Modulus &modulus, MemoryPoolHandle pool) : pool_(move(pool))
        {
#ifdef SEAL_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(coeff_count_power, modulus);
        }

        void NTTTables::initialize(int coeff_count_power, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if ((coeff_count_power < get_power_of_two(SEAL_POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > get_power_of_two(SEAL_POLY_MOD_DEGREE_MAX))
            {
                throw invalid_argument("coeff_count_power out of range");
            }
#endif
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;

            // Allocate memory for the tables
            root_powers_ = allocate_uint(coeff_count_, pool_);
            inv_root_powers_ = allocate_uint(coeff_count_, pool_);
            scaled_root_powers_ = allocate_uint(coeff_count_, pool_);
            scaled_inv_root_powers_ = allocate_uint(coeff_count_, pool_);
            modulus_ = modulus;

            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                throw invalid_argument("invalid modulus");
            }

            uint64_t inverse_root;
            if (!try_invert_uint_mod(root_, modulus_, inverse_root))
            {
                throw invalid_argument("invalid modulus");
            }

            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            if (!try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_))
            {
                throw invalid_argument("invalid modulus");
            }
            scaled_inv_degree_ = shoupify(inv_degree_modulo_, modulus_.value());

            reduce_precomp_ = shoupify(1, modulus_.value());

            // Populate the tables storing (scaled version of) powers of root
            // mod q in bit-scrambled order.
            ntt_powers_of_primitive_root(root_, root_powers_.get());
            ntt_scale_powers_of_primitive_root(root_powers_.get(), scaled_root_powers_.get());

            // Populate the tables storing (scaled version of) powers of
            // (root)^{-1} mod q in bit-scrambled order.
            ntt_powers_of_primitive_root(inverse_root, inv_root_powers_.get());
            // Reordering inv_root_powers_ so that the access pattern in inverse NTT is sequential.
            auto temp = allocate_uint(coeff_count_, pool_);
            uint64_t *temp_ptr = temp.get() + 1;
            for (size_t m = (coeff_count_ >> 1); m > 0; m >>= 1)
            {
                for (size_t i = 0; i < m; i++)
                {
                    *temp_ptr++ = inv_root_powers_[m + i];
                }
            }
            set_uint(temp.get() + 1, coeff_count_ - 1, inv_root_powers_.get() + 1);
            // merge the last inv_root_powers with n^{-1}
            inv_root_powers_[coeff_count_ - 1] = multiply_uint_mod(inv_root_powers_[coeff_count_ - 1], inv_degree_modulo_, modulus_);
            ntt_scale_powers_of_primitive_root(inv_root_powers_.get(), scaled_inv_root_powers_.get());
        }

        void NTTTables::ntt_powers_of_primitive_root(uint64_t root, uint64_t *destination) const
        {
            uint64_t *destination_start = destination;
            *destination_start = 1;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                uint64_t *next_destination = destination_start + reverse_bits(i, coeff_count_power_);
                *next_destination = multiply_uint_mod(*destination, root, modulus_);
                destination = next_destination;
            }
        }

        // Compute floor (input * beta /q), where beta is a 64k power of 2 and  0 < q < beta.
        void NTTTables::ntt_scale_powers_of_primitive_root(const uint64_t *input, uint64_t *destination) const
        {
            uint64_t p = modulus_.value();
            auto shoupifier = [&p](uint64_t x) { return shoupify(x, p); };
            std::transform(input, input + coeff_count_, destination, shoupifier);
        }

        class NTTTablesCreateIter
        {
        public:
            using value_type = NTTTables;
            using pointer = void;
            using reference = value_type;
            using difference_type = std::ptrdiff_t;

            // LegacyInputIterator allows reference to be equal to value_type so we can construct
            // the return objects on the fly and return by value.
            using iterator_category = std::input_iterator_tag;

            // Require default constructor
            NTTTablesCreateIter()
            {}

            // Other constructors
            NTTTablesCreateIter(int coeff_count_power, vector<Modulus> modulus, MemoryPoolHandle pool)
                : coeff_count_power_(coeff_count_power), modulus_(modulus), pool_(pool)
            {}

            // Require copy and move constructors and assignments
            NTTTablesCreateIter(const NTTTablesCreateIter &copy) = default;

            NTTTablesCreateIter(NTTTablesCreateIter &&source) = default;

            NTTTablesCreateIter &operator=(const NTTTablesCreateIter &assign) = default;

            NTTTablesCreateIter &operator=(NTTTablesCreateIter &&assign) = default;

            // Dereferencing creates NTTTables and returns by value
            inline value_type operator*() const
            {
                return { coeff_count_power_, modulus_[index_], pool_ };
            }

            // Pre-increment
            inline NTTTablesCreateIter &operator++() noexcept
            {
                index_++;
                return *this;
            }

            // Post-increment
            inline NTTTablesCreateIter operator++(int) noexcept
            {
                NTTTablesCreateIter result(*this);
                index_++;
                return result;
            }

            // Must be EqualityComparable
            inline bool operator==(const NTTTablesCreateIter &compare) const noexcept
            {
                return (compare.index_ == index_) && (coeff_count_power_ == compare.coeff_count_power_);
            }

            inline bool operator!=(const NTTTablesCreateIter &compare) const noexcept
            {
                return !operator==(compare);
            }

            // Arrow operator must be defined
            value_type operator->() const
            {
                return **this;
            }

        private:
            size_t index_ = 0;
            int coeff_count_power_ = 0;
            vector<Modulus> modulus_;
            MemoryPoolHandle pool_;
        };

        void CreateNTTTables(
            int coeff_count_power, const vector<Modulus> &modulus, Pointer<NTTTables> &tables, MemoryPoolHandle pool)
        {
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
            if (!modulus.size())
            {
                throw invalid_argument("invalid modulus");
            }
            // coeff_count_power and modulus will be validated by "allocate"

            NTTTablesCreateIter iter(coeff_count_power, modulus, pool);
            tables = allocate(iter, modulus.size(), pool);
        }

         struct SlothfulNTT {
             uint64_t p, Lp; // for now, Lp = 2*p
             uint64_t rdp; // floor(2^64 / p)
             explicit SlothfulNTT(uint64_t p, uint64_t Lp, uint64_t rdp) : p(p), Lp(Lp), rdp(rdp) {
#ifdef SEAL_DEBUG
                 if (p >= (1UL << SEAL_USER_MOD_BIT_COUNT_MAX)) {
                     throw std::logic_error("SlothfulNTT: |p| out-of-bound");
                 }
#endif
             }

             // return 0 if cond = true, else return b if cond = false
             inline uint64_t select(uint64_t b, bool cond) const {
                 return (b & -(uint64_t) cond) ^ b;
             }

             // x * y mod p using Shoup's trick, i.e., yshoup = floor(2^64 * y / p)
             inline uint64_t mulmodLazy(uint64_t x, uint64_t y, uint64_t yshoup) const {
                 unsigned long long q;
                 multiply_uint64_hw64(x, yshoup, &q);
                 return x * y - q * p;
             }

             // Basically mulmodLazy(x, 1, shoup(1))
             inline uint64_t reduceBarrettLazy(uint64_t x) const {
                 unsigned long long q;
                 multiply_uint64_hw64(x, rdp, &q);
                 return x - q * p;
             }

             // x0' <- x0 + w * x1 mod p
             // x1' <- x0 - w * x1 mod p
             inline void ForwardLazy(uint64_t *x0, uint64_t *x1, uint64_t w, uint64_t wshoup) const {
                 uint64_t u, v;
                 u = *x0;
                 v = mulmodLazy(*x1, w, wshoup);

                 *x0 = u + v;
                 *x1 = u - v + Lp;
             }

             inline void ForwardLazyLast(uint64_t *x0, uint64_t *x1, uint64_t w, uint64_t wshoup) const {
                 uint64_t u, v;
                 u = reduceBarrettLazy(*x0);
                 v = mulmodLazy(*x1, w, wshoup);

                 *x0 = u + v;
                 *x1 = u - v + Lp;
             }

             // x0' <- x0 + x1 mod p
             // x1' <- x0 - w * x1 mod p
             inline void BackwardLazy(uint64_t *x0, uint64_t *x1, uint64_t w, uint64_t wshoup) const {
                 uint64_t u = *x0;
                 uint64_t v = *x1;
                 uint64_t t = u + v;
                 t -= select(Lp, t < Lp);
                 *x0 = t;
                 *x1 = mulmodLazy(u - v + Lp, w, wshoup);
             }

             inline void BackwardLazyLast(uint64_t *x0, uint64_t *x1, uint64_t inv_n, uint64_t inv_n_s, uint64_t inv_n_w, uint64_t inv_n_w_s) const {
                 uint64_t u = *x0;
                 uint64_t v = *x1;
                 uint64_t t = u + v;
                 t -= select(Lp, t < Lp);
                 *x0 = mulmodLazy(t, inv_n, inv_n_s);
                 *x1 = mulmodLazy(u - v + Lp, inv_n_w, inv_n_w_s);
             }
        };

        /**
        This function computes in-place the negacyclic NTT. The input is
        a polynomial a of degree n in R_q, where n is assumed to be a power of
        2 and q is a prime such that q = 1 (mod 2n).

        The output is a vector A such that the following hold:
        A[j] =  a(psi**(2*bit_reverse(j) + 1)), 0 <= j < n.
        */
        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
            const uint64_t p = tables.modulus().value();
            const size_t n = size_t(1) << tables.coeff_count_power();
            SlothfulNTT sntt(p, p << 1, tables.get_reduce_precomp());

            const uint64_t *w = tables.root_powers() + 1;
            const uint64_t *wshoup = tables.scaled_root_powers() + 1;

            // main loop: for h >= 4
            size_t m = 1;
            size_t h = n >> 1;
            for (; h > 2; m <<= 1, h >>= 1) {
                // invariant: h * m = degree / 2
                // different buttefly groups
                uint64_t *x0 = operand;
                uint64_t *x1 = x0 + h; // invariant: x1 = x0 + h during the iteration
                for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
                    for (size_t i = 0; i < h; i += 4) { // unrolling
                        sntt.ForwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.ForwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.ForwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.ForwardLazy(x0++, x1++, *w, *wshoup);
                    }
                    x0 += h;
                    x1 += h;
                }
            }

            // m = degree / 4, h = 2
            m = n >> 2;
            uint64_t *x0 = operand;
            uint64_t *x1 = x0 + 2;
            for (size_t r = 0; r < m; ++r, ++w, ++wshoup) { // unrolling
                sntt.ForwardLazy(x0++, x1++, *w, *wshoup);
                sntt.ForwardLazy(x0, x1, *w, *wshoup); // combine the incr to following steps
                x0 += 3;
                x1 += 3;
            }

            // m = degree / 2, h = 1
            m = n >> 1;
            x0 = operand;
            x1 = x0 + 1;
            for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
                sntt.ForwardLazyLast(x0, x1, *w, *wshoup);
                x0 += 2;
                x1 += 2;
            }
            // At the end operand[0 .. n) stay in [0, 4p).
        }

        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("operand");
            }
#endif
            const uint64_t p = tables.modulus().value();
            const size_t n = 1L << tables.coeff_count_power();
            const uint64_t *w = tables.inv_root_powers() + 1;
            const uint64_t *wshoup = tables.scaled_inv_root_powers() + 1;
            SlothfulNTT sntt(p, 2 * p, /*dummy*/0);
            // first loop: m = degree / 2, h = 1
            // m > 1 to skip the last layer
            size_t m = n >> 1;
            auto x0 = operand;
            auto x1 = x0 + 1; // invariant: x1 = x0 + h during the iteration
            for (size_t r = 0; m > 1 && r < m; ++r, ++w, ++wshoup) {
                sntt.BackwardLazy(x0, x1, *w, *wshoup);
                x0 += 2;
                x1 += 2;
            }

            // second loop: m = degree / 4, h = 2
            // m > 1 to skip the last layer
            m = n >> 2;
            x0 = operand;
            x1 = x0 + 2;
            for (size_t r = 0; m > 1 && r < m; ++r, ++w, ++wshoup) {
                sntt.BackwardLazy(x0++, x1++, *w, *wshoup);
                sntt.BackwardLazy(x0, x1, *w, *wshoup);
                x0 += 3;
                x1 += 3;
            }
            // main loop: for h >= 4
            m = n >> 3;
            size_t h = 4;
            // m > 1 to skip the last layer
            for (; m > 1; m >>= 1, h <<= 1) {
                x0 = operand;
                x1 = x0 + h;
                for (size_t r = 0; r < m; ++r, ++w, ++wshoup) {
                    for (size_t i = 0; i < h; i += 4) { // unrolling
                        sntt.BackwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.BackwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.BackwardLazy(x0++, x1++, *w, *wshoup);
                        sntt.BackwardLazy(x0++, x1++, *w, *wshoup);
                    }
                    x0 += h;
                    x1 += h;
                }
            }

            // At the end operand[0 .. n) lies in [0, 2p)
            x0 = operand;
            x1 = x0 + (n >> 1);
            uint64_t inv_n = *(tables.get_inv_degree_modulo());
            uint64_t inv_n_s = tables.get_scaled_inv_degree_modulo();
            for (size_t r = n >> 1; r < n; ++r) {
                sntt.BackwardLazyLast(x0++, x1++, inv_n, inv_n_s, *w, *wshoup);
            }
        }
    } // namespace util
} // namespace seal
