// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"

using namespace std;
using namespace seal;

void example_rotation_bfv()
{
    print_example_banner("Rotation BFV");

    EncryptionParameters parms(scheme_type::BFV);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(8192));
    parms.set_plain_modulus(65537);
    auto context = SEALContext::Create(parms);
    print_parameters(context);

    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    BatchEncoder batch_encoder(context);

    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    pod_matrix[0] = 0ULL;
    pod_matrix[1] = 1ULL;
    pod_matrix[2] = 2ULL;
    pod_matrix[3] = 3ULL;
    pod_matrix[row_size] = 4ULL;
    pod_matrix[row_size + 1] = 5ULL;
    pod_matrix[row_size + 2] = 6ULL;
    pod_matrix[row_size + 3] = 7ULL;

    cout << endl;
    cout << "Input plaintext matrix:" << endl;
    print_matrix(pod_matrix, row_size);

    /*
    First we use BatchEncoder to compose the matrix into a plaintext.
    */
    Plaintext plain_matrix;
    cout << "-- Encoding plaintext matrix: ";
    batch_encoder.encode(pod_matrix, plain_matrix);
    cout << "Done" <<endl;

    /*
    Next we encrypt the plaintext as usual.
    */
    Ciphertext encrypted_matrix;
    cout << "-- Encrypting: ";
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "Done" << endl;
    cout << "\tNoise budget in fresh encryption: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    Rotation requires galois keys.
    */
    GaloisKeys gal_keys = keygen.galois_keys();

    /*
    Now rotate the rows to the left 3 steps, decrypt, decompose, and print.
    */
    evaluator.rotate_rows_inplace(encrypted_matrix, 3, gal_keys);
    cout << "-- Rotated rows 3 steps left: " << endl;
    Plaintext plain_result;
    decryptor.decrypt(encrypted_matrix, plain_result);
    batch_encoder.decode(plain_result, pod_matrix);
    print_matrix(pod_matrix, row_size);
    cout << "\tNoise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
    cout << endl;

    /*
    Rotate columns (swap rows), decrypt, decompose, and print.
    */
    evaluator.rotate_columns_inplace(encrypted_matrix, gal_keys);
    cout << "-- Rotated columns: " << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    batch_encoder.decode(plain_result, pod_matrix);
    print_matrix(pod_matrix, row_size);
    cout << "\tNoise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
    cout << endl;

    /*
    Rotate rows to the right 4 steps, decrypt, decompose, and print.
    */
    evaluator.rotate_rows_inplace(encrypted_matrix, -4, gal_keys);
    cout << "-- Rotated rows 4 steps right: " << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    batch_encoder.decode(plain_result, pod_matrix);
    print_matrix(pod_matrix, row_size);
    cout << "\tNoise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    We can see that rotation does not consume noise budget.
    */
}

void example_rotation_ckks()
{
    print_example_banner("Rotation CKKS");

    /*
    We show how to apply vector rotations on the encrypted data. This
    is very similar to how matrix rotations work in the BFV scheme. We try this
    with three sizes of Galois keys. In some cases it is desirable for memory
    reasons to create Galois keys that support only specific rotations. This can
    be done by passing to KeyGenerator::galois_keys(...) a vector of signed
    integers specifying the desired rotation step counts. Here we create Galois
    keys that only allow cyclic rotation by a single step (at a time) to the left.
    */
    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(8192));
    auto context = SEALContext::Create(parms);
    print_parameters(context);

    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys = keygen.relin_keys();
    GaloisKeys gal_keys = keygen.galois_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder ckks_encoder(context);

    size_t slot_count = ckks_encoder.slot_count();
    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0, step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++, curr_point += step_size)
    {
        input.push_back(curr_point);
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    auto scale = pow(2.0, 50);
    Plaintext plain;
    ckks_encoder.encode(input, scale, plain);
    Ciphertext encrypted;
    encryptor.encrypt(plain, encrypted);

    Ciphertext rotated;
    evaluator.rotate_vector(encrypted, 2, gal_keys, rotated);
    decryptor.decrypt(rotated, plain);
    vector<double> result;
    ckks_encoder.decode(plain, result);
    cout << "Rotated:" << endl;
    print_vector(result, 3, 7);
}

void example_rotation()
{
    print_example_banner("Example: Rotation");

    example_rotation_bfv();

    example_rotation_ckks();
}