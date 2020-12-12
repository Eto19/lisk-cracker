import time
import random
import hashlib
import operator
import binascii
import os
import math
import secrets
import csv

indexbytes = operator.getitem
int2byte = operator.methodcaller("to_bytes", 1, "big")

start_time = time.time()

b = 256
q = 2 ** 255 - 19
l = 2 ** 252 + 27742317777372353535851937790883648493

def H(m):
    return hashlib.sha512(m).digest()

def egcd(a, m):
    if a == 0:
        return (m, 0, 1)
    else:
        g, y, x = egcd(m % a, a)
        return (g, x - (m // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    return x % m

def inv(z):
    return modinv(z, q)

d = -121665 * inv(121666) % q
I = pow(2, (q - 1) // 4, q)

def xrecover(y):
    xx = (y * y - 1) * inv(d * y * y + 1)
    x = pow(xx, (q + 3) // 8, q)

    if (x * x - xx) % q != 0:
        x = (x * I) % q

    if x % 2 != 0:
        x = q-x

    return x

By = 4 * inv(5)
Bx = xrecover(By)
B = (Bx % q, By % q, 1, (Bx * By) % q)
ident = (0, 1, 1, 0)

def edwards_add(P, Q):
    (x1, y1, z1, t1) = P
    (x2, y2, z2, t2) = Q

    a = (y1-x1)*(y2-x2) % q
    bb = (y1+x1)*(y2+x2) % q

    c = t1*2*d*t2 % q
    dd = z1*2*z2 % q

    e = bb - a
    f = dd - c
    g = dd + c
    h = bb + a

    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)

def edwards_double(P):
    (x1, y1, z1, t1) = P

    a = x1*x1 % q
    bb = y1*y1 % q
    c = 2*z1*z1 % q

    xyxy = (x1+y1)*(x1+y1) % q
    
    dd = -a % q

    e = (xyxy + dd - bb) % q
    g = (bb - a) % q
    f = g - c
    h = dd - bb
    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)

Bpow = []

def make_Bpow():
    P = B
    for i in range(253):
        Bpow.append(P)
        P = edwards_double(P)

make_Bpow()

def scalarmult_B(e):
    e = e % l
    P = ident
    for i in range(253):
        if e & 1:
            P = edwards_add(P, Bpow[i])
        e = e // 2
    assert e == 0, e
    return P

def bit(h, i):
    return (indexbytes(h, i // 8) >> (i % 8)) & 1

def isoncurve(P):
    (x, y, z, t) = P
    return (z % q != 0 and
            x*y % q == z*t % q and
            (y*y - x*x - z*z - d*t*t) % q == 0)

def encodepoint(P):
    (x, y, z, t) = P

    zi = inv(z)

    x = (x * zi) % q
    y = (y * zi) % q

    bits = [(y >> i) & 1 for i in range(b - 1)] + [x & 1]

    return b''.join([
        int2byte(sum([bits[i * 8 + j] << j for j in range(8)]))
        for i in range(b // 8)
    ])

def decodepoint(s):
    y = sum(2 ** i * bit(s, i) for i in range(0, b - 1))
    x = xrecover(y)
    if x & 1 != bit(s, b-1):
        x = q - x
    P = (x, y, 1, (x*y) % q)
    
    if not isoncurve(P):
        return "decoding point that is not on curve"
    
    return P

import numpy as np

def int_to_nparray(x, size=32):
    return np.frombuffer(int(x).to_bytes(size, byteorder='little'), dtype='uint8')

def int_to_nparray32(x, size=32):
    return np.frombuffer(int(x).to_bytes(size, byteorder='little'), dtype='uint32')

def int_to_nparray32_2(x, size=32):
    return np.frombuffer(int(x).to_bytes(size, byteorder='big'), dtype='uint32')

def nparray_to_int(x):
    return int.from_bytes(x.tobytes(), byteorder='little')

def nparray_to_int2(x):
    return int.from_bytes(x.tobytes(), byteorder='big')

def hex_to_nparray(x):
    return np.frombuffer(binascii.unhexlify(x), dtype='uint8')

def nparray_to_hex(x):
    return binascii.hexlify(x.tobytes())

def int_to_hex(x):
    return nparray_to_hex(int_to_nparray(x))

def int_to_nparray2(x, size=32):
    return np.frombuffer(int(x).to_bytes(size, byteorder='big'), dtype='uint8')

def int_to_hex2(x):
    return nparray_to_hex(int_to_nparray2(x))

def hex_to_int(x):
    return nparray_to_int(hex_to_nparray(x))

def decodeint(s):
    return sum(2 ** i * bit(s, i) for i in range(0, 256))

def encodeint(y):
    bits = [(y >> i) & 1 for i in range(256)]
    return b''.join([
         int2byte(sum([bits[i * 8 + j] << j for j in range(8)]))
         for i in range(256//8)
    ])

def Hint(m):
    h = H(m)
    return sum(2 ** i * bit(h, i) for i in range(2 * 256))

def signature_unsafe(m, sk, pk):
    sk_int = hex_to_int(sk)
    r = (secrets.randbelow(2**200))
    R = scalarmult_B(r)
    S = (r + Hint(encodepoint(R) + pk + m) * sk_int) % l
    return encodepoint(R) + encodeint(S)

def public_key_sha256_dual(pk):
    hashed = hashlib.sha256(pk).digest()
    addr1 = int.from_bytes(hashed[0:8], 'little')
    pk[31] ^= 128
    hashed = hashlib.sha256(pk).digest()
    addr2 = int.from_bytes(hashed[0:8], 'little')
    
    return (addr1, addr2)

def int_to_pk(a):
  A = edp_BasePointMult(a)
  (x, y, z, t, cut) = A
  return (x%q, y%q)

def H256(m):
    return hashlib.sha256(m).digest()

def get_transaction_bytes(transaction):
    tx_type = int(transaction['type']).to_bytes(1, 'little')
    tx_timestamp = int(transaction['timestamp']).to_bytes(4, 'little')
    tx_amount = int(transaction['amount']).to_bytes(8, 'little')
    tx_publickey = binascii.unhexlify(transaction['senderPublicKey'])
    tx_recipient = int(transaction['recipientId'][:-1]).to_bytes(8, 'big')

    if transaction["signature"] != "":
        tx_signature = binascii.unhexlify(transaction['signature'])
        return (tx_type + tx_timestamp + tx_publickey + tx_recipient + tx_amount + tx_signature)

    return (tx_type + tx_timestamp + tx_publickey + tx_recipient + tx_amount)

def get_transaction_hash(transaction):
    tx_bytes = get_transaction_bytes(transaction)
    return H256(tx_bytes)

def get_transaction_id(transaction):
    tx_bytes = get_transaction_bytes(transaction)
    h = H256(tx_bytes)
    addr = bytearray(h[0:8])
    addr.reverse()

    return str(int.from_bytes(addr, byteorder='big', signed=False))

def liskpktoaddr(pk):
    pk = binascii.unhexlify(pk)
    h = H256(pk)

    addr = bytearray(h[0:8])
    addr.reverse()

    return str(int.from_bytes(addr, byteorder='big', signed=False))

def privatetopublickey(pk):
    pk1_edp = edp_BasePointMult(pk)
    (x, y, z, t, cut) = pk1_edp
    pk1 = (x, y, z, t)

    pk2_edp = edp_BasePointMult((l-pk)%l)
    (x, y, z, t, cut) = pk2_edp
    pk2 = (x, y, z, t)

    A0  = pk1
    encoded0 = encodepoint(A0)

    A1  = pk2
    encoded1 = encodepoint(A1)

    loworder = decodepoint(binascii.unhexlify("ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"))

    A2  = edwards_add(pk1, loworder)
    encoded2 = encodepoint(A2)

    A3  = edwards_add(pk2, loworder)
    encoded3 = encodepoint(A3)

    loworder2 = decodepoint(binascii.unhexlify("0000000000000000000000000000000000000000000000000000000000000000"))

    A4  = edwards_add(pk1, loworder2)
    encoded4 = encodepoint(A4)

    A5  = edwards_add(pk2, loworder2)
    encoded5 = encodepoint(A5)

    A6  = edwards_add(edwards_add(pk1, loworder2),loworder)
    encoded6 = encodepoint(A6)

    A7  = edwards_add(edwards_add(pk2, loworder2),loworder)
    encoded7 = encodepoint(A7)

    return [binascii.hexlify(encoded0), binascii.hexlify(encoded1), binascii.hexlify(encoded2), binascii.hexlify(encoded3), binascii.hexlify(encoded4), binascii.hexlify(encoded5), binascii.hexlify(encoded6), binascii.hexlify(encoded7)]

def save_cubin(kernel, filename):
    file = open(filename,"wb") 
    file.write(kernel) 
    file.close() 

def get_define(name, value, file):
    s = "\n"+"#define "+name.upper()+" "+str(value)+"\n"
    return str(s) + file

"""
Load address list from file
"""
def load_list(filename, target_n):

    with open(filename) as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        address_list = []
        for row1 in csv_reader1:
            address_list.append([int(row1[1])])
            if len(address_list)>target_n:
              return np.asarray(address_list, dtype="uint64")

    i = 0
    while len(address_list)<target_n:
        address_list.append([int(i)])
        i += 1

    return np.asarray(address_list, dtype="uint64")


def load_balances(filename, target_n):
    array_addresses = []

    with open(filename) as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        for row1 in csv_reader1:
            array_addresses.append((row1[1],float(row1[2])))
            if len(array_addresses)>target_n:
              return array_addresses
    i = 0
    while len(array_addresses)<target_n:
        array_addresses.append((i, 0))
        i += 1

    return array_addresses

# Initialize np buffers
fold8_ypx = np.zeros((256, 32), dtype="uint8")
fold8_ymx = np.zeros((256, 32), dtype="uint8")
fold8_t  = np.zeros((256, 32), dtype="uint8")

fold8_table = []
with open('data/fold8.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for fold8_i, row in enumerate(csv_reader):
        YpX, YmX, T2d = row

        YpX = int(YpX)
        YmX = int(YmX)
        T2d = int(T2d)

        fold8_table.append((YpX, YmX, T2d))

        fold8_ypx[fold8_i] = int_to_nparray(YpX)
        fold8_ymx[fold8_i] = int_to_nparray(YmX)
        fold8_t[fold8_i] = int_to_nparray(T2d)


def cut_fold8(sk):
    Y = []
    X = int_to_nparray32(sk)
    a = 0
    for i in reversed(range(32)):
        for j in reversed(range(8)):
            a = ((a << 1) + ((X[j] >> i) & 1))
        Y.append(a%256)

    return Y

def edwards_add2(P, Q):
    (x1, y1, z1, t1) = P
    (x2, y2, t2) = Q

    a = (y1-x1)*y2 % q
    bb = (y1+x1)*x2 % q

    c = t1*t2 % q
    dd = (z1+z1) % q

    e = bb - a
    f = dd - c
    g = dd + c
    h = bb + a

    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)

def edp_BasePointMult(sk):
    cut = cut_fold8(sk)

    P0 = fold8_table[cut[0]]
    Sx = (P0[0] - P0[1]) % q
    Sy = (P0[0] + P0[1]) % q

    S = (Sx, Sy, 2, 0)

    for i in range(1, 32):
        S = edwards_double(S)
        S = edwards_add2(S, fold8_table[cut[i]])

    (x, y, z, t) = S

    invz = inv(z) % q
    x = (x * invz) % q
    y = (y * invz) % q
    t = (x * y) % q

    return (x, y, 1, t, cut)

def debug_generate(device, blocks, n_batch, regen_sk, scalar_y_table, random_sk_list, n_address_list, address_list_orig):
    scalar_x_index = random.randint(0, blocks-1)
    scalar_y_index = random.randint(0, n_batch-1)
    random_iteration = random.randint(0, regen_sk-1)

    test_inc1 = (scalar_y_table[scalar_y_index]+scalar_y_table[(n_batch-1)]*random_iteration)%l
    test_inc2 = (l-scalar_y_table[scalar_y_index]+scalar_y_table[(n_batch-1)]*random_iteration)%l

    test_sk1 = random_sk_list[scalar_x_index]
    test_sk1 = (test_sk1 + test_inc1)%l
    test_pk = int_to_pk(test_sk1)

    test_sk2 = (random_sk_list[scalar_x_index] + test_inc2)%l
    test_pk2 = int_to_pk(test_sk2)

    address_list = np.copy(address_list_orig)

    # Generate 16 addresses
    address_gen_test_a = [0,1,2,3,4,5,6,7]

    address_gen_test_a[0] = public_key_sha256_dual(int_to_nparray(test_pk[1]%q).copy())
    address_gen_test_a[1] = public_key_sha256_dual(int_to_nparray(test_pk[0]*I%q).copy())
    address_gen_test_a[2] = public_key_sha256_dual(int_to_nparray(((test_pk[1]*-1))%q).copy())
    address_gen_test_a[3] = public_key_sha256_dual(int_to_nparray(((test_pk[0]*I*-1%q))%q).copy())

    address_gen_test_a[4] = public_key_sha256_dual(int_to_nparray(test_pk2[1]%q).copy())
    address_gen_test_a[5] = public_key_sha256_dual(int_to_nparray(test_pk2[0]*I%q).copy())
    address_gen_test_a[6] = public_key_sha256_dual(int_to_nparray(((test_pk2[1]*-1))%q).copy())
    address_gen_test_a[7] = public_key_sha256_dual(int_to_nparray(((test_pk2[0]*I*-1%q))%q).copy())

    type_b = random.randint(0, 7)
    address_gen_test = address_gen_test_a[type_b]

    type_a = random.randint(0, 1)

    address_i = random.randint(0, n_address_list - 1)
    address_list[address_i] = address_gen_test[type_a]

    out_str = f"""
--------------------------
- DEBUG/TEST PARAMETERS
- Device : {device}
- Scalar x index: {scalar_x_index}
- Scalar y index: {scalar_y_index}
- Scalar x : {random_sk_list[scalar_x_index]}
- Scalar y 1 : {test_inc1}
- Scalar y 2 : {test_inc2}
- Random iteration : {random_iteration}
- Hex secret key 1 (x+(y*iter)) : {int_to_hex(test_sk1).decode('utf-8')}
- Hex secret key 2 (x+(l-y*iter)) : {int_to_hex(test_sk2).decode('utf-8')}
- List of generated addresses : {address_gen_test_a}
- Address index : addresses[{type_a}][{type_b}]
- Target Test address : {address_list[address_i][0]}L
--------------------------
                    """
    print(out_str)

    address_list.sort(axis=0)

    return address_list

def test_found(device, found_flag, inc_count, n_batch, B_mult_table_sk, blocks, random_sk_list, balances, output_filename, debug_test, target_list):
    debug_test_found = False
    print()
    print("Checking found...")

    found_flag_i = found_flag[0] >> 1
    increment_ff = math.floor(((found_flag_i) % (n_batch)))
    increment_1 = B_mult_table_sk[increment_ff] + inc_count * B_mult_table_sk[(n_batch - 1)]
    increment_2 = l - B_mult_table_sk[increment_ff] + inc_count * B_mult_table_sk[(n_batch - 1)]

    c_id1 = (found_flag_i) - (blocks * n_batch)
    c_id2 = (found_flag_i) - (blocks * n_batch) * 2
    c_id3 = (found_flag_i) - (blocks * n_batch) * 3

    c_id_array = [(found_flag_i - increment_ff) // n_batch, (c_id1 - increment_ff) // n_batch,
                  (c_id2 - increment_ff) // n_batch, (c_id3 - increment_ff) // n_batch]
    for sk_index in c_id_array:
        if sk_index < len(random_sk_list):
            sk_index = math.floor(sk_index)

            found_sk_orig = random_sk_list[sk_index]

            found_sk_1 = found_sk_orig + increment_1
            found_sk_2 = found_sk_orig + increment_2

            found_sk_hex_calc_1 = int_to_hex(found_sk_1)
            found_sk_hex_calc_2 = int_to_hex(found_sk_2)

            found_pks_1 = privatetopublickey(found_sk_1)
            found_pks_2 = privatetopublickey(found_sk_2)

            found_pk = 0
            select_address = 0
            found = 0

            found_sk_hex_calc = ''
            increment = 0

            for pk in found_pks_1:
                if np.isin(int(liskpktoaddr(pk)), target_list):
                    found_pk = pk
                    select_address = str(int(liskpktoaddr(pk)))
                    found_sk_hex_calc = found_sk_hex_calc_1
                    increment = increment_1
                    found = 1

            for pk in found_pks_2:
                if np.isin(int(liskpktoaddr(pk)), target_list):
                    found_pk = pk
                    select_address = str(int(liskpktoaddr(pk)))
                    found_sk_hex_calc = found_sk_hex_calc_2
                    increment = increment_2
                    found = 1

            if found:
                balance = 0
                for address_balance in balances:
                    if str(select_address) == str(address_balance[0]):
                        balance = address_balance[1]

                out_str = f"""
--------------------------
- Device : {device}
- Target found at gid {found_flag[0]} | Iteration {inc_count} | Scalar x index {sk_index}
- Scalar x : {found_sk_orig}
- Scalar y : {increment}
- Hex secret key (x+y) : {found_sk_hex_calc.decode('utf-8')}
- Public Key : {found_pk.decode('utf-8')}
- Balance : {balance} LSK
- Target : {select_address}L
--------------------------
- address,public_key,secret_key,balance
- {select_address}L,{found_pk.decode('utf-8')},{found_sk_hex_calc.decode('utf-8')},{balance}
--------------------------
                """
                print(out_str)

                if not debug_test:
                    res_file = open(str(output_filename), "a+")
                    res_file.write(f"{select_address}L,{found_pk.decode('utf-8')},{found_sk_hex_calc.decode('utf-8')},{balance}\n")
                    res_file.close()

                if debug_test:
                    debug_test_found = True

                break

    return debug_test_found
