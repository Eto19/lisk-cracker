import time
import random
import hashlib
import operator
import sys
import binascii
import os
import math
import itertools
import secrets
import csv
import argparse
import nacl.bindings
import json

indexbytes = operator.getitem
intlist2bytes = bytes
int2byte = operator.methodcaller("to_bytes", 1, "big")

start_time = time.time()

b = 256
q = 2 ** 255 - 19
l = 2 ** 252 + 27742317777372353535851937790883648493

def H(m):
    return hashlib.sha512(m).digest()

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

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
    b = (y1+x1)*(y2+x2) % q

    c = t1*2*d*t2 % q
    dd = z1*2*z2 % q

    e = b - a
    f = dd - c
    g = dd + c
    h = b + a

    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)

def edwards_double(P):
    (x1, y1, z1, t1) = P

    a = x1*x1 % q
    b = y1*y1 % q
    c = 2*z1*z1 % q

    xyxy = (x1+y1)*(x1+y1) % q
    
    d = -a % q

    e = (xyxy + d - b) % q
    g = (b - a) % q
    f = g - c
    h = d - b
    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)


def scalarmult(P, e):
    if e == 0:
        return ident
    Q = scalarmult(P, e // 2)
    Q = edwards_double(Q)
    if e & 1:
        Q = edwards_add(Q, P)
    return Q

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
    """
    Not safe to use with secret keys or secret data.
    This function should be used for testing only.
    """
    sk_int = hex_to_int(sk)
    r = (secrets.randbelow(2**200))
    R = scalarmult_B(r)
    S = (r + Hint(encodepoint(R) + pk + m) * sk_int) % l
    return encodepoint(R) + encodeint(S)

def public_key_sha256_dual(pk):
    hashed = hashlib.sha256(pk).digest()
    addr1 = int.from_bytes(hashed[0:8], 'little')
    pk[31] ^= 128;
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

import pycuda.compiler
import pycuda.tools
import pycuda.driver as drv

def save_cubin(kernel, filename):
    file = open(filename,"wb") 
    file.write(kernel) 
    file.close() 

def load_module_new(module_name, module_file, nvcc_options, nvcc_include_dirs, cubin_cache_enable):
    cu_hexhash = hashlib.md5(bytearray(module_file, 'utf-8')).hexdigest()
    cu_hexhash_from_file = ''

    if not (os.path.exists("cubin_cache/"+str(module_name)+".txt")):
        cache_file = open("cubin_cache/"+str(module_name)+".txt", 'w+')
        cache_file.write(cu_hexhash)
        cache_file.close() 
    else:
        cache_file = open("cubin_cache/"+str(module_name)+".txt", 'r')
        cu_hexhash_from_file = cache_file.read()
        cache_file.close() 

    if (cu_hexhash_from_file == cu_hexhash) & (os.path.isfile("cubin/"+str(cu_hexhash_from_file)+"_cubin.cubin")) & cubin_cache_enable:
        print("Load cached %s kernel !" % str(module_name))
        return drv.module_from_file("cubin/"+str(cu_hexhash)+"_cubin.cubin")
    else:
        if (os.path.isfile("cubin/"+str(cu_hexhash_from_file)+"_cubin.cubin")):
            os.remove("cubin/"+str(cu_hexhash_from_file)+"_cubin.cubin")

    cache_file = open("cubin_cache/"+str(module_name)+".txt", 'w')
    cache_file.write(cu_hexhash) 
    cache_file.close() 

    print("Caching %s kernel !" % str(module_name))

    cubin = pycuda.compiler.compile(module_file, options=nvcc_options, include_dirs=nvcc_include_dirs, cache_dir=None)
    save_cubin(cubin, "cubin/"+str(cu_hexhash)+"_cubin.cubin")

    return drv.module_from_file("cubin/"+str(cu_hexhash)+"_cubin.cubin")


def get_define(name, value, file):
    s = "\n"+"#define "+name.upper()+" "+str(value)+"\n"
    return str(s) + file

def print_reg(kernel, kernel_name):
    print("%s Registers : %d" % (kernel_name, kernel.num_regs))
    print("%s Local bytes : %d" % (kernel_name, kernel.local_size_bytes))
    print("%s Shared bytes : %d" % (kernel_name, kernel.shared_size_bytes))
    print()

"""
Load address list from file
"""
def load_list(filename, target_n):
    address_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        address_list = []
        for row in csv_reader:
            address_list.append([int(row[1])])
            if len(address_list)>target_n:
              return np.asarray(address_list, dtype="uint64")

    i = 0
    while len(address_list)<target_n:
        address_list.append([int(i)])
        i += 1

    return np.asarray(address_list, dtype="uint64")


def load_balances(filename, target_n):
    address_list = []
    array_addresses = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            array_addresses.append((row[1],float(row[2])))
            if len(array_addresses)>target_n:
              return array_addresses
    i = 0
    while len(array_addresses)<target_n:
        array_addresses.append((i, 0))
        i += 1

    return array_addresses

list_sk = []

for i in itertools.combinations([0,1,2,3,4,5,6,7], 8):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2])) + (2**(32*i[3])) + (2**(32*i[4])) + (2**(32*i[5])) + (2**(32*i[6])) + (2**(32*i[7]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 7):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2])) + (2**(32*i[3])) + (2**(32*i[4])) + (2**(32*i[5])) + (2**(32*i[6]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 6):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2])) + (2**(32*i[3])) + (2**(32*i[4])) + (2**(32*i[5]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 5):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2])) + (2**(32*i[3])) + (2**(32*i[4]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 4):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2])) + (2**(32*i[3]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 3):
    sk = (2**(32*i[0])) + (2**(32*i[1])) + (2**(32*i[2]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 2):
    sk = (2**(32*i[0])) + (2**(32*i[1]))
    list_sk.append(sk)

for i in itertools.combinations([0,1,2,3,4,5,6,7], 1):
    sk = (2**(32*i[0]))
    list_sk.append(sk)

list_sk.sort()

from nextprime import *

# Initialize np buffers
fold8_ypx = np.zeros((256, 32), dtype="uint8")
fold8_ymx = np.zeros((256, 32), dtype="uint8")
fold8_t  = np.zeros((256, 32), dtype="uint8")

fold8_ypx[0] = int_to_nparray(1)
fold8_ymx[0] = int_to_nparray(1)
fold8_t[0] = int_to_nparray(0)

table_b_0 = [(1, 1, 0)]

for i in range(1, 256):
    P = scalarmult_B(list_sk[i-1])
    (x, y, z, t) = P

    invz = inv(z)
    x = (x * invz) % q
    y = (y * invz) % q
    t = (x * y) % q

    YpX = (y+x)%q
    YmX = (y-x)%q
    T2d = (2*d*t)%q

    # append (YpX, YmX, T2d)
    fold8_ypx[i] = int_to_nparray(YpX)
    fold8_ymx[i] = int_to_nparray(YmX)
    fold8_t[i] = int_to_nparray(T2d)

    table_b_0.append((YpX, YmX, T2d))

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
    b = (y1+x1)*x2 % q

    c = t1*t2 % q
    dd = (z1+z1) % q

    e = b - a
    f = dd - c
    g = dd + c
    h = b + a

    x3 = e*f % q
    y3 = g*h % q
    t3 = e*h % q
    z3 = f*g % q

    return (x3, y3, z3, t3)


def edp_BasePointMult(sk):
    cut = cut_fold8(sk)

    P0 = table_b_0[cut[0]]
    Sx = (P0[0] - P0[1]) % q
    Sy = (P0[0] + P0[1]) % q

    S = (Sx, Sy, 2, 0)

    for i in range(1, 32):
        S = edwards_double(S)
        S = edwards_add2(S, table_b_0[cut[i]])

    (x, y, z, t) = S

    invz = inv(z) % q
    x = (x * invz) % q
    y = (y * invz) % q
    t = (x * y) % q

    return (x, y, 1, t, cut)

def debug_generate(blocks, n_batch, regen_sk, B_mult_table_sk, random_sk_array, n_address_list, address_list_orig):
    sk_index_test = random.randint(0, blocks-1)
    B_mult_table_sk_random = random.randint(0, (n_batch-1))
    regen_sk_random = random.randint(0, regen_sk-1)

    test_inc = (B_mult_table_sk[B_mult_table_sk_random]+B_mult_table_sk[(n_batch-1)]*regen_sk_random)%l
    test_inc2 = (l-B_mult_table_sk[B_mult_table_sk_random]+B_mult_table_sk[(n_batch-1)]*regen_sk_random)%l

    test_sk = random_sk_array[sk_index_test]
    test_sk = (test_sk + test_inc)%l
    test_pk = int_to_pk(test_sk)

    test_sk2 = (random_sk_array[sk_index_test] + test_inc2)%l
    test_pk2 = int_to_pk(test_sk2)

    address_i = random.randint(0, n_address_list-1)

    address_list = np.copy(address_list_orig)

    # Address to test
    address_gen_test_a = [0,1,2,3,4,5,6,7]

    address_gen_test_a[0] = public_key_sha256_dual(int_to_nparray(test_pk[1]%q).copy())
    address_gen_test_a[1] = public_key_sha256_dual(int_to_nparray(test_pk[0]*I%q).copy())
    address_gen_test_a[2] = public_key_sha256_dual(int_to_nparray(((test_pk[1]*-1))%q).copy())
    address_gen_test_a[3] = public_key_sha256_dual(int_to_nparray(((test_pk[0]*I*-1%q))%q).copy())

    address_gen_test_a[4] = public_key_sha256_dual(int_to_nparray(test_pk2[1]%q).copy())
    address_gen_test_a[5] = public_key_sha256_dual(int_to_nparray(test_pk2[0]*I%q).copy())
    address_gen_test_a[6] = public_key_sha256_dual(int_to_nparray(((test_pk2[1]*-1))%q).copy())
    address_gen_test_a[7] = public_key_sha256_dual(int_to_nparray(((test_pk2[0]*I*-1%q))%q).copy())

    print(address_gen_test_a)

    type_b = random.randint(0, 7)
    address_gen_test = address_gen_test_a[type_b]

    type_a = random.randint(0, 1)
    address_list[address_i] = address_gen_test[type_a]
    print("\nSelect test address: %s - sk index %d - sk %s - inc %d - B_mult_table_sk_random %d - address_i %d - type_a %d - type_b %d\n" % (str(address_list[address_i][0]), sk_index_test, int_to_hex(random_sk_array[sk_index_test]), test_inc, B_mult_table_sk_random, address_i, type_a, type_b))
    print("Calculated sk %s | regen_sk_random %s \n" % (int_to_hex(test_sk), regen_sk_random))

    address_list.sort(axis=0)

    return address_list


print("--- Lib Initialization %s seconds ---" % (time.time() - start_time))