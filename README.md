# Lisk cracker

Here is a proof-of-concept cracker for uninitialized lisk addresses.

The software is written in python/cuda and should work on most NVIDIA GPUs (tested on a GTX 1080).

It can be used to recover the funds of your uninitialized lisk address if you lost your passphrase.

It only work on uninitialized lisk addresses.

Multi-GPU isn't supported. To use multiple GPUs you must run multiple instances of the program (1 per GPUs).

## Initialize your lisk address, secure your funds !
* https://www.lisk.support/lisk-support/initialize-your-address/
* https://medium.com/@simonwarta/economics-of-stealing-uninitialized-lisk-accounts-9a6c2529cbd4
* https://research.kudelskisecurity.com/2018/01/16/blockchains-how-to-steal-millions-in-264-operations/
* https://lisk.io/

## Tested performance

Tested performance : ~1.5 billion keys per second on a single GTX 1080 8gb.

Expected results:
- find 1 key every ~8 hours using 50 GTX 1080 GPUs attacking 10000 addresses
- find 1 key every ~80 hours using 50 GTX 1080 GPUs attacking 1000 addresses

Tested on windows 10 + GTX 1080 8gb + 
cuda 10.2 + pycuda + pynacl + python 3.7.0

## Installation
* Install Anaconda3 : https://www.anaconda.com/products/individual

* Install cuda 10.2 : https://developer.nvidia.com/cuda-10.2-download-archive

* Add anaconda3/python3/cuda to PATH

* Install pycuda and pynacl :

  * `pip install pycuda`

  * `pip install pynacl`

* Check cuda installation

  * `nvidia-smi`

  * `nvcc -V`

## Usage

`python main.py --help`

Multi-GPU :

Run for each gpu id :

* `python main.py --gpu-idx GPU_ID`

## Files

List of uninitialized lisk addresses above 1 LSK (id,address,balance):

`list_01.txt`

Results from `main.py` (address,public_key,secret_key,balance):

`results.txt`


## Tool
Generate raw transactions from keypairs in `results.txt` if enough funds for fees:

`python generate_raw_transaction.py --help`

Outputs raw transactions in file :

`transactions.txt`

## Testing

To test the performance you can use :

`python main.py --n-targets 25000000`

It should find a random small address (shorter than 25000000L) in less than 30 min if your keyrate is >1000 Mkeys/s.

Using --n-targets 25000000 will make the program much slower (can take several minutes to initialize the program), and you can get an OOM error if not enough VRAM.

You should use values less than 20000 for normal usage.


## Disclaimer
* The code is very messy but it's the fastest lisk cracker available (2020-08) and it should work fine if installed and used correctly.
* Some code/optimizations may broke in futures lisk updates.
* Do not reuse lisk addresses generated by this program as the signing code may be flawed, only use for testing.
* The program generate raw secret keys instead of passphrases and to generate transactions with them you should use the included script `generate_raw_transactions.py`
* Dont forget to change the recipient address when using `generate_raw_transactions.py` or else it will send the fund to 123456789L



If it was useful please consider donating

Donate XMR : `8BU6iGasRaTagSuCVD5EiyTz1osufL9oTar9VXAuWXkhAByJGfRMfNGjkN1s7JrJWwh7PWV2wbUohDgwvYVypZiE5sMBzGC`

Twitter : @Eto19_
