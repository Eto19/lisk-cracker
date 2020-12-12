from lib_ed import *
from nextprime import *

import threading
import sys
import argparse
import pycuda.compiler
import pycuda.tools
import pycuda.driver as drv

def load_module_new(module_name, module_file, nvcc_options, nvcc_include_dirs, cubin_cache_enable, kernel_str='default'):
    cu_hexhash = hashlib.md5(bytearray(module_file, 'utf-8')).hexdigest()
    kernel_hexhash = hashlib.md5(bytearray(kernel_str, 'utf-8')).hexdigest()
    cu_hexhash_from_file = ''

    if not (os.path.exists(f"cubin_cache/{module_name}_{kernel_hexhash}.txt")):
        cache_file = open(f"cubin_cache/{module_name}_{kernel_hexhash}.txt", 'w+')
        cache_file.write(cu_hexhash)
        cache_file.close()
    else:
        cache_file = open(f"cubin_cache/{module_name}_{kernel_hexhash}.txt", 'r')
        cu_hexhash_from_file = cache_file.read()
        cache_file.close()

    if (cu_hexhash_from_file == cu_hexhash) & (os.path.isfile(f"cubin/{cu_hexhash_from_file}_{kernel_hexhash}_cubin.cubin")) & cubin_cache_enable:
        print(f"Load cached {module_name} kernel !")
        return drv.module_from_file(f"cubin/{cu_hexhash}_{kernel_hexhash}_cubin.cubin")
    else:
        if os.path.isfile(f"cubin/{cu_hexhash_from_file}_{kernel_hexhash}_cubin.cubin"):
            os.remove(f"cubin/{cu_hexhash_from_file}_{kernel_hexhash}_cubin.cubin")

    cache_file = open(f"cubin_cache/{module_name}_{kernel_hexhash}.txt", 'w')
    cache_file.write(cu_hexhash)
    cache_file.close()

    print(f"Caching {module_name} kernel !")

    cubin = pycuda.compiler.compile(module_file, options=nvcc_options, include_dirs=nvcc_include_dirs, cache_dir=None)
    save_cubin(cubin, f"cubin/{cu_hexhash}_{kernel_hexhash}_cubin.cubin")

    return drv.module_from_file(f"cubin/{cu_hexhash}_{kernel_hexhash}_cubin.cubin")

class GPUThread(threading.Thread):
    def __init__(self, gpu_id, batch_size, tid):
        threading.Thread.__init__(self)

        self._stop_event = threading.Event()

        self.gpu_id = gpu_id
        self.tid = tid
        self.n_batch = batch_size

        self.dev = drv.Device(gpu_idx)
        self.blocks = self.dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT) * 8 * 64

        self.grid = (self.blocks, 1)
        self.grid_2 = (self.blocks*2, 1)
        self.grid256 = (self.blocks//self.n_batch, 1)
        self.gridsha256 = (self.blocks*8, 1)
        self.block = (self.n_batch, 1, 1)

        self.grid_scalarmult = (self.blocks//256, 1)
        self.block_scalarmult = (256, 1, 1)

        self.global_size = self.n_batch * self.blocks
        self.target_count = target_list.size

        # Calculate bloom filter size
        self.bloom_filter_k_hash = bloom_filter_k_hash
        self.bloom_filter_p = bloom_filter_p

        self.bloom_filter_size = bloom_filter_size

        batch_ge_add_file = open("cu/batch_ge_add.cu", "r").read()
        batch_inv_file = open("cu/batch_inv.cu", "r").read()
        sha256_file = open("cu/sha256_2.cu", "r").read()
        scalarmult_b_file = open("cu/scalarmult_b.cu", "r").read()

        batch_inv_file = get_define("BLOCKS", self.blocks, batch_inv_file)
        batch_inv_file = get_define("N_BATCH", self.n_batch, batch_inv_file)

        sha256_file = get_define("BLOCKS", self.blocks, sha256_file)
        sha256_file = get_define("N_BATCH", self.n_batch, sha256_file)
        sha256_file = get_define("SIZE_LIST", self.target_count, sha256_file)
        sha256_file = get_define("SIZE_LIST_ODD", (self.target_count & 1), sha256_file)
        sha256_file = get_define("BLOOM_FILTER_SIZE", self.bloom_filter_size, sha256_file)
        sha256_file = get_define("K_HASH", self.bloom_filter_k_hash, sha256_file)

        batch_ge_add_file = get_define("BLOCKS", self.blocks, batch_ge_add_file)
        batch_ge_add_file = get_define("N_BATCH", self.n_batch, batch_ge_add_file)

        scalarmult_b_file = get_define("BLOCKS", self.blocks, scalarmult_b_file)
        scalarmult_b_file = get_define("N_BATCH", 256, scalarmult_b_file)

        nvcc_options = ['-use_fast_math', '--generate-line-info']
        nvcc_include_dirs = [os.path.realpath("") + "/inc_cu"]

        self.ctx = self.dev.retain_primary_context()
        self.ctx.push()

        if profiling_enabled:
            drv.start_profiler()

        self.kernel_str = f"{self.blocks}x{self.n_batch}_{self.dev.name().lower().replace(' ', '_')}"

        batch_ge_add = load_module_new('batch_ge_add', batch_ge_add_file, nvcc_options, nvcc_include_dirs, cubin_cache, self.kernel_str)
        batch_inv = load_module_new('batch_inv', batch_inv_file, nvcc_options, nvcc_include_dirs, cubin_cache, self.kernel_str)
        sha256_quad = load_module_new('sha256_quad', sha256_file, nvcc_options, nvcc_include_dirs, cubin_cache, self.kernel_str)
        scalarmult_b_module = load_module_new('scalarmult_b', scalarmult_b_file, nvcc_options, nvcc_include_dirs, cubin_cache, self.kernel_str)

        self.func_batch_ge_add_step0 = batch_ge_add.get_function("batch_ge_add_step0")
        self.func_batch_ge_add_step1 = batch_ge_add.get_function("batch_ge_add_step1")

        self.func_batch_inv_step0 = batch_inv.get_function("batch_inv_step0")
        self.func_batch_inv_step1 = batch_inv.get_function("batch_inv_step1")
        self.func_batch_inv_step2 = batch_inv.get_function("batch_inv_step2")
        self.func_calc_next_point = batch_inv.get_function("calc_next_optimized_point")

        self.func_sha256_quad = sha256_quad.get_function("sha256_quad")

        self.func_scalarmult_b = scalarmult_b_module.get_function("scalarmult_b")
        self.func_bloom_filter_init = sha256_quad.get_function("bloom_filter_init")

        self.y_seed_table_xmy = np.zeros((self.n_batch, 32), dtype="uint8")
        self.y_seed_table_xpy = np.zeros((self.n_batch, 32), dtype="uint8")
        self.y_seed_table_t2d = np.zeros((self.n_batch, 32), dtype="uint8")

        self.found_flag  = np.empty((1), dtype="int32")
        self.found_flag[0] = -1

        self.np_random_x_seed_table = np.zeros((self.blocks, 8), dtype="uint32")

        # Allocate GPU buffers
        # CONSTANT
        self.const_b_table_xmy_gpu = drv.mem_alloc(self.y_seed_table_xmy.nbytes)
        self.const_b_table_xpy_gpu = drv.mem_alloc(self.y_seed_table_xpy.nbytes)
        self.const_b_table_t_gpu = drv.mem_alloc(self.y_seed_table_t2d.nbytes)
        self.fold8_ypx_gpu = drv.mem_alloc(fold8_ypx.nbytes)
        self.fold8_ymx_gpu = drv.mem_alloc(fold8_ymx.nbytes)
        self.fold8_t_gpu = drv.mem_alloc(fold8_t.nbytes)

        # MEMSET 0
        self.data_out_xy_gpu = drv.mem_alloc(self.global_size*4*32)
        self.data_in_z_gpu = drv.mem_alloc(self.global_size*32)
        self.data_w_gpu = drv.mem_alloc(self.global_size*32)
        self.found_flag_gpu = drv.mem_alloc(self.found_flag.nbytes)

        # VARIABLES
        self.data_in_xmy_gpu = drv.mem_alloc(self.blocks*32)
        self.data_in_xpy_gpu = drv.mem_alloc(self.blocks*32)
        self.data_in_t_gpu = drv.mem_alloc(self.blocks*32)

        self.random_x_seed_table_gpu = drv.mem_alloc(self.np_random_x_seed_table.nbytes)

        self.target_list = target_list
        self.target_list.sort(axis=0)

        # Calculate total GPU memory allocated
        total_allocated = (self.y_seed_table_xmy.nbytes*3) + (self.global_size*32*2) + (self.blocks*32*3) + (self.global_size*4*32) + (self.target_list.nbytes) + (self.found_flag.nbytes) + (self.np_random_x_seed_table.nbytes) + (fold8_ypx.nbytes*3) + (self.bloom_filter_size//8)
        print(f"Allocated {round(total_allocated/1000000000, 4)} GB on GPU #{self.gpu_id}, Thread {self.tid}, {self.dev.name()}")

        self.y_seed_table = []
        self.random_x_seed_table = []

        self.target_list_low = []
        self.target_list_high = []

        self.balances = balances
        self.target_list_orig = load_list(list_filename, target_n)

        self.full_haystack = drv.mem_alloc(self.target_list.nbytes)
        self.bsearch_haystack_low = drv.mem_alloc(self.target_list.nbytes//2)
        self.bsearch_haystack_high = drv.mem_alloc(self.target_list.nbytes//2)
        self.bloom_filter_gpu = drv.mem_alloc(self.bloom_filter_size // 8)

        # Prepare functions
        self.func_batch_ge_add_step0.prepare(["P"] * 8)
        self.func_batch_ge_add_step1.prepare(["P"] * 5)

        self.func_batch_inv_step0.prepare(["P"] * 2)
        self.func_batch_inv_step1.prepare(["P"])
        self.func_batch_inv_step2.prepare(["P"] * 3)

        self.func_calc_next_point.prepare(["P"] * 5)

        self.func_sha256_quad.prepare(["P"] * 8)

        self.func_scalarmult_b.prepare(["P"] * 7)
        self.func_bloom_filter_init.prepare(["P"] * 3)

        self.stream = drv.Stream()
        self.start_event = drv.Event()
        self.end_event = drv.Event()

        self.debug_test_found = True

        self.total_iter_count = 0
        self.last_iter_count = 0
        self.total_time = 0
        self.last_stats_str = ''
        self.last_hashrate = 0
        self.output_stats_time = time.time()

        self.total_iter_time = 0
        self.total_gpu_time = 0

        self.scalarmult_b_time = 0

    def gpu_init(self):
        self.y_seed_table = []
        for sk_index in range(1, self.n_batch+1):
            y_seed = (secrets.randbelow(2**200) * 8)
            self.y_seed_table.append(y_seed)

            (y_seed_x, y_seed_y, y_seed_z, y_seed_t, cut) = edp_BasePointMult(y_seed)

            # precomp
            t2d = 2*d*y_seed_t % q
            ymx = (y_seed_y-y_seed_x) % q
            ypx = (y_seed_y+y_seed_x) % q

            # append (xmy, xpy, t2d)
            self.y_seed_table_xmy[sk_index-1] = int_to_nparray(ymx)
            self.y_seed_table_xpy[sk_index-1] = int_to_nparray(ypx)
            self.y_seed_table_t2d[sk_index-1] = int_to_nparray(t2d)

        self.random_x_seed_table = []
        for block in range(self.blocks):
            random_sk = (secrets.randbelow(2**240) * 8)
            self.random_x_seed_table.append(random_sk)
            self.np_random_x_seed_table[block] = int_to_nparray32(random_sk)

        if debug_test:
            if not self.debug_test_found:
                print("ERROR - TEST ADDRESS NOT FOUND\n")
                exit()
            self.debug_test_found = False
            self.target_list = debug_generate(f"GPU #{self.gpu_id}, Thread {self.tid}, {self.dev.name()}",self.blocks, self.n_batch, rekey, self.y_seed_table, self.random_x_seed_table, self.target_count, self.target_list_orig)

        # Prepare target list
        target_list_2 = self.target_list.view(dtype=np.dtype([('f1', np.uint32), ('f2', np.uint32)]))

        self.target_list_low = []
        self.target_list_high = []

        for address in target_list_2:
            self.target_list_low.append([address[0][1]])
            self.target_list_high.append([address[0][0]])

        # low 32 bits of addresses
        self.target_list_low = np.asarray(self.target_list_low, dtype="uint32")

        # high 32 bits of addresses
        self.target_list_high = np.asarray(self.target_list_high, dtype="uint32")

        """
        HOST TO DEVICE MEMORY TRANSFER
        """
        # CONSTANTS
        drv.memcpy_htod(self.const_b_table_xmy_gpu, self.y_seed_table_xmy)
        drv.memcpy_htod(self.const_b_table_xpy_gpu, self.y_seed_table_xpy)
        drv.memcpy_htod(self.const_b_table_t_gpu, self.y_seed_table_t2d)

        drv.memcpy_htod(self.fold8_ypx_gpu, fold8_ypx)
        drv.memcpy_htod(self.fold8_ymx_gpu, fold8_ymx)
        drv.memcpy_htod(self.fold8_t_gpu, fold8_t)

        drv.memcpy_htod(self.random_x_seed_table_gpu, self.np_random_x_seed_table)

        drv.memcpy_htod(self.full_haystack, self.target_list)
        drv.memcpy_htod(self.bsearch_haystack_low, self.target_list_low)
        drv.memcpy_htod(self.bsearch_haystack_high, self.target_list_high)

        # MEMSET 0
        drv.memset_d8(self.data_out_xy_gpu, 0, self.global_size*4*32)
        drv.memset_d8(self.data_in_z_gpu, 0, self.global_size*32)
        drv.memset_d8(self.data_w_gpu, 0, self.global_size*32)

        drv.memset_d8(self.data_in_xmy_gpu, 0, self.blocks*32)
        drv.memset_d8(self.data_in_xpy_gpu, 0, self.blocks*32)
        drv.memset_d8(self.data_in_t_gpu, 0, self.blocks*32)

        drv.memset_d8(self.bloom_filter_gpu, 0, self.bloom_filter_size // 8)

        # VARIABLES
        drv.memcpy_htod(self.found_flag_gpu, self.found_flag)

        self.scalarmult_b_time = self.func_scalarmult_b.prepared_timed_call(
                self.grid_scalarmult, self.block_scalarmult,
                self.fold8_ypx_gpu,
                self.fold8_ymx_gpu,
                self.fold8_t_gpu,
                self.data_in_xmy_gpu,
                self.data_in_xpy_gpu,
                self.data_in_t_gpu,
                self.random_x_seed_table_gpu)()

        self.func_bloom_filter_init.prepared_timed_call(
                (1, 1), (1, 1, 1),
                self.bsearch_haystack_low,
                self.bsearch_haystack_high,
                self.bloom_filter_gpu
        )()

    def run(self):
        self.ctx.push()

        stop = False

        while not stop:
            for iter_count in range(rekey + 1):
                iter_time = time.time()
                self.start_event.record(self.stream)

                # Group addition step 0
                self.func_batch_ge_add_step1.prepared_async_call(
                    self.grid, self.block, self.stream,
                    self.const_b_table_t_gpu,
                    self.data_in_t_gpu,
                    self.data_out_xy_gpu,
                    self.data_in_z_gpu,
                    self.data_w_gpu)

                # Group addition step 1
                self.func_batch_ge_add_step0.prepared_async_call(
                    self.grid_2, self.block, self.stream,
                    self.const_b_table_xpy_gpu,
                    self.const_b_table_xmy_gpu,
                    self.const_b_table_t_gpu,
                    self.data_in_xmy_gpu,
                    self.data_in_xpy_gpu,
                    self.data_in_t_gpu,
                    self.data_out_xy_gpu,
                    self.data_w_gpu)

                # Batched group inversion step 0
                self.func_batch_inv_step0.prepared_async_call(
                    self.grid256, self.block, self.stream,
                    self.data_in_z_gpu,
                    self.data_w_gpu)

                # Batched group inversion step 1
                self.func_batch_inv_step1.prepared_async_call(
                    self.grid256, self.block, self.stream,
                    self.data_w_gpu)

                # Batched group inversion step 2
                self.func_batch_inv_step2.prepared_async_call(
                    self.grid256, self.block, self.stream,
                    self.data_in_z_gpu,
                    self.data_in_z_gpu,
                    self.data_w_gpu)

                # Calculate next optimized point
                self.func_calc_next_point.prepared_async_call(
                    self.grid256, self.block, self.stream,
                    self.data_in_xmy_gpu,
                    self.data_in_xpy_gpu,
                    self.data_in_t_gpu,
                    self.data_out_xy_gpu,
                    self.data_in_z_gpu)

                # SHA256 -> Bloom filter -> Binary search
                self.func_sha256_quad.prepared_async_call(
                    self.gridsha256, self.block, self.stream,
                    self.data_out_xy_gpu,
                    self.data_w_gpu,
                    self.data_in_z_gpu,
                    self.bloom_filter_gpu,
                    self.bsearch_haystack_high,
                    self.bsearch_haystack_low,
                    self.full_haystack,
                    self.found_flag_gpu)

                # Get result
                drv.memcpy_dtoh_async(self.found_flag, self.found_flag_gpu, self.stream)

                self.end_event.record(self.stream)

                self.stream.synchronize()
                self.end_event.synchronize()

                dur = self.start_event.time_till(self.end_event)

                if self.found_flag[0] != -1:
                    self.debug_test_found = test_found(f"GPU #{self.gpu_id}, Thread {self.tid}, {self.dev.name()}", self.found_flag, iter_count, self.n_batch, self.y_seed_table, self.blocks, self.random_x_seed_table, self.balances, args.output_filename, debug_test, self.target_list)

                    self.found_flag[0] = -1
                    drv.memcpy_htod(self.found_flag_gpu, self.found_flag)

                iter_time = time.time() - iter_time

                self.total_time += iter_time
                self.total_iter_time += iter_time
                self.total_gpu_time += dur

                if (time.time()-self.output_stats_time)>output_stats_sec and self.total_iter_count>0:
                    if self.stopped():
                        stop = True
                        break

                    since_last = self.total_iter_count-self.last_iter_count
                    self.last_stats_str = f"[GPU #{self.gpu_id}, {self.dev.name()}|Iter #{self.total_iter_count}|Speed {round(((self.global_size*16 * since_last) / self.total_time) / 1000000, 2)} MH/s|{round((self.total_gpu_time/since_last), 2)}/{round((self.total_iter_time/since_last) * 1000, 2)} ms/iter]"
                    self.last_hashrate = ((self.global_size*16 * since_last) / self.total_time) / 1000000
                    self.output_stats_time = time.time()
                    self.last_iter_count = self.total_iter_count
                    self.total_time = 0
                    self.total_iter_time = 0
                    self.total_gpu_time = 0

                self.total_iter_count +=1

            if not stop:
                # Rekey
                self.gpu_init()

        self.gpu_cleanup()

    def gpu_cleanup(self):
        self.ctx.synchronize()

        if profiling_enabled:
            drv.stop_profiler()

        self.ctx.pop()

        self.fold8_ypx_gpu.free()
        self.fold8_ymx_gpu.free()
        self.fold8_t_gpu.free()

        self.const_b_table_xmy_gpu.free()
        self.const_b_table_xpy_gpu.free()
        self.const_b_table_t_gpu.free()

        self.full_haystack.free()
        self.bloom_filter_gpu.free()
        self.bsearch_haystack_low.free()
        self.bsearch_haystack_high.free()

        self.data_out_xy_gpu.free()
        self.data_in_z_gpu.free()
        self.data_w_gpu.free()
        self.found_flag_gpu.free()

        self.data_in_xmy_gpu.free()
        self.data_in_xpy_gpu.free()
        self.data_in_t_gpu.free()

        self.random_x_seed_table_gpu.free()

    def stop(self):
        self.ctx.pop()
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    print("current python version (tested on 3.8.5): %s" % sys.version)
    print("current pycuda version (tested on 2020.1): %s" % pycuda.VERSION_TEXT)

    drv.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-idxs', dest="gpu_idxs", default="0", help="Run on specified gpus idxs separated by comma. Default: 0")
    parser.add_argument('--input-targets-file', dest="list_filename", default="list_01.txt", help="CSV list of targets, comma delimiter, target in row 1. Default: list_01.txt")
    parser.add_argument('--n-targets', dest="n_target", type=int, default=5000, help="Number of targets to search, recommended below 10000 for better performance. Default: 5000")
    parser.add_argument('--output-results-file', dest="output_filename", default="results.txt", help="Write found targets to this file. Default: results.txt")
    parser.add_argument('--test-mode', dest="debug_test", type=str2bool, default=False, help="Enable test mode (crash if there is a calculation error). Default: False")
    parser.add_argument('--batch-sizes', dest="n_batchs", default='512', help="batch size per gpu separated by comma, lower to use less ram/vram but degrade performance. Default: 512")

    args = parser.parse_args()

    if args.debug_test:
        print("Warning : TEST MODE ACTIVATED. " + args.list_filename + " unused !")

    if args.n_target>50000:
        print("Warning : very high --n-targets value, performance degraded")

    gpu_idxs = args.gpu_idxs
    if ',' in gpu_idxs:
        gpu_idxs = [int(x) for x in gpu_idxs.split(',')]
    else:
        gpu_idxs = [int(gpu_idxs)]

    n_batchs = args.n_batchs
    if ',' in n_batchs:
        n_batchs = [int(x) for x in n_batchs.split(',')]
    else:
        n_batchs = [int(n_batchs)]

    while len(n_batchs) < len(gpu_idxs):
        n_batchs.append(512)

    for n_batch in n_batchs:
        if n_batch not in [16, 32, 64, 128, 256, 512]:
            print()
            print(f"Error : Invalid batch size {n_batch}, valid batch sizes : [16, 32, 64, 128, 256, 512]")
            exit()

    list_filename = args.list_filename
    debug_test = args.debug_test
    output_stats_sec = 5 # OUTPUT STATS EVERY N SECONDS
    profiling_enabled = False
    cubin_cache = True  # SAVE COMPILED KERNELS TO DISK

    # BLOOM FILTER CONFIG
    bloom_filter_k_hash = 5
    bloom_filter_p = 1.0e-10
    target_n = args.n_target

    target_list = load_list(list_filename, target_n)
    balances = load_balances(list_filename, target_n)

    bloom_filter_size = next_prime(math.ceil(target_list.size * (-bloom_filter_k_hash / math.log(1 - math.exp(math.log(bloom_filter_p) / bloom_filter_k_hash)))))

    # CHOOSE MODE (REAL/DEMO)
    rekey = 1000
    if debug_test:
        rekey = 100

    print("LIST OF AVAILABLE CUDA DEVICES :")
    for i in range(drv.Device.count()):
        dev_info = drv.Device(i)
        print(" -GPU IDX: %d, %s" % (i, dev_info.name()))
        print("   Compute Capability: %d.%d" % dev_info.compute_capability())
        print("   Total Memory: %s KB" % (dev_info.total_memory() // (1024)))
        print("   MULTIPROCESSOR_COUNT: %d" % (dev_info.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)))
        print("   WARP_SIZE: %d" % (dev_info.get_attribute(drv.device_attribute.WARP_SIZE)))

    print()

    available_idxs = [int(x) for x in range(drv.Device.count())]
    for gpu_idx in gpu_idxs:
        if gpu_idx not in available_idxs:
            print()
            print(f"Error : Invalid gpu_idx {gpu_idx}")
            exit()

    output_stats_time = time.time()

    gpu_thread_list = []
    for idx, gpu_idx in enumerate(gpu_idxs):
        gpu_thread = GPUThread(gpu_idx, n_batchs[idx], idx)
        gpu_thread.gpu_init()
        gpu_thread_list.append(gpu_thread)

    try:
        for gpu_idx in range(len(gpu_idxs)):
            gpu_thread_list[gpu_idx].start()

        done = False
        while not done:
            for gpu_idx in range(len(gpu_idxs)):
                if not gpu_thread_list[gpu_idx].is_alive():
                    done = True

            time.sleep(1)
            if (time.time()-output_stats_time)>output_stats_sec:
                out_str = ''
                total_speed = 0
                for gpu_idx in range(len(gpu_idxs)):
                    total_speed += gpu_thread_list[gpu_idx].last_hashrate

                    if gpu_thread_list[gpu_idx].last_stats_str != '':
                        out_str += gpu_thread_list[gpu_idx].last_stats_str
                        out_str += '\n'
                out_str += f"[Total speed {round(total_speed, 2)} MH/s]  "
                print(out_str)
                output_stats_time = time.time()

        for gpu_idx in range(len(gpu_idxs)):
            gpu_thread_list[gpu_idx].join()

        print()

    except KeyboardInterrupt:
        for gpu_idx in range(len(gpu_idxs)):
            gpu_thread_list[gpu_idx].stop()
        print("\nShutdown requested...exiting")
    except:
        print("Unexpected error...exiting")
        raise