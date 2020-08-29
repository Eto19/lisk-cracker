def main():
    try:
        print("Starting...")
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu-idx', dest="device_id", type=int, default=0, help="Run on specified gpu idx. multi-gpu not supported. Default: 0")
        parser.add_argument('--input-targets-file', dest="list_filename", default="list_01.txt", help="CSV list of targets, comma delimiter, target in row 1. Default: list_01.txt")
        parser.add_argument('--n-targets', dest="n_target", type=int, default=1000, help="Number of targets to search, fill with targets from 0 to n if n>target in file, recommended below 10000 for better performance. Default: 1000")
        parser.add_argument('--output-results-file', dest="output_filename", default="results.txt", help="Write found targets to this file. Default: results.txt")
        parser.add_argument('--testing-mode', dest="debug_test", type=bool, default=False, help="Activate this option to test dummy targets, used to check computations and performance, if there is an error the program will crash, if everything is working it will output results every 20 iterations. Default: False")
        parser.add_argument('--batch-size', dest="n_batch", type=int, default=512, choices=[16, 32, 64, 128, 256, 512], help="batch size, lower to use less vram. Default: 512")

        args = parser.parse_args()

        if args.debug_test:
            print("DEBUG MODE ACTIVATED. NOT SEARCHING IN " + args.list_filename + " !")

        device_id = args.device_id
        list_filename = args.list_filename
        n_batch = args.n_batch
        debug_test = args.debug_test
        output_stats = 50 # OUTPUT STATS EVERY X ITERATIONS
        regen_sk = 1000 # REGENERATE RANDOM SEED KEYS EVERY X ITERATIONS
        profiling_enabled = False
        cubin_cache = True # SAVE COMPILED KERNELS TO DISK

        # BLOOM FILTER CONFIG
        bloom_filter_k_hash = 5
        bloom_filter_p = 1.0e-10
        target_n = args.n_target

        print(pycuda.VERSION)
        print(pycuda.VERSION_STATUS)
        print(pycuda.VERSION_TEXT)

        drv.init()
        print(sys.version)

        print("LIST OF AVAILABLE CUDA DEVICES :")
        for i in range(drv.Device.count()):
            dev = drv.Device(i)
            print(" -GPU IDX: %d, name: %s" % (i, dev.name()))
            print("   Compute Capability: %d.%d" % dev.compute_capability())
            print("   Total Memory: %s KB" % (dev.total_memory()//(1024)))
            print("   MULTIPROCESSOR_COUNT: %d" % (dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)))
            print("   WARP_SIZE: %d" % (dev.get_attribute(drv.device_attribute.WARP_SIZE)))

        blocks = dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)*8*64

        dev = drv.Device(device_id)

        # drv.ctx_flags.SCHED_BLOCKING_SYNC | drv.event_flags.BLOCKING_SYNC
        ctx = dev.make_context(drv.ctx_flags.SCHED_BLOCKING_SYNC)

        stream = drv.Stream(drv.event_flags.BLOCKING_SYNC)
        start_event = drv.Event(drv.event_flags.BLOCKING_SYNC)
        end_event = drv.Event(drv.event_flags.BLOCKING_SYNC)

        if profiling_enabled:
            drv.start_profiler()

        device_data = pycuda.tools.DeviceData()

        print()
        print("SELECTED GPU IDX : %d " % device_id)
        print("registers : %d " % device_data.registers)
        print("shared_memory : %d " % device_data.shared_memory)
        print()

        nvcc_options = ['-use_fast_math', '--generate-line-info']
        nvcc_include_dirs = [os.path.realpath("") + "/inc_cu"]

        # CHOOSE MODE (REAL/DEMO)
        if debug_test:
            n_address_list = target_n
            n_loop = 100000
            regen_sk = 20 # IN TESTING MODE REGEN SK EVERY 20 ITERATION
        else:
            address_list = load_list(list_filename, target_n)
            balances = load_balances(list_filename, target_n)
            n_address_list = address_list.size
            n_loop = 100000000000000

        # Calculated parameters
        block_size = n_batch
        grid = (blocks, 1)
        grid_2 = (blocks*2, 1)
        grid256 = (blocks//n_batch, 1)
        grid256_2 = (blocks//n_batch, 1)
        gridsha256 = (blocks*8, 1)
        block = (block_size, 1, 1)

        grid_scalarmult = (blocks//256, 1)
        block_scalarmult = (256, 1, 1)

        global_size = block_size*blocks

        # Calculate bloom filter size
        bloom_filter_size = next_prime(math.ceil(n_address_list * (-bloom_filter_k_hash / math.log(1 - math.exp(math.log(bloom_filter_p) / bloom_filter_k_hash)))))

        print("blocks: %d" % blocks)
        print("block_size: %d" % block_size)
        print("global_size: %d" % global_size)
        print("grid: %d" % grid[0])
        print("grid256: %d" % grid256[0])
        print("gridsha256: %d" % gridsha256[0])
        print()
        print("Addresses to search : %d" % n_address_list)
        print("bloom_filter_size : %d" % bloom_filter_size)
        print()

        # Load cuda sources files
        batch_ge_add_file = open("cu/batch_ge_add.cu", "r").read()
        batch_inv_file = open("cu/batch_inv.cu", "r").read()
        sha256_file = open("cu/sha256_2.cu", "r").read()
        scalarmult_b_file = open("cu/scalarmult_b.cu", "r").read()

        # Add define
        sha256_file = get_define("BLOCKS", blocks, sha256_file)
        sha256_file = get_define("N_BATCH", n_batch, sha256_file)

        batch_inv_file = get_define("BLOCKS", blocks, batch_inv_file)
        batch_inv_file = get_define("N_BATCH", n_batch, batch_inv_file)

        sha256_file = get_define("SIZE_LIST", n_address_list, sha256_file)
        sha256_file = get_define("SIZE_LIST_ODD", (n_address_list&1), sha256_file)
        sha256_file = get_define("BLOOM_FILTER_SIZE", bloom_filter_size, sha256_file)
        sha256_file = get_define("K_HASH", bloom_filter_k_hash, sha256_file)
        
        batch_ge_add_file = get_define("BLOCKS", blocks, batch_ge_add_file)
        batch_ge_add_file = get_define("N_BATCH", n_batch, batch_ge_add_file)

        scalarmult_b_file = get_define("BLOCKS", blocks, scalarmult_b_file)
        scalarmult_b_file = get_define("N_BATCH", 256, scalarmult_b_file)

        # Load modules
        batch_ge_add = load_module_new('batch_ge_add', batch_ge_add_file, nvcc_options, nvcc_include_dirs, cubin_cache)
        batch_inv = load_module_new('batch_inv', batch_inv_file, nvcc_options, nvcc_include_dirs, cubin_cache)
        sha256_quad = load_module_new('sha256_quad', sha256_file, nvcc_options, nvcc_include_dirs, cubin_cache)
        scalarmult_b_module = load_module_new('scalarmult_b', scalarmult_b_file, nvcc_options, nvcc_include_dirs, cubin_cache) 

        print()

        # Load kernels from modules
        func_batch_ge_add_step0 = batch_ge_add.get_function("batch_ge_add_step0")
        func_batch_ge_add_step1 = batch_ge_add.get_function("batch_ge_add_step1")

        func_batch_inv_step0 = batch_inv.get_function("batch_inv_step0")
        func_batch_inv_step1 = batch_inv.get_function("batch_inv_step1")
        func_batch_inv_step2 = batch_inv.get_function("batch_inv_step2")
        func_calc_next_point = batch_inv.get_function("calc_next_optimized_point")

        func_sha256_quad = sha256_quad.get_function("sha256_quad")

        func_scalarmult_b = scalarmult_b_module.get_function("scalarmult_b")
        func_bloom_filter_init = sha256_quad.get_function("bloom_filter_init")

        # Initialize np buffers
        B_mult_table_xmy = np.zeros((n_batch, 32), dtype="uint8")
        B_mult_table_xpy = np.zeros((n_batch, 32), dtype="uint8")
        B_mult_table_t2d = np.zeros((n_batch, 32), dtype="uint8")

        data_out_xy = np.empty((global_size*4, 32), dtype="uint8")
        data_in_z   = np.empty((global_size, 32), dtype="uint8")
        data_w      = np.empty((global_size, 32), dtype="uint8")

        found_flag  = np.empty((1), dtype="int32")
        found_flag[0] = -1

        data_in_xmy = np.zeros((blocks, 32), dtype="uint8")
        data_in_xpy = np.zeros((blocks, 32), dtype="uint8")
        data_in_t = np.zeros((blocks, 32), dtype="uint8")

        # FILL WITH DUMMY TARGETS
        if debug_test:
            address_list_orig = load_list(list_filename, target_n)
            balances = load_balances(list_filename, target_n)
            if target_n >= address_list_orig.size:
                address_list = np.copy(address_list_orig)
            else:
                address_list_orig = np.random.randint(0, 2**64, (target_n, 1), dtype="uint64")
                address_list = np.copy(address_list_orig)

        # Allocate GPU buffers
        # CONSTANT
        const_b_table_xmy_gpu = drv.mem_alloc(B_mult_table_xmy.nbytes)
        const_b_table_xpy_gpu = drv.mem_alloc(B_mult_table_xpy.nbytes)
        const_b_table_t_gpu = drv.mem_alloc(B_mult_table_t2d.nbytes)
        fold8_ypx_gpu = drv.mem_alloc(fold8_ypx.nbytes)
        fold8_ymx_gpu = drv.mem_alloc(fold8_ymx.nbytes)
        fold8_t_gpu = drv.mem_alloc(fold8_t.nbytes)

        # MEMSET 0
        data_out_xy_gpu = drv.mem_alloc(data_out_xy.nbytes)
        data_in_z_gpu = drv.mem_alloc(data_in_z.nbytes)
        data_w_gpu = drv.mem_alloc(data_w.nbytes)
        found_flag_gpu = drv.mem_alloc(found_flag.nbytes)

        # VARIABLES
        data_in_xmy_gpu = drv.mem_alloc(data_in_xmy.nbytes)
        data_in_xpy_gpu = drv.mem_alloc(data_in_xpy.nbytes)
        data_in_t_gpu = drv.mem_alloc(data_in_t.nbytes)

        np_random_sk_array = np.zeros((blocks, 8), dtype="uint32")
        np_random_sk_array_gpu = drv.mem_alloc(np_random_sk_array.nbytes)

        # Calculate total memory allocated
        total_allocated = (B_mult_table_xmy.nbytes*3) + (data_w.nbytes*2) + (data_in_xmy.nbytes*3) + (data_out_xy.nbytes) + (address_list.nbytes) + (found_flag.nbytes) + (np_random_sk_array.nbytes) + (fold8_ypx.nbytes*3) + (bloom_filter_size//8)
        print("Total VRAM usage: %f GBytes" % (total_allocated/1000000000))

        # Generate optimized B table
        B_mult_table_sk = []

        # Generate optimized B table
        B_mult_table_sk = []

        k = 2*d % q
        for i in range(1, n_batch+1):
            B_mult_sk = (secrets.randbelow(2**200) * 8)
            xB = edp_BasePointMult(B_mult_sk)
            B_mult_table_sk.append(B_mult_sk)
            (x, y, z, t, cut) = xB

            # precomp
            t2d = k*t % q
            xmy = (y-x) % q
            xpy = (y+x) % q

            # append (xmy, xpy, t2d)
            B_mult_table_xmy[i-1] = int_to_nparray(xmy)
            B_mult_table_xpy[i-1] = int_to_nparray(xpy)
            B_mult_table_t2d[i-1] = int_to_nparray(t2d)

        """
        INITIAL HOST TO DEVICE MEMORY TRANSFER
        """
        # CONSTANT
        drv.memcpy_htod(const_b_table_xmy_gpu, B_mult_table_xmy)
        drv.memcpy_htod(const_b_table_xpy_gpu, B_mult_table_xpy)
        drv.memcpy_htod(const_b_table_t_gpu, B_mult_table_t2d)

        # MEMSET 0
        drv.memset_d8(data_out_xy_gpu, 0, data_out_xy.nbytes)
        drv.memset_d8(data_in_z_gpu, 0, data_in_z.nbytes)
        drv.memset_d8(data_w_gpu, 0, data_w.nbytes)

        # VARIABLES
        drv.memcpy_htod(data_in_xmy_gpu, data_in_xmy)
        drv.memcpy_htod(data_in_xpy_gpu, data_in_xpy)
        drv.memcpy_htod(data_in_t_gpu, data_in_t)

        drv.memcpy_htod(fold8_ypx_gpu, fold8_ypx)
        drv.memcpy_htod(fold8_ymx_gpu, fold8_ymx)
        drv.memcpy_htod(fold8_t_gpu, fold8_t)

        # set found flag to -1
        drv.memcpy_htod(found_flag_gpu, found_flag)

        func_scalarmult_b.prepare(["P"]*7)
        func_bloom_filter_init.prepare(["P"]*3)

        elapsed_time = 0

        # GENERATE SECRET KEYS AND PUBLIC KEYS
        random_sk_array = []
        for i in range(blocks):
            random_sk = (secrets.randbelow(2**240) * 8)
            random_sk_array.append(random_sk)
            np_random_sk_array[i] = int_to_nparray32(random_sk)


        drv.memcpy_htod(np_random_sk_array_gpu, np_random_sk_array)

        elapsed_time_1 = func_scalarmult_b.prepared_timed_call(
                grid_scalarmult, block_scalarmult,
                fold8_ypx_gpu,
                fold8_ymx_gpu,
                fold8_t_gpu,
                data_in_xmy_gpu, 
                data_in_xpy_gpu, 
                data_in_t_gpu,
                np_random_sk_array_gpu)()

        print("-- Generate sk and pk --")

        # GENERATE OF TESTING PARAMETERS
        if debug_test:
            address_list = debug_generate(blocks, n_batch, regen_sk, B_mult_table_sk, random_sk_array, n_address_list, address_list_orig)

        # Sort haystack and transfer to device
        address_list.sort(axis=0)

        full_haystack = drv.mem_alloc(address_list.nbytes)
        drv.memcpy_htod(full_haystack, address_list)

        address_list2 = address_list.view(dtype=np.dtype([('f1', np.uint32), ('f2', np.uint32)]))

        address_list4 = []
        address_list3 = []
        for x in address_list2:
            address_list3.append([x[0][1]])
            address_list4.append([x[0][0]])

        address_list3 = np.asarray(address_list3, dtype="uint32")
        address_list4 = np.asarray(address_list4, dtype="uint32")

        bsearch_haystack2 = drv.mem_alloc(address_list3.nbytes)
        drv.memcpy_htod(bsearch_haystack2, address_list3)

        bsearch_haystack = drv.mem_alloc(address_list4.nbytes)
        drv.memcpy_htod(bsearch_haystack, address_list4)

        bloom_filter = np.zeros(bloom_filter_size//8, dtype="uint8")
        bloom_filter_gpu = drv.mem_alloc(bloom_filter.nbytes)
        drv.memcpy_htod(bloom_filter_gpu, bloom_filter)

        func_bloom_filter_init.prepared_timed_call(
                (1, 1), (1, 1, 1),
                bsearch_haystack2,
                bsearch_haystack,
                bloom_filter_gpu
        )()

        # Prepare functions
        func_batch_ge_add_step0.prepare(["P"]*8)
        func_batch_ge_add_step1.prepare(["P"]*5)

        func_batch_inv_step0.prepare(["P"]*2)
        func_batch_inv_step1.prepare(["P"])
        func_batch_inv_step2.prepare(["P"]*3)

        func_calc_next_point.prepare(["P"]*5)

        func_sha256_quad.prepare(["P"]*8)

        print()
        total_computed = 0
        loop_count_1 = 1
        loop_count = 1
        inc_count = 0
        if debug_test:
            debug_test_found = 0

        for x in range(0, n_loop):
            if ((x%regen_sk)==0) and (x>0):
                print("-- Regen sk and pk --")
                # GENERATE SECRET KEYS AND PUBLIC KEYS
                random_sk_array = []
                for i in range(blocks):
                    random_sk = (secrets.randbelow(2**245) * 8)
                    random_sk_array.append(random_sk)
                    np_random_sk_array[i] = int_to_nparray32(random_sk)

                # GENERATE OF TESTING PARAMETERS
                if debug_test:
                    if debug_test_found == 0:
                        print("ERROR NOT FOUND\n")
                        break
                    debug_test_found = 0

                    address_list = debug_generate(blocks, n_batch, regen_sk, B_mult_table_sk, random_sk_array, n_address_list, address_list_orig)

                full_haystack = drv.mem_alloc(address_list.nbytes)
                drv.memcpy_htod(full_haystack, address_list)
                
                address_list2 = address_list.view(dtype=np.dtype([('f1', np.uint32), ('f2', np.uint32)]))

                address_list4 = []
                address_list3 = []
                for address_conv in address_list2:
                    address_list3.append([address_conv[0][1]])
                    address_list4.append([address_conv[0][0]])

                address_list3 = np.asarray(address_list3, dtype="uint32")
                address_list4 = np.asarray(address_list4, dtype="uint32")

                drv.memcpy_htod(bsearch_haystack, address_list4)
                drv.memcpy_htod(bsearch_haystack2, address_list3)

                bloom_filter = np.zeros(bloom_filter_size//8, dtype="uint8")
                bloom_filter_gpu = drv.mem_alloc(bloom_filter.nbytes)
                drv.memcpy_htod(bloom_filter_gpu, bloom_filter)

                func_bloom_filter_init.prepared_timed_call(
                        (1, 1), (1, 1, 1),
                        bsearch_haystack2,
                        bsearch_haystack,
                        bloom_filter_gpu
                )()

                # CONSTANT
                drv.memcpy_htod(const_b_table_xmy_gpu, B_mult_table_xmy)
                drv.memcpy_htod(const_b_table_xpy_gpu, B_mult_table_xpy)
                drv.memcpy_htod(const_b_table_t_gpu, B_mult_table_t2d)

                # MEMSET 0
                drv.memset_d8(data_out_xy_gpu, 0, data_out_xy.nbytes)
                drv.memset_d8(data_in_z_gpu, 0, data_in_z.nbytes)
                drv.memset_d8(data_w_gpu, 0, data_w.nbytes)

                # VARIABLES
                drv.memcpy_htod(data_in_xmy_gpu, data_in_xmy)
                drv.memcpy_htod(data_in_xpy_gpu, data_in_xpy)
                drv.memcpy_htod(data_in_t_gpu, data_in_t)

                drv.memcpy_htod(fold8_ypx_gpu, fold8_ypx)
                drv.memcpy_htod(fold8_ymx_gpu, fold8_ymx)
                drv.memcpy_htod(fold8_t_gpu, fold8_t)

                # set found flag to -1
                drv.memcpy_htod(found_flag_gpu, found_flag)

                drv.memcpy_htod(np_random_sk_array_gpu, np_random_sk_array)

                elapsed_time_1 = func_scalarmult_b.prepared_timed_call(
                        grid_scalarmult, block_scalarmult,
                        fold8_ypx_gpu,
                        fold8_ymx_gpu,
                        fold8_t_gpu,
                        data_in_xmy_gpu, 
                        data_in_xpy_gpu, 
                        data_in_t_gpu,
                        np_random_sk_array_gpu)()

                inc_count = 0

            start_event.record(stream)

            # Batched ge addition 1
            func_batch_ge_add_step1.prepared_async_call(
                grid, block, stream,
                const_b_table_t_gpu,
                data_in_t_gpu, 
                data_out_xy_gpu, 
                data_in_z_gpu,
                data_w_gpu)

            # Batched ge addition 0
            func_batch_ge_add_step0.prepared_async_call(
                grid_2, block, stream,
                const_b_table_xpy_gpu,
                const_b_table_xmy_gpu,
                const_b_table_t_gpu,
                data_in_xmy_gpu, 
                data_in_xpy_gpu,
                data_in_t_gpu, 
                data_out_xy_gpu,
                data_w_gpu)

            # Batched inversion step 0
            func_batch_inv_step0.prepared_async_call(
                grid256, block, stream,
                data_in_z_gpu, 
                data_w_gpu)

            # Batched inversion step 1
            func_batch_inv_step1.prepared_async_call(
                grid256, block, stream,
                data_w_gpu)

            # Batched inversion step 2
            func_batch_inv_step2.prepared_async_call(
                grid256, block, stream,
                data_in_z_gpu, 
                data_in_z_gpu, 
                data_w_gpu)

            # Calculate next optimized point
            func_calc_next_point.prepared_async_call(
                grid256_2, block, stream,
                data_in_xmy_gpu, 
                data_in_xpy_gpu, 
                data_in_t_gpu, 
                data_out_xy_gpu,
                data_in_z_gpu)

            # Sha256 X4
            func_sha256_quad.prepared_async_call(
                gridsha256, block, stream,
                data_out_xy_gpu, 
                data_w_gpu,
                data_in_z_gpu,
                bloom_filter_gpu,
                bsearch_haystack,
                bsearch_haystack2,
                full_haystack,
                found_flag_gpu)

            # Get result
            drv.memcpy_dtoh_async(found_flag, found_flag_gpu, stream)

            end_event.record(stream)

            stream.synchronize()
            end_event.synchronize()

            elapsed_time += start_event.time_till(end_event)

            if found_flag[0] != -1:
                print("found_flag activated... Checking...")
                found_flag_i = found_flag[0]>>1
                increment_ff = math.floor(((found_flag_i) % (n_batch)))
                increment_1 =     B_mult_table_sk[increment_ff] + inc_count * B_mult_table_sk[(n_batch-1)]
                increment_2 = l - B_mult_table_sk[increment_ff] + inc_count * B_mult_table_sk[(n_batch-1)]


                c_id1 = (found_flag_i) - (blocks * n_batch);
                c_id2 = (found_flag_i) - (blocks * n_batch)*2;
                c_id3 = (found_flag_i) - (blocks * n_batch)*3;

                cid_1 = ((found_flag_i) < (blocks * n_batch));
                cid_2 = ((found_flag_i) < (blocks * n_batch)*2);
                cid_3 = ((found_flag_i) < (blocks * n_batch)*3);

                c_id_array = [(found_flag_i-increment_ff)//n_batch, (c_id1-increment_ff)//n_batch, (c_id2-increment_ff)//n_batch, (c_id3-increment_ff)//n_batch]
                for sk_index in c_id_array:
                    if sk_index < len(random_sk_array):
                        sk_index = math.floor(sk_index)

                        found_sk_orig = random_sk_array[sk_index]
                        found_sk_hex = int_to_hex(found_sk_orig)

                        found_sk_1 = found_sk_orig + increment_1
                        found_sk_2 = found_sk_orig + increment_2

                        found_sk_hex_calc_1 = int_to_hex(found_sk_1)
                        found_sk_hex_calc_2 = int_to_hex(found_sk_2)

                        found_pks_1 = privatetopublickey(found_sk_1)
                        found_pks_2 = privatetopublickey(found_sk_2)

                        found_pk = 0
                        select_address = 0
                        found = 0

                        for pk in found_pks_1: 
                            if np.isin(int(liskpktoaddr(pk)), address_list):
                                found_pk = pk
                                select_address = str(int(liskpktoaddr(pk)))
                                found_sk_hex_calc = found_sk_hex_calc_1
                                increment = increment_1
                                found = 1

                        for pk in found_pks_2: 
                            if np.isin(int(liskpktoaddr(pk)), address_list):
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

                            print("Target found at gid : " + str(found_flag[0]) + " / loop " + str(inc_count) + " / sk index " + str(sk_index))
                            print("Original sk : " + str(found_sk_hex))
                            print("Calculated sk / increment : " + str(found_sk_hex_calc) + " / " + str(increment)) 
                            print("Calculated pk : " + str(found_pk))
                            print("Balance : " + str(balance))
                            print("Target : " + str(select_address) + "L")
                            print(str(select_address) + "L," + str(found_pk.decode('utf-8')) + "," + str(found_sk_hex_calc.decode('utf-8')))
                            print()

                            if debug_test == False:
                                res_file = open(str(args.output_filename),"a+") 
                                res_file.write(str(select_address) + "L," + str(found_pk.decode('utf-8')) + "," + str(found_sk_hex_calc.decode('utf-8')) + "," + str(balance) + "\n") 
                                res_file.close() 


                            if debug_test:
                                debug_test_found = 1

                            found_flag[0] = -1
                            drv.memcpy_htod(found_flag_gpu, found_flag)
                            break

            if ((x%output_stats)==0) and (x>0):
                n_generated = global_size*16*loop_count_1
                print("Iteration #%d | Hashrate : %s MH/s" % (loop_count-1, str((n_generated/elapsed_time)/1000)))
                elapsed_time = 0
                loop_count_1 = 0

            total_computed += global_size*16
            loop_count_1 +=1
            loop_count += 1
            inc_count += 1

    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except:
        print("Unexpected error...exiting")
        raise
    print()

    ctx.synchronize()
    if profiling_enabled:
        drv.stop_profiler()
    ctx.pop()
    del ctx

    # FREE GPU buffers
    # CONSTANT
    const_b_table_xmy_gpu.free()
    const_b_table_xpy_gpu.free()
    const_b_table_t_gpu.free()

    bsearch_haystack.free()
    bloom_filter_gpu.free()
    bsearch_haystack2.free()
    # MEMSET 0
    data_out_xy_gpu.free()
    data_in_z_gpu.free()
    data_w_gpu.free()
    found_flag_gpu.free()

    # VARIABLES
    data_in_xmy_gpu.free()
    data_in_xpy_gpu.free()
    data_in_t_gpu.free()

    print("Register usage per kernels")
    print_reg(func_scalarmult_b, 'func_scalarmult_b')
    print_reg(func_batch_ge_add_step0, 'func_batch_ge_add_step0')
    print_reg(func_batch_ge_add_step1, 'func_batch_ge_add_step1')
    print_reg(func_batch_inv_step0, 'func_batch_inv_step0')
    print_reg(func_batch_inv_step1, 'func_batch_inv_step1')
    print_reg(func_batch_inv_step2, 'func_batch_inv_step2')
    print_reg(func_calc_next_point, 'func_calc_next_point')
    print_reg(func_sha256_quad, 'func_sha256_quad')

if __name__ == "__main__":
    from lib_ed import *
    main()