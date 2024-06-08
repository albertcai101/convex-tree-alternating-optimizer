from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process
from datetime import datetime
import numpy as np
import pandas as pd
import tracemalloc
import time


def work_with_shared_memory(shm_name, shape, dtype):
    # print(f'With SharedMemory: {current_process()=}')
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.recarray from the buffer of the shared memory
    np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)

    # sleep for 0.0811159610748291 seconds
    # time.sleep(0.0811159610748291)
    
    return np.nansum(np_array.val)


def work_no_shared_memory(np_array: np.recarray):
    # print(f'No SharedMemory: {current_process()=}')
    # Without shared memory, the np_array is copied into the child process
    return np.nansum(np_array.val)


if __name__ == "__main__":
    # Make a large data frame with date, float and character columns
    a = [
        (datetime.today(), 1, 'string'),
        (datetime.today(), np.nan, 'abc'),
    ] * 340909
    df = pd.DataFrame(a, columns=['date', 'val', 'character_col'])
    # Convert into numpy recarray to preserve the dtypes 
    np_array = df.to_records(index=False, column_dtypes={'character_col': 'S6'})
    del df
    shape, dtype = np_array.shape, np_array.dtype
    print(f"np_array's size={np_array.nbytes/1e6}MB")

    num_computations_list = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    # With shared memory
    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    with SharedMemoryManager() as smm:
        # Create a shared memory of size np_arry.nbytes
        shm = smm.SharedMemory(np_array.nbytes)
        # Create a np.recarray using the buffer of shm
        shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
        # Copy the data into the shared memory
        np.copyto(shm_np_array, np_array)
        # Spawn some processes to do some work
        for num_computations in num_computations_list:
            sub_start_time = time.time()
            with ProcessPoolExecutor(cpu_count()) as exe:
                fs = [exe.submit(work_with_shared_memory, shm.name, shape, dtype)
                    for _ in range(num_computations)]
                for _ in as_completed(fs):
                    pass
            # print the result of the computation
            result = [f.result() for f in fs]
            # print(f"Result: {result}")
            np.copyto(shm_np_array, np_array)
            print(f"Time elapsed for {num_computations} computations: {time.time()-sub_start_time:.2f}s")
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f'Time elapsed: {time.time()-start_time:.2f}s')
    tracemalloc.stop()

    # Without shared memory
    tracemalloc.start()
    start_time = time.time()
    for num_computations in num_computations_list:
        sub_start_time = time.time()
        with ProcessPoolExecutor(cpu_count()) as exe:
            fs = [exe.submit(work_no_shared_memory, np_array)
                for _ in range(num_computations)]
            for _ in as_completed(fs):
                pass
        # update shared memory
        np.copyto(shm_np_array, np_array)
        print(f"Time elapsed for {num_computations} computations: {time.time()-sub_start_time:.2f}s")
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f'Time elapsed: {time.time()-start_time:.2f}s')
    tracemalloc.stop()

    # Without multiprocessing at all
    tracemalloc.start()
    start_time = time.time()
    for num_computations in num_computations_list:
        sub_start_time = time.time()
        results = [work_no_shared_memory(np_array.copy()) for _ in range(num_computations)]
        print(f"Time elapsed for {num_computations} computations: {time.time()-sub_start_time:.2f}s")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f'Time elapsed: {time.time()-start_time:.2f}s')
    tracemalloc.stop()

    # multiprocessing with shared memory: 1.94 seconds
    # multiprocessing without shared memory: 4.01 seconds
    # single process: 0.37 seconds