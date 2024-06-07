from dask import delayed, compute, visualize
import time
from dask.distributed import LocalCluster, Client

if __name__ == '__main__':
    cluster = LocalCluster(resources={'CPU': 1})
    # client = Client()
    # client.get_versions(check=True)

    def inc(x):
        time.sleep(1)
        return x + 1

    def double(x):
        time.sleep(1)
        return x * 2

    def add(x, y):
        time.sleep(1)
        return x + y

    data = [1, 2, 3, 4, 5]

    output = []

    for i in data:
        a = delayed(inc)(i)
        b = delayed(double)(i)
        c = delayed(add)(a, b)
        output.append(c)

    start_time = time.time()
    final_result = compute(*output)
    end_time = time.time()

    print(final_result)
    print(f"Time taken: {end_time - start_time} seconds")