from dask import delayed, compute, visualize
import time

@delayed
def dummy_task(x):
    import time
    time.sleep(10)  # Simulate some work
    return x + 1

# Create a list of independent tasks
tasks = [dummy_task(i) for i in range(4)]

start = time.time()
# Compute tasks in parallel
results = compute(*tasks)
end = time.time()

print(f"Time taken: {end - start}")

visualize(*tasks)

print(results)