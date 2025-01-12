import generate_trace as gen
import numpy as np

def plan_one_job_multiple_traces():
    job_size = 9600.0 # Gigabytes
    first_hop_band = 10.0 # Gigabit per Sec
    deadline = 140 # Steps from origin
    n_days = 2 # Horizon
    n_hops = 3 # src - connector - dst
    thrpt_scale = 1/21
    power_min_limit = 125.0
    power_max_limit = 300.0
    power_scale = 1/180

    bandwidth_limit = 0.4 # 40% of bandwidth

    env = gen.Environment(
        job_size=job_size,
        first_hop_link_band=first_hop_band,
        deadline=deadline
    )

    # best guess estimate
    max_throughput = first_hop_band
    env.set_throughput_curve(thrpt_scale, max_throughput)
    env.set_power_curve(power_scale, power_min_limit, power_max_limit)

    trace = gen.create_trace(n_days, n_hops)
    n_steps = n_days * trace.steps_per_day
    step_time = (24 / trace.steps_per_day) * 3600 # seconds

    intensities = np.array(trace.multi_intensities)
    weights = np.ones(n_hops).reshape(n_hops, 1)
    # augmented_trace = np.vstack((np.arange(n_steps), intensities)).T
    intensity_sums = (weights * intensities).sum(axis=0)
    sorted_steps = intensity_sums.argsort()

    # Fill-in
    threads = np.zeros(n_steps)
    bytes_left = job_size
    target = max_throughput * bandwidth_limit
    carbon_emitted = 0.0
    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s < deadline:
            # try to fill in marginal threads
            # th, thrpt, pow = env.marginal_thread()
            # threads[s] = float(th)
            th = env.throughput_thread_curve(target)
            threads[s] = th
            time = (bytes_left * 8) / target
            if step_time < time:
                time = step_time
            bytes_transferred = (target * time) / 8
            bytes_left -= bytes_transferred
            carbon_emitted += env.thread_power_curve(int(th)) * (intensity_sums[s] / 3600000) * time

    print("Trace")
    print(intensities)
    print("Timezones:", trace.timezones)
    print("Plan")
    print(threads)
    print("gCO2:", round(carbon_emitted, 2), "g")
    print("bytes left:", bytes_left, "GB")

if __name__ == "__main__":
    plan_one_job_multiple_traces()
