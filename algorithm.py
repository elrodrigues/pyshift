from numpy._typing import NDArray
import generate_trace as gen
import numpy as np

def plan_one_job_multiple_traces():
    job_size = 9600.0 # Gigabytes
    first_hop_band = 10.0 # Gigabit per Sec
    deadline = 140 # Steps from origin
    n_days = 2 # Horizon
    n_nodes = 3 # src - connector - dst
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

    trace = gen.create_trace(n_days, n_nodes)
    n_steps = n_days * trace.steps_per_day
    step_time = (24 / trace.steps_per_day) * 3600 # seconds

    intensities = np.array(trace.multi_intensities)
    # give equal weight to all nodes
    weights = np.ones(n_nodes).reshape(n_nodes, 1)
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

def _produce_plan(bytes_left, cur_step, step_time, trace_sum: NDArray, env: gen.Environment) -> NDArray:
    threads = np.zeros(trace_sum.shape[0])
    sorted_steps = trace_sum.argsort()
    deadline = env.deadline

    th, thrpt_target = env.marginal_thread()

    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s <= cur_step:
            continue
        if s < deadline:
            threads[s] = th
            time = (bytes_left * 8) / thrpt_target
            if step_time < time:
                time = step_time
            bytes_transferred = (thrpt_target * time) / 8
            bytes_left -= bytes_transferred

    return threads

def execute_one_job_multiple_traces():
    job_size = 9600.0 # Gigabytes
    first_hop_band = 10.0 # Gigabit per Sec
    deadline = 140 # Steps from origin
    n_days = 2 # Horizon
    n_nodes = 3 # src - connector - dst
    thrpt_scale = 1/21
    power_min_limit = 125.0
    power_max_limit = 300.0
    power_scale = 1/180

    thrpt_limit = 0.4 # 40% of bandwidth

    env = gen.Environment(
        job_size=job_size,
        first_hop_link_band=first_hop_band,
        deadline=deadline
    )

    # best guess estimate
    max_throughput = first_hop_band
    env.set_throughput_curve(thrpt_scale, max_throughput)
    env.set_power_curve(power_scale, power_min_limit, power_max_limit)

    forecast = gen.create_trace(n_days, n_nodes)
    n_steps = n_days * forecast.steps_per_day
    step_time = (24 / forecast.steps_per_day) * 3600 # seconds

    intensities = np.array(forecast.multi_intensities)
    # give equal weight to all nodes
    weights = np.ones(n_nodes).reshape(n_nodes, 1)
    # augmented_trace = np.vstack((np.arange(n_steps), intensities)).T
    intensity_sums = (weights * intensities).sum(axis=0)
    sorted_steps = intensity_sums.argsort()

    # Fill-in
    threads = np.zeros(n_steps)
    bytes_left = job_size
    target_thrpt = max_throughput * thrpt_limit
    carbon_emitted = 0.0
    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s < deadline:
            # try to fill in marginal threads
            # th, thrpt, pow = env.marginal_thread()
            # threads[s] = float(th)
            thr = env.throughput_thread_curve(target_thrpt)
            threads[s] = thr
            time = (bytes_left * 8) / target_thrpt
            if step_time < time:
                time = step_time
            bytes_transferred = (target_thrpt * time) / 8
            bytes_left -= bytes_transferred
            carbon_emitted += env.thread_power_curve(int(thr)) * (intensity_sums[s] / 3600000) * time

    print("Forecast")
    print(intensities)
    print("Timezones:", forecast.timezones)
    print("Worst-Case Plan")
    print(threads)
    print("Worst-Case gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")

    # Generate actual intensity trace
    trace = gen.create_trace(n_days, n_nodes, use_timezones=forecast.timezones)
    real_intensities = np.array(trace.multi_intensities)
    real_intensity_sums = (weights * real_intensities).sum(axis=0)
    start_step = end_step = 0
    for s in range(n_steps):
        if threads[s] != 0.:
            start_step = s
            break
    for s in range(n_steps-1, -1, -1):
        if threads[s] != 0.:
            end_step = s
            break

    # SIM STATE MACHINE
    # Set-up for run
    real_env = gen.Environment(
        job_size=job_size,
        first_hop_link_band=first_hop_band,
        deadline=deadline
    )
    effective_bandwidth = 0.7 * max_throughput
    real_env.set_throughput_curve(thrpt_scale, effective_bandwidth)

    thread_limit = 20 # 20 available threads
    psteps_per_step = 4
    pstep_time = step_time / psteps_per_step

    carbon_emitted = 0.0
    bytes_left = job_size

    discovery_started = end_sim = False
    plan_adjusted = True
    s = start_step
    while s < deadline:
        thr = threads[s]
        thrpt = env.thread_throughput_curve(thr)
        for _ in range(psteps_per_step):
            if bytes_left <= 0:
                end_sim = True
                break

            # 1. Move data
            time = (bytes_left * 8) / thrpt
            if pstep_time < time:
                time = pstep_time
            bytes_transferred = (thrpt * time) / 8
            bytes_left -= bytes_transferred
            carbon_emitted += env.thread_power_curve(thr) * (real_intensity_sums[s] / 3600000) * time

            if not plan_adjusted:
                # 3. Adjust
                observed_thrpt = effective_bandwidth
                env.set_throughput_curve(thrpt_scale, observed_thrpt)
                threads = _produce_plan(
                    bytes_left=bytes_left,
                    cur_step=s,
                    step_time=step_time,
                    trace_sum=real_intensity_sums,
                    env=env
                )
                plan_adjusted = True
                fresh_state = False

            if not discovery_started:
                # 2. Discovery
                th = thread_limit
                discovery_started = True
                plan_adjusted = False

        if end_sim:
            break

        for s_n in range(s+1, n_steps):
            if threads[s_n] != 0.:
                s = s_n
                break

    print("Simulated Trace")
    print(real_intensities)
    print("Timezones:", trace.timezones)
    print("Executed Plan")
    print(threads)
    print("Deadline (steps)", deadline)
    print("Expected end-time (steps)", end_step)
    print("Actual end-time (steps)", s)
    print("Simulated gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")


if __name__ == "__main__":
    execute_one_job_multiple_traces()
