from numpy._typing import NDArray
from scipy.optimize import linprog
from math import ceil
import generate_trace as gen
import numpy as np

def _produce_plan(bytes_left, cur_step, step_time, trace_sum: NDArray, env: gen.Environment) -> NDArray:
    threads = np.zeros(trace_sum.shape[0])
    sorted_steps = trace_sum.argsort()
    deadline = env.deadline

    th, thrpt_target = env.marginal_thread(marginal_limit=0.05)
    print("DEBUG: new target thrpt", thrpt_target)

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

def _estimate_carbon(plan, step_time, trace_sum, env: gen.Environment) -> tuple[float, float]:
    steps = len(plan)

    carbon_emitted = 0.0
    bytes_left = env.job_size
    for s in range(steps):
        thr = plan[s]
        if bytes_left <= 0.:
            break

        if thr > 0.:
            thrpt = env.thread_throughput_curve(thr)
            pow = env.thread_power_curve(int(thr))

            time = (bytes_left * 8) / thrpt
            if step_time < time:
                time = step_time

            carbon_emitted += pow * (trace_sum[s] / 3600000) * time
            bytes_transferred = (thrpt * time) / 8
            bytes_left -= bytes_transferred

    return (carbon_emitted, bytes_left)

def _debug_evaluate_thrpt_plan(plan, step_time):
    bytes_transferred = 0.
    for thrpt in plan:
        bytes_transferred += thrpt * step_time
    return bytes_transferred / 8.

def execute_one_job_multiple_traces_heur():
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
    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s < deadline:
            # try to fill in marginal threads
            # th, thrpt, pow = env.marginal_thread()
            # threads[s] = float(th)
            thr = env.throughput_thread_curve(target_thrpt, use_ceil=True)
            thrpt = env.thread_throughput_curve(thr)
            threads[s] = thr
            time = (bytes_left * 8) / target_thrpt
            if step_time < time:
                time = step_time
            bytes_transferred = (thrpt * time) / 8
            bytes_left -= bytes_transferred

    print("Forecast")
    print(intensities)
    print("Timezones:", forecast.timezones)
    print("Worst-Case Plan")
    print(threads)
    carbon_emitted, bytes_left = _estimate_carbon(threads, step_time, intensity_sums, env)
    print("Worst-Case gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")
    # print("DEBUG: _estimate_carbon", carbon_emitted, bytes_left)

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

def execute_one_job_multiple_traces_lp():
    job_size = 10000.0 # Gigabytes
    first_hop_band = 10.0 # Gigabit per Sec
    deadline = 140 # Steps from origin
    n_days = 2 # Horizon
    n_nodes = 3 # src - connector - dst
    thrpt_scale = 1/21
    power_min_limit = 125.0
    power_max_limit = 300.0
    power_scale = 1/180
    thread_limit = 20

    thrpt_limit = 0.35 # 40% of bandwidth

    env = gen.Environment(
        job_size=job_size,
        first_hop_link_band=first_hop_band,
        deadline=deadline,
        thread_limit=thread_limit
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
    intensity_sums = (weights * intensities).sum(axis=0)
    sorted_steps = intensity_sums.argsort()

    sorted_steps = intensity_sums.argsort()

    # Fill-in
    threads = np.zeros(n_steps)
    bytes_left = job_size
    target_thrpt = max_throughput * thrpt_limit
    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s < deadline:
            # try to fill in marginal threads
            # th, thrpt, pow = env.marginal_thread()
            # threads[s] = float(th)
            thr = env.throughput_thread_curve(target_thrpt, use_ceil=False)
            thrpt = env.thread_throughput_curve(thr)
            threads[s] = thr
            time = (bytes_left * 8) / thrpt
            if step_time < time:
                time = step_time
            bytes_transferred = (thrpt * time) / 8
            bytes_left -= bytes_transferred

    print("Timezones:", forecast.timezones)
    print("Worst-Case Heuristic Plan")
    print(threads.astype(int))
    carbon_emitted, bytes_left = _estimate_carbon(threads, step_time, intensity_sums, env)
    print("Worst-Case gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")

    # Build Linear Program
    intensity_vector = intensity_sums[:deadline].reshape(1, deadline)
    target_thrpt = max_throughput * thrpt_limit
    time_vector = (-step_time * np.ones(deadline)).reshape(1, deadline)
    byte_vector = np.array(-8 * job_size)

    res = linprog(
        c=intensity_vector,
        A_ub=time_vector,
        b_ub=byte_vector,
        bounds=(0, target_thrpt),
    )
    thrpt_plan = np.append(res.x, np.zeros(n_steps - deadline))
    threads = np.array([env.throughput_thread_curve(t, use_ceil=False) for t in thrpt_plan]).astype(int)

    print("LP Forecast")
    print(threads)
    carbon_emitted, bytes_left = _estimate_carbon(threads, step_time, intensity_sums, env)
    print("Worst-Case gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")
    print("DEBUG: thrpt", env.thread_throughput_curve(1))

    # RUNTIME
    env.set_throughput_curve(thrpt_scale, 0.7 * max_throughput)

    # Heuristic
    threads = np.zeros(n_steps)
    bytes_left = job_size
    target_thrpt = max_throughput * 0.59
    for s in sorted_steps:
        if bytes_left <= 0:
            break
        if s < deadline:
            # try to fill in marginal threads
            # th, thrpt, pow = env.marginal_thread()
            # threads[s] = float(th)
            thr = env.throughput_thread_curve(target_thrpt, use_ceil=False)
            thrpt = env.thread_throughput_curve(thr)
            threads[s] = thr
            time = (bytes_left * 8) / thrpt
            if step_time < time:
                time = step_time
            bytes_transferred = (thrpt * time) / 8
            bytes_left -= bytes_transferred

    print("Timezones:", forecast.timezones)
    print("Heuristic Plan")
    print(threads.astype(int))
    carbon_emitted, bytes_left = _estimate_carbon(threads, step_time, intensity_sums, env)
    print("gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")

    # LP
    res = linprog(
        c=intensity_vector,
        A_ub=time_vector,
        b_ub=byte_vector,
        bounds=(0, max_throughput * 0.59),
    )
    thrpt_plan = np.append(res.x, np.zeros(n_steps - deadline))
    threads = np.array([env.throughput_thread_curve(t, use_ceil=False) for t in thrpt_plan]).astype(int)

    print("LP Forecast")
    print(threads)
    carbon_emitted, bytes_left = _estimate_carbon(threads, step_time, intensity_sums, env)
    print("gCO2:", round(carbon_emitted, 2), "g")
    print("Bytes left:", bytes_left, "GB")

def execute_two_jobs_three_nodes_lp():
    job_sizes = [10000.0, 5600.0, 7200.0, 2400.0, 6000.0] # Gigabytes
    first_hop_band = 10.0 # Gigabit per Sec
    deadlines = [160, 130, 120, 70, 100] # Steps from origin
    n_days = 2 # Horizon
    n_nodes = 3 # src - connector - dst
    n_jobs = len(job_sizes)
    thrpt_scale = 1/21
    power_min_limit = 125.0
    power_max_limit = 300.0
    power_scale = 1/180
    thread_limit = 20

    thrpt_limit = 0.35 # 40% of bandwidth

    envs = [
        gen.Environment(
            job_size=job_sizes[i],
            first_hop_link_band=first_hop_band,
            deadline=deadlines[i],
            thread_limit=thread_limit
        )
        for i in range(n_jobs)
    ]

    tzs = [
        [0, 4, 8],
        [0, 0, 4],
        [0, 8, 8],
        [0, 8, 12],
        [0, 8, 4]
    ]

    max_throughput = first_hop_band
    for env in envs:
        env.set_throughput_curve(thrpt_scale, max_throughput)
        env.set_power_curve(power_scale, power_min_limit, power_max_limit)

    forecasts = [
        gen.create_trace(n_days, n_nodes, use_timezones=tzs[i])
        for i in range(n_jobs)
    ]
    for forecast in forecasts[1:]:
        forecast.multi_intensities[0] = forecasts[0].multi_intensities[0]

    n_steps = n_days * forecasts[0].steps_per_day
    step_time = (24 / forecasts[0].steps_per_day) * 3600 # seconds

    weights = np.ones(n_nodes).reshape(n_nodes, 1)
    forecast_sums = [
        (weights * forecast.multi_intensities).sum(axis=0)
        for forecast in forecasts
    ]

    # LP constraints
    deadline_sum = np.sum(deadlines)
    intensity_vector = np.array([])
    for i in range(n_jobs):
        intensity_vector = np.append(intensity_vector, forecast_sums[i][:deadlines[i]])
    intensity_vector = intensity_vector.reshape(1, deadline_sum)

    target_thrpt = max_throughput * thrpt_limit
    # A_ub - order matters
    A_ub = np.array([])
    offset = 0
    for i in range(n_jobs):
        byte_sum_vec = np.zeros(deadline_sum)
        end_index = offset + deadlines[i]
        byte_sum_vec[offset:end_index] = -step_time * np.ones(deadlines[i])
        # byte_sum_vec = byte_sum_vec.reshape(1, deadline_sum)
        A_ub = np.append(A_ub, byte_sum_vec)
        offset += deadlines[i]

    max_deadline = max(deadlines)
    for i in range(max_deadline):
        v = np.zeros(deadline_sum)# .reshape(1, deadline_sum)

        offset = 0
        for d in deadlines:
            if i < d:
                v[offset + i] = 1.
            offset += d

        A_ub = np.append(A_ub, v)

    A_ub = A_ub.reshape(max_deadline + n_jobs, deadline_sum)

    # b_ub - same order
    b_ub = byte_vector = -8 * np.array(job_sizes)
    b_ub = np.append(b_ub, target_thrpt * np.ones(max_deadline))

    res = linprog(
        c=intensity_vector,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, target_thrpt),
    )

    # merge plan
    thrpt_plan = np.zeros(n_steps * n_jobs).reshape(n_steps, n_jobs)
    thrpt_unprocessed = res.x
    for i in range(max_deadline):
        offset = 0
        for j in range(n_jobs):
            if i < deadlines[j]:
                thrpt_plan[i, j] = thrpt_unprocessed[offset + i]
            offset += deadlines[j]





if __name__ == "__main__":
    execute_two_jobs_three_nodes_lp()
