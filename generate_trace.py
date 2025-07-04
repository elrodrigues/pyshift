from datetime import timezone
import random
import math
from numpy import ceil
from numpy.typing import NDArray

class SpaTempTrace:
    def __init__(
        self, n_days: int,
        steps_per_day: int,
        multi_intensities: list[list[list[float]]] = [],
    ):
        self.n_days: int = n_days
        self.steps_per_day: int = steps_per_day
        self.multi_intensities: list[list[float]] = multi_intensities

class Trace:
    def __init__(
        self, n_days: int,
        steps_per_day: int,
        high_intensity: float,
        low_intensity: float,
        intensities: list[float] = [],
        multi_intensities: list[list[float]] = [],
        timezones: list[int] = [],
    ):
        self.n_days: int = n_days
        self.steps_per_day: int = steps_per_day
        self.high_intensity: float = high_intensity
        self.low_intensity: float = low_intensity
        self.intensities: list[float] = intensities
        self.multi_intensities: list[list[float]] = multi_intensities
        self.timezones: list[int] = timezones

def create_trace(days: int, nodes: int, use_timezones: list[int] | None = None) -> Trace:
    # suppose we sample carbon intensity
    # every 15 minutes: 24 * 4 = 96
    steps_per_day = 24
    intensities = [[0. for _ in range(steps_per_day * days)] for _ in range(nodes)]

    start_boundary = 0
    end_boundary = steps_per_day

    high_intensity_floor = 300.0
    low_intensity_floor = 100.0

    high_intensity = high_intensity_floor + random.gauss(75.0, 25.0)
    high_intensity_dev = 0.05 * high_intensity

    # 4 timezones in continental US, so
    # set origin to West Coast, then increments of +0, +4, +8, +12
    timezones = [4 * random.randint(0, 3) for _ in range(nodes)]
    if use_timezones is not None:
        timezones = use_timezones

    consts = [20/96, 28/96]
    _s1 = math.floor(consts[0] * steps_per_day)
    _s2 = math.ceil(consts[1] * steps_per_day)

    for _ in range(days):
        low_intensity = low_intensity_floor + random.gauss(25.0, 5.0)
        diff = (high_intensity - low_intensity) / 2
        midpoint = (high_intensity + low_intensity) / 2

        for n in range(nodes):
            # 5 am to 7 am + time zone differences
            start_solar = random.randint(_s1, _s2) + start_boundary + timezones[n]
            # 7 pm to 5 pm
            end_solar = end_boundary - start_solar + start_boundary + timezones[n]
            time_diff = end_solar - start_solar

            sol = lambda t : diff * math.cos(2*math.pi*(t - start_solar) / time_diff) + midpoint

            for i in range(start_boundary, start_solar):
                intensities[n][i] = round(high_intensity + random.gauss(0.0, high_intensity_dev), 2)

            for i in range(end_solar, end_boundary):
                intensities[n][i] = round(high_intensity + random.gauss(0.0, high_intensity_dev), 2)

            # fill in active solar times
            for i in range(start_solar, end_solar):
                low_int = sol(i)
                intensities[n][i] = round(low_int + random.gauss(0.0, 0.1*low_int), 2)

        start_boundary += steps_per_day
        end_boundary += steps_per_day

    trace = Trace(
        n_days=days,
        steps_per_day=steps_per_day,
        high_intensity=high_intensity,
        low_intensity=low_intensity_floor,
        multi_intensities=intensities,
        timezones=timezones
    )
    return trace

class Environment:
    # in Gigabytes
    # job_size: float

    # in Gigabits per Second
    # first_hop_link_bandwidth: float

    # in absolute steps
    # deadline: int

    # thread-throughput curve (in Gbps)
    # thrpt_scale: float
    # thrpt_limit: float

    # thread-power curve in (in Watts)
    # power_scale: float
    # power_min_limit: float
    # power_max_limit: float

    def __init__(
        self, job_size: float,
        first_hop_link_band: float,
        deadline: int,
        thread_limit: int = 20,
    ):
        self.job_size: float = job_size
        self.first_hop_link_bandwidth: float = first_hop_link_band
        self.deadline: int = deadline

        self.thrpt_scale: float = 0.
        self.thrpt_limit: float = 0.
        self.power_scale: float = 0.
        self.power_min_limit: float = 0.
        self.power_max_limit: float = 0.
        self.thread_limit: int = thread_limit

    def set_throughput_curve(self, thrpt_scale: float, thrpt_limit: float):
        self.thrpt_scale = thrpt_scale
        self.thrpt_limit = thrpt_limit

    def set_power_curve(self, power_scale: float, power_min_limit: float, power_max_limit: float):
        self.power_scale = power_scale
        self.power_min_limit = power_min_limit
        self.power_max_limit = power_max_limit

    def thread_throughput_curve(self, thread: int | float) -> float:
        throughput: float = self.thrpt_limit * (1 - (1 / (self.thrpt_scale * self.thrpt_limit * thread + 1)))
        return throughput

    def throughput_thread_curve(self, thrpt, use_ceil=False, no_round=False):
        threads: float = thrpt / ((self.thrpt_limit * self.thrpt_scale) * (self.thrpt_limit - thrpt))
        if no_round:
            return threads
        if use_ceil:
            threads = ceil(threads)
        else:
            threads = round(threads, 0)
            if thrpt > 0. and threads == 0.:
                threads = 1
        return threads

    def marginal_thread(self, marginal_limit=0.1) -> tuple[int, float]:
        c = marginal_limit
        s = self.thrpt_scale
        L = self.thrpt_limit
        threads: float = math.floor((1 / math.sqrt(s * c)) - 1/(s*L))
        return (threads, self.thread_throughput_curve(threads))

    def thread_power_curve(self, thread: int | float) -> float:
        diff = self.power_max_limit - self.power_min_limit
        power: float = diff * (1 - (1 / (self.power_scale * diff * thread + 1))) + self.power_min_limit
        return power

if __name__ == "__main__":
    print(create_trace(2, 1))
