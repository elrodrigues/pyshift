import random
import math

class Trace:
    def __init__(
        self, n_days: int,
        steps_per_day: int,
        high_intensity: float,
        low_intensity: float,
        intensities: list[float]
    ):
        self.n_days: int = n_days
        self.steps_per_day: int = steps_per_day
        self.high_intensity: float = high_intensity
        self.low_intensity: float = low_intensity
        self.intensities: list[float] = intensities

def create_trace(days: int) -> Trace:
    # suppose we sample carbon intensity
    # every 15 minutes: 24 * 4 = 96
    intensities = [0. for _ in range(96 * days)]

    start_boundary = 0
    end_boundary = 96

    high_intensity_floor = 300.0
    low_intensity_floor = 100.0

    high_intensity = high_intensity_floor + random.gauss(75.0, 25.0)
    high_intensity_dev = 0.05 * high_intensity

    for _ in range(days):
        low_intensity = low_intensity_floor + random.gauss(25.0, 5.0)
        diff = (high_intensity - low_intensity) / 2
        midpoint = (high_intensity + low_intensity) / 2

        # 5 am to 7 am
        start_solar = random.randint(20, 28) + start_boundary
        # 7 pm to 5 pm
        end_solar = end_boundary - start_solar + start_boundary
        time_diff = end_solar - start_solar

        sol = lambda t : diff * math.cos(2*math.pi*(t - start_solar) / time_diff) + midpoint

        for i in range(start_boundary, start_solar):
            intensities[i] = round(high_intensity + random.gauss(0.0, high_intensity_dev), 2)

        for i in range(end_solar, end_boundary):
            intensities[i] = round(high_intensity + random.gauss(0.0, high_intensity_dev), 2)

        # fill in active solar times
        for i in range(start_solar, end_solar):
            low_int = sol(i)
            intensities[i] = round(low_int + random.gauss(0.0, 0.1*low_int), 2)

        start_boundary += 96
        end_boundary += 96

    trace = Trace(
        n_days=days,
        steps_per_day=96,
        high_intensity=high_intensity,
        low_intensity=low_intensity_floor,
        intensities=intensities
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
        deadline: int
    ):
        self.job_size: float = job_size
        self.first_hop_link_bandwidth: float = first_hop_link_band
        self.deadline: int = deadline

        self.thrpt_scale: float = 0.
        self.thrpt_limit: float = 0.
        self.power_scale: float = 0.
        self.power_min_limit: float = 0.
        self.power_max_limit: float = 0.

    def set_throughput_curve(self, thrpt_scale: float, thrpt_limit: float):
        self.thrpt_scale = thrpt_scale
        self.thrpt_limit = thrpt_limit

    def set_power_curve(self, power_scale: float, power_min_limit: float, power_max_limit: float):
        self.power_scale = power_scale
        self.power_min_limit = power_min_limit
        self.power_max_limit = power_max_limit

    def thread_throughput_curve(self, thread: int) -> float:
        throughput: float = self.thrpt_limit * (1 - (1 / (self.thrpt_scale * self.thrpt_limit * thread + 1)))
        return throughput

    def throughput_thread_curve(self, thrpt: float) -> float:
        threads: float = round(1/(self.thrpt_scale * (self.thrpt_limit - thrpt)) - 1/(self.thrpt_scale * self.thrpt_limit), 0)
        return threads

    def marginal_thread(self) -> tuple[int, float, float]:
        threads: float = math.floor(math.sqrt(self.thrpt_scale) - 1/self.thrpt_limit)
        return (threads, self.thread_throughput_curve(threads), self.thread_power_curve(threads))

    def thread_power_curve(self, thread: int) -> float:
        diff = self.power_max_limit - self.power_min_limit
        power: float = diff * (1 - (1 / (self.power_scale * diff * thread + 1))) + self.power_min_limit
        return power

if __name__ == "__main__":
    print(create_trace(2))
