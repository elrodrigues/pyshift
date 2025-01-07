import random
import math

class Trace:
    n_days: int
    steps_per_day: int
    high_intensity: float
    low_intensity: float
    intensities: list[float]

    def __init__(self, n_days, steps_per_day, high_intensity, low_intensity, intensities):
        self.n_days = n_days
        self.steps_per_day = steps_per_day
        self.high_intensity = high_intensity
        self.low_intensity = low_intensity
        self.intensities = intensities

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
    job_size: float
    # in Gigabits per Second
    first_hop_link_bandwidth: float
    # in absolute steps
    deadline: int
    # thread-throughput curve
    scale: float
    limit: float

    def thread_throughput_curve(self, thread: int) -> float:
        throughput: float = self.limit * (1 - (1 / (self.scale * self.limit * thread + 1)))
        return throughput


if __name__ == "__main__":
    print(create_trace(2))
