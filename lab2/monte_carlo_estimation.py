import numpy as np
import sobol_seq
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor
import time

# Параметры опциона и модели
S0 = 100        # начальная цена актива
T = 1           # конечное время 
mu = 0.05       # средний темп роста
sigma = 0.2     # волатильность
N = 100000      # количество моделируемых путей

K = 100         # цена страйка

# Функция для оценки цены опциона с использованием последовательности Соболя
def monte_carlo_option_price(num_paths, seed, european=False):
    # Генерация квазислучайных чисел Соболя
    sobol_points = sobol_seq.i4_sobol_generate(1, num_paths)
    norm_randoms = norm.ppf(sobol_points).flatten()  # Преобразование к нормальному распределению

    # Расчет конечных цен актива
    S_T = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * norm_randoms)

    # Вычисление цены опциона по формуле дисконтированной средней прибыли
    if european:
        payoff = np.maximum(S_T - K, 0)
        option_price = np.exp(-mu * T) * np.mean(payoff)
    else:
        option_price = np.mean(S_T)

    return option_price

# Распараллеливание процесса
def parallel_monte_carlo(N, num_workers):
    paths_per_worker = N // num_workers
    seeds = np.random.randint(1, 100000, size=num_workers)  # семена для разных потоков

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(monte_carlo_option_price, [paths_per_worker] * num_workers, seeds)

    return np.mean(list(results))

if __name__ == "__main__":
    # Оценка производительности и масштабируемости
    num_workers_list = [1, 2, 4, 8]
    for num_workers in num_workers_list:
        start_time = time.time()
        option_price = parallel_monte_carlo(N, num_workers)
        elapsed_time = time.time() - start_time
        print(f"Число потоков: {num_workers}, Цена опциона: {option_price:.4f}, Время выполнения: {elapsed_time:.2f} секунд")