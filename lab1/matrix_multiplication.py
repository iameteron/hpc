from mpi4py import MPI
import numpy as np

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Ранг текущего процесса
size = comm.Get_size()   # Общее количество процессов

# Определение размеров матрицы и вектора
n = 1000  # Количество строк в матрице (можно задать другое значение)
m = 1000  # Количество столбцов в матрице (должно совпадать с размером вектора)

# Главный процесс (с rank=0) инициализирует матрицу и вектор
if rank == 0:
    # Создаем случайную матрицу и вектор
    matrix = np.random.rand(n, m)
    vector = np.random.rand(m)
else:
    # Инициализация пустых значений для остальных процессов
    matrix = None
    vector = None

# Рассылаем вектор всем процессам
vector = comm.bcast(vector, root=0)

# Вычисляем количество строк на каждый процесс
rows_per_process = n // size
extra_rows = n % size

# Определяем количество строк для каждого процесса
if rank < extra_rows:
    local_rows = rows_per_process + 1
    start_row = rank * local_rows
else:
    local_rows = rows_per_process
    start_row = rank * local_rows + extra_rows

end_row = start_row + local_rows

# Каждому процессу выделяем часть матрицы, за которую он отвечает
if rank == 0:
    # Главный процесс отправляет подматрицы
    for i in range(1, size):
        if i < extra_rows:
            local_rows = rows_per_process + 1
            start_row = i * local_rows
        else:
            local_rows = rows_per_process
            start_row = i * local_rows + extra_rows
        end_row = start_row + local_rows
        comm.send(matrix[start_row:end_row, :], dest=i)
    local_matrix = matrix[0:end_row, :]
else:
    # Остальные процессы получают подматрицы
    local_matrix = comm.recv(source=0)

# Умножаем подматрицу на вектор
local_result = np.dot(local_matrix, vector)

# Собираем результаты от всех процессов
result = None
if rank == 0:
    # Главный процесс собирает результаты
    result = np.zeros(n)
    result[0:len(local_result)] = local_result
    for i in range(1, size):
        part_result = comm.recv(source=i)
        start_row = i * rows_per_process + min(i, extra_rows)
        result[start_row:start_row + len(part_result)] = part_result
else:
    # Остальные процессы отправляют свои результаты
    comm.send(local_result, dest=0)

# Печать результата на главном процессе
if rank == 0:
    print("Результат умножения матрицы на вектор:")
    print(result)
