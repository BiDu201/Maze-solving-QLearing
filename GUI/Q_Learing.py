import pygame
import numpy as np

# Khởi tạo các biến và hằng số
WIDTH = 600  # Độ rộng của cửa sổ pygame
HEIGHT = 600  # Chiều cao của cửa sổ pygame
CELL_SIZE = 60  # Kích thước của mỗi ô vuông trong mê cung
MAZE_WIDTH = WIDTH // CELL_SIZE  # Số ô vuông theo chiều ngang
MAZE_HEIGHT = HEIGHT // CELL_SIZE  # Số ô vuông theo chiều dọc

maze = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
])

start_pos = (0, 6)  # Vị trí bắt đầu
end_pos = (9, 4)  # Vị trí kết thúc

q_table = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, 4))  # Bảng Q

actions = {
    0: (-1, 0),  # Di chuyển lên
    1: (1, 0),  # Di chuyển xuống
    2: (0, -1),  # Di chuyển sang trái
    3: (0, 1)  # Di chuyển sang phải
}

# Khởi tạo cửa sổ pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Maze Solver")


def draw_maze():
    for i in range(MAZE_HEIGHT):
        for j in range(MAZE_WIDTH):
            if maze[i, j] == 0:
                pygame.draw.rect(screen, (0, 0, 0), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, (255, 255, 255), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_character(position):
    x = position[1] * CELL_SIZE
    y = position[0] * CELL_SIZE
    pygame.draw.circle(screen, (255, 0, 0), (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2 - 2)


def draw_end_point():
    x = end_pos[1] * CELL_SIZE
    y = end_pos[0] * CELL_SIZE
    pygame.draw.circle(screen, (0, 255, 0), (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2 - 2)

def get_valid_actions(position):
    valid_actions = []
    for action, (dx, dy) in actions.items():
        new_x = position[0] + dx
        new_y = position[1] + dy
        if 0 <= new_x < MAZE_HEIGHT and 0 <= new_y < MAZE_WIDTH and maze[new_x, new_y] == 1:
            valid_actions.append(action)
    return valid_actions


def update_q_table(position, action, next_position, reward):
    alpha = 0.1 # Tốc độ học
    gamma = 0.9 # Hệ số giảm
    q_table[position[0], position[1], action] = (1 - alpha) * q_table[position[0], position[1], action] + alpha * (reward + gamma * np.max(q_table[next_position]))


def choose_action(position, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.choice(get_valid_actions(position))
    else:
        return np.argmax(q_table[position[0], position[1]])

def move(position, action):
    dx, dy = actions[action]
    new_x = position[0] + dx
    new_y = position[1] + dy
    if 0 <= new_x < MAZE_HEIGHT and 0 <= new_y < MAZE_WIDTH and maze[new_x, new_y] == 1:
        return new_x, new_y
    return position

def run_q_learning():
    episodes = 100 # Số lượng tập huấn luyện
    max_steps = 100 # Số lượng bước tối đa trong mỗi tập
    epsilon = 0.9  # Tham số epsilon trong thuật toán ε-greedy

    for episode in range(episodes):
        position = start_pos
        for step in range(max_steps):
            action = choose_action(position, epsilon)
            next_position = move(position, action)
            reward = 1 if next_position == end_pos else 0

            update_q_table(position, action, next_position, reward)

            position = next_position

            if position == end_pos:
                break

# Chạy thuật toán Q-learning
run_q_learning()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Vẽ mê cung
    draw_maze()

    # Vẽ nhân vật
    draw_character(start_pos)

    # Vẽ điểm kết thúc
    draw_end_point()

    # Cập nhật cửa sổ pygame
    pygame.display.update()

    # Tốc độ chạy
    clock = pygame.time.Clock()
    clock.tick(10)
    # Lựa chọn hành động dựa trên bảng Q
    action = np.argmax(q_table[start_pos[0], start_pos[1]])

    # Di chuyển nhân vật
    next_position = move(start_pos, action)
    start_pos = next_position

    # Kiểm tra điều kiện kết thúc
    if start_pos == end_pos:
        print("Đã đến đích")
        running = False

pygame.quit()