import pygame, sys, random, time


def main_exit():
    print("exit game")
    exit(0)


def convert_to_list_input(x, length):
    if not isinstance(x, (tuple, list)):
        return [x for _ in range(length)]
    x = list(x)
    tail = x[-1]
    while len(x) < length:
        tail.append(tail)
    return x


class Block:
    def __init__(self, x, y):
        self.locx = x
        self.locy = y

    @staticmethod
    def rand(start, end):
        """random x = [start[0],end[0]), y = [start[1], end[1])"""
        start = convert_to_list_input(start, 2)
        end = convert_to_list_input(end, 2)

        x = random.randint(start[0], end[0])
        y = random.randint(start[1], end[1])
        return Block(x, y)

    def mov_left(self):
        self.locx -= 1

    def mov_right(self):
        self.locx += 1

    def mov_up(self):
        self.locy -= 1

    def mov_down(self):
        self.locy += 1

    def pos(self, cellsize=1):
        return self.locx * cellsize, self.locy * cellsize

    def setup(self, other):
        self.locx = other.locx
        self.locy = other.locy

    def copy(self):
        return Block(self.locx, self.locy)

    def bound_check(self, start, end):
        if start is None or end is None:
            return True

        start = convert_to_list_input(start, 2)
        end = convert_to_list_input(end, 2)
        if self.locx < start[0] or self.locx > end[0]:
            return False
        if self.locy < start[1] or self.locy > end[1]:
            return False
        return True

    def __eq__(self, other):
        return self.locx == other.locx and self.locy == other.locy

    def __str__(self):
        return f"Block {self.locx}, {self.locy}"


class Snake:
    class Direction:
        left = Block(-1, 0)
        right = Block(1, 0)
        up = Block(0, -1)
        down = Block(0, 1)

    def __init__(self, length, x, y):
        snake = [Block(x + i, y) for i in range(length)]
        self.snake = snake

    def grow(self):
        """length add 1"""
        tail = self.snake[-1].copy()
        tai2 = self.snake[-2]
        dx = tail.locx - tai2.locx
        dy = tail.locy - tai2.locy
        tail.locx += dx
        tail.locy += dy
        self.snake.append(tail)

    def move(self, direction: Block):
        head = self.snake[0]
        _ohead = head.copy()
        head.locx += direction.locx
        head.locy += direction.locy
        self._mov_body(_ohead)

    def move_left(self):
        self.move(self.Direction.left)

    def move_right(self):
        self.move(self.Direction.right)

    def move_up(self):
        self.move(self.Direction.up)

    def move_down(self):
        self.move(self.Direction.down)

    def _mov_body(self, _h):
        for b in self.body():
            tb = b.copy()
            b.setup(_h)
            _h = tb

    def __iter__(self):
        yield from self.snake

    def head(self):
        return self.snake[0]

    def body(self):
        itbdy = iter(self.snake)
        next(itbdy)
        try:
            while True:
                yield next(itbdy)
        except StopIteration:
            pass

    def direction(self):
        h1 = self.snake[0]
        h2 = self.snake[1]
        return Block(h1.locx - h2.locx, h1.locy - h2.locy)

    def has_block(self, block: Block, isbodyonly=False):
        itb = iter(self.snake)
        if isbodyonly:
            next(itb)
        try:
            while True:
                if block == next(itb):
                    return True
        except StopIteration:
            pass
        return False


pygame.init()
window_w = 640
window_h = 480
cell_size = 32
assert window_w % cell_size == 0 and window_h % cell_size == 0
Maze_W = window_w // cell_size - 2
Maze_H = window_h // cell_size - 2

screen = pygame.display.set_mode((window_w, window_h), 0, 32)
pygame.display.set_caption("snake game")
mainfont = pygame.font.SysFont("arial", 16)

color_line = (20, 200, 20)
color_head = (200, 20, 20)
color_body = (200, 200, 200)
color_apple = (200, 200, 20)

# pygame.event.post(
#     pygame.event.Event(pygame.KEYDOWN, {"key": K_LEFT, "mod": 0, "unicode": "<-"})
# )


def rand_apple(snake: Snake):
    while True:
        apple = Block.rand(1, (Maze_W, Maze_H))
        if not snake.has_block(apple):
            return apple


def init_snake_apple():
    sn = Snake(5, (Maze_W - 5) // 2, Maze_H - 2)
    apple = rand_apple(sn)
    return sn, apple


EVENT_MOVE = pygame.USEREVENT + 1
pygame.time.set_timer(EVENT_MOVE, 200)


def draw_board():
    for i in range(cell_size, window_w, cell_size):
        pygame.draw.line(screen, color_line, (i - 1, cell_size),
                         (i - 1, window_h - cell_size), 2)
    for i in range(cell_size, window_h, cell_size):
        pygame.draw.line(screen, color_line, (cell_size, i - 1),
                         (window_w - cell_size, i - 1), 2)


def draw_snake(snake: Snake):
    shead = snake.head()
    pygame.draw.rect(screen, color_head,
                     pygame.Rect(shead.pos(cell_size), (cell_size, cell_size)))
    for b in snake.body():
        pygame.draw.rect(screen, color_body,
                         pygame.Rect(b.pos(cell_size), (cell_size, cell_size)))


sn, apple = init_snake_apple()

while True:
    screen.fill((20, 20, 20))
    draw_board()
    draw_snake(sn)
    pygame.draw.rect(screen, color_apple,
                     pygame.Rect(apple.pos(cell_size), (cell_size, cell_size)))

    for event in pygame.event.get():
        moveDir = sn.direction()
        if event.type == pygame.QUIT:
            main_exit()
        if event.type == EVENT_MOVE:
            pass

        if event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_LEFT or key == pygame.K_a:
                moveDir = sn.Direction.left
            elif key == pygame.K_RIGHT or key == pygame.K_d:
                moveDir = sn.Direction.right
            elif key == pygame.K_UP or key == pygame.K_w:
                moveDir = sn.Direction.up
            elif key == pygame.K_DOWN or key == pygame.K_s:
                moveDir = sn.Direction.down
            elif key == pygame.K_q or key == pygame.K_ESCAPE:
                main_exit()
            elif key == pygame.K_g:
                sn.grow()
            hdir = sn.direction()
            cond1 = hdir.locx == -moveDir.locx and hdir.locy == moveDir.locy
            cond2 = hdir.locy == -moveDir.locy and hdir.locx == moveDir.locx
            if cond1 or cond2:
                continue

        sn.move(moveDir)

        if not sn.head().bound_check(1, (Maze_W, Maze_H)):
            print("meet wall")
            sn, apple = init_snake_apple()
            pygame.time.wait(500)
            break

        if sn.head() == apple:
            sn.grow()
            apple = rand_apple(sn)

        if sn.has_block(sn.head(), True):
            print("eat your self")
            pygame.time.wait(500)
            sn, apple = init_snake_apple()
            break

    pygame.display.update()
