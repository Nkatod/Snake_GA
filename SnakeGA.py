import pygame, sys, random, time
import gym
from gym import spaces
import numpy as np

'''
    Функция получения выжившей популяции
        Входные параметры:
        - popul - наша популяция
        - val - текущие значения
        - nsurv - количество выживших
        - reverse - указываем требуемую операцию поиска результата: максимизация или минимизация
'''
game_heigh = 15
def getSurvPopul(
        popul,
        val,
        nsurv,
        reverse
):
    newpopul = []  # Двумерный массив для новой популяции

    for i in range(len(val)):
        if val[i] != 0:
            val[i] = val[i] + np.random.random() * 0.001

    sval = sorted(val, reverse=reverse)  # Сортируем зачения в val в зависимости от параметра reverse

    for i in range(nsurv):  # Проходимся по циклу nsurv-раз (в итоге в newpopul запишется nsurv-лучших показателей)
        if sval[i] == 0:
            continue
        index = val.index(sval[i])  # Получаем индекс i-того элемента sval в исходном массиве val
        newpopul.append(popul[index])  # В новую папуляцию добавляем элемент из текущей популяции с найденным индексом
    while len(newpopul) < nsurv:
        bot = np.random.random((game_heigh, 3))
        newpopul.append(bot)

    return newpopul, sval  # Возвращаем новую популяцию (из nsurv элементов) и сортированный список


'''
    Функция получения родителей
        Входные параметры:
        - curr_popul - текущая популяция
        - nsurv - количество выживших
'''


def getParents(
        curr_popul,
        nsurv
):
    indexp1 = random.randint(0, nsurv - 1)  # Случайный индекс первого родителя в диапазоне от 0 до nsurv - 1
    indexp2 = random.randint(0, nsurv - 1)  # Случайный индекс второго родителя в диапазоне от 0 до nsurv - 1
    botp1 = curr_popul[indexp1]  # Получаем первого бота-родителя по indexp1
    botp2 = curr_popul[indexp2]  # Получаем второго бота-родителя по indexp2

    return botp1, botp2  # Возвращаем обоих полученных ботов


'''
    Функция смешивания (кроссинговера) двух родителей
        Входные параметры:
        - botp1 - первый бот-родитель
        - botp2 - второй бот-родитель
        - j - номер компонента бота
'''


def crossPointFrom2Parents(
        botp1,
        botp2,
        j
):
    pindex = np.random.random()  # Получаем случайное число в диапазоне от 0 до 1

    # Если pindex меньше 0.5, то берем значения от первого бота, иначе от второго
    if pindex < 0.5:
        x = botp1[j]
    else:
        x = botp2[j]
    return x  # Возвращаем значние бота


class Snake(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width=50, thickness=10, bot=None):
        super(Snake, self).__init__()
        # necessary internal gym things
        self.id = "Snake"
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.mode = "auto"
        self.current_step = 0

        # make it square for the sake of defining the state space
        self.height = width
        self.possible_actions = ['LEFT', 'FORWARD', 'RIGHT']
        # Define action and observation spaces following openai standard
        self.action_space = spaces.Discrete(len(self.possible_actions))
        # Define other internal variables
        self.width = width
        self.thickness = thickness
        self.gameMatrix = np.zeros((self.width, self.width))
        self.agent = bot
        # np.zeros((self.width, len(self.possible_actions)))
        # whether it should render anything
        # self.display = display
        # Important varibles
        self.reset()

        # Colors
        self.red = pygame.Color(255, 0, 0)  # gameover
        self.green = pygame.Color(0, 255, 0)  # snake
        self.black = pygame.Color(0, 0, 0)  # score
        self.white = pygame.Color(255, 255, 255)  # background
        self.brown = pygame.Color(165, 42, 42)  # food
        self.yellow = pygame.Color(255, 255, 0)
        self.blue = pygame.Color(0, 0, 255)

    def close_render(self):
        self.pygame.quit()

    def get_states(self):
        ''' Gets the relevant states of the game used in the AI
        '''
        state_dict = {
            'snake_pos_x': self.snakePos[1],
            'snake_pos_y': self.snakePos[0],
            'food_pos_x': self.foodPos[1],
            'food_pos_y': self.foodPos[0],
            #             'snake_body': self.snakeBody,
            # 'is_game_over': 1 if self.game_over else 0]],
            'border_distances_right': abs((self.width - 1) - self.snakePos[0]),
            'border_distances_left': self.snakePos[0],
            'border_distances_up': abs((self.height - 1) - self.snakePos[1]),
            'border_distances_down': self.snakePos[1]
        }
        return state_dict

    def render_game_over(self):
        if self.game_over:
            myFont = self.pygame.font.SysFont('monaco', self.height)
            GOsurf = myFont.render('Game over!', True, self.red)
            GOrect = GOsurf.get_rect()
            GOrect.midtop = ((self.thickness * self.width / 2), 1.5 * self.thickness)
            self.playSurface.blit(GOsurf, GOrect)
            self.render_score(0)
            self.pygame.display.flip()
            time.sleep(4)
            self.close_render()

    def render_score(self, choice=1):
        sFont = self.pygame.font.SysFont('monaco', 24)
        Ssurf = sFont.render('Score : {0}'.format(self.score), True, self.black)
        Srect = Ssurf.get_rect()
        if choice == 1:
            Srect.midtop = (8 * self.thickness, self.thickness)
        else:
            Srect.midtop = (self.thickness * (self.width / 2), 12 * self.thickness)
        self.playSurface.blit(Ssurf, Srect)

    def get_pressed_key(self):
        ''' Gets the key that was pressed by the user
        '''
        change_to = 'FORWARD'  # self.direction
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.game_over = True
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_RIGHT or event.key == ord('d'):
                    change_to = 'RIGHT'
                if event.key == self.pygame.K_LEFT or event.key == ord('a'):
                    change_to = 'LEFT'
                if event.key == self.pygame.K_ESCAPE:
                    self.pygame.event.post(self.pygame.event.Event(self.pygame.QUIT))
        return change_to

    def move_snake(self, change_to):
        ''' Move the snake
        '''
        self.spawn_food()
        wasChanged = False
        if not wasChanged and change_to == 'RIGHT' and self.direction == 'UP':
            self.direction = 'RIGHT'
            wasChanged = True
        if not wasChanged and change_to == 'RIGHT' and self.direction == 'RIGHT':
            self.direction = 'DOWN'
            wasChanged = True
        if not wasChanged and change_to == 'RIGHT' and self.direction == 'DOWN':
            self.direction = 'LEFT'
            wasChanged = True
        if not wasChanged and change_to == 'RIGHT' and self.direction == 'LEFT':
            self.direction = 'UP'
            wasChanged = True
        if not wasChanged and change_to == 'LEFT' and self.direction == 'UP':
            self.direction = 'LEFT'
            wasChanged = True
        if not wasChanged and change_to == 'LEFT' and self.direction == 'LEFT':
            self.direction = 'DOWN'
            wasChanged = True
        if not wasChanged and change_to == 'LEFT' and self.direction == 'DOWN':
            self.direction = 'RIGHT'
            wasChanged = True
        if not wasChanged and change_to == 'LEFT' and self.direction == 'RIGHT':
            self.direction = 'UP'
            wasChanged = True

        # Update snake position [x,y]
        if self.direction == 'RIGHT':
            self.snakePos[0] += 1  # self.thickness
        if self.direction == 'LEFT':
            self.snakePos[0] -= 1  # self.thickness
        if self.direction == 'UP':
            self.snakePos[1] -= 1  # self.thickness
        if self.direction == 'DOWN':
            self.snakePos[1] += 1  # self.thickness

        # Snake body mechanism
        self.snakeBody.insert(0, list(self.snakePos))
        if self.is_in_food_position():
            self.score += 1
            self.foodSpawn = False
            # test not growing snake
        #             self.snakeBody.pop()
        else:
            self.snakeBody.pop()
        self.check_game_over()

    def is_in_food_position(self):
        return self.snakePos[0] == self.foodPos[0] and self.snakePos[1] == self.foodPos[1]

    def spawn_food(self):
        # Food Spawn
        if self.foodSpawn == False:
            self.foodPos = [random.randrange(1, self.height - 1),  # *self.thickness,
                            random.randrange(1, self.width - 1)  # *self.thickness
                            ]
        self.foodSpawn = True

    def check_game_over(self):
        # Bound
        if self.snakePos[0] > (self.height - 1) or self.snakePos[0] < 0:
            self.game_over = True
        if self.snakePos[1] > (self.width - 1) or self.snakePos[1] < 0:
            self.game_over = True
        # Self hit
        for block in self.snakeBody[1:]:
            if self.snakePos[0] == block[0] and self.snakePos[1] == block[1]:
                self.game_over = True
        self.current_step += 1
        if self.current_step > self.max_episode_steps:
            self.game_over = True

    def play_game(self):
        #self.init_interface()
        while self.game_over == False:
            #direction = self.get_pressed_key()
            self.updateGameMatrix()
            direction = self.get_direction_from_GA()
            self.move_snake(direction)
            if self.mode != "auto":
                self.render()
        # if self.mode != "auto":
        #    self.render_game_over()

    def reset(self):
        ''' reset method for openai
        '''
        self.snakePos = [10, 5]
        self.snakeBody = [[10, 5], [9, 5], [8, 5]]
        self.foodSpawn = False
        self.spawn_food()
        self.game_over = False
        self.direction = 'RIGHT'
        self.score = 0
        self.current_step = 0

    def render(self, mode='human', close=False):
        ''' Render method for openai
        '''
        self.init_interface()
        # Background
        self.playSurface.fill(self.yellow)
        # Draw Snake
        firstElement = True
        for pos in self.snakeBody:
            if firstElement:
                self.pygame.draw.rect(self.playSurface, self.blue,
                                      self.pygame.Rect(self.thickness * pos[0], self.thickness * pos[1], self.thickness,
                                                       self.thickness))
                firstElement = False
            else:
                self.pygame.draw.rect(self.playSurface, self.green,
                                      self.pygame.Rect(self.thickness * pos[0], self.thickness * pos[1], self.thickness,
                                                       self.thickness))

        self.pygame.draw.rect(self.playSurface, self.brown,
                              self.pygame.Rect(self.thickness * self.foodPos[0], self.thickness * self.foodPos[1],
                                               self.thickness, self.thickness))
        self.render_score()
        self.pygame.display.flip()
        self.fpsController.tick(10000)
        # self.draw_toColab()

    def init_interface(self):
        self.pygame = pygame
        self.pygame.init()
        self.fpsController = self.pygame.time.Clock()
        # Play surface
        self.playSurface = self.pygame.display.set_mode((self.height * self.thickness, self.width * self.thickness))
        self.pygame.display.set_caption('Snake game!')

    def updateGameMatrix(self):
        for i in range(self.gameMatrix.shape[0]):
            for j in range(self.gameMatrix.shape[1]):
                self.gameMatrix[i][j] = 0.01
        # update snake pos
        for i in self.snakeBody:
            self.gameMatrix[i[0]][i[1]] = 0.5;
        self.gameMatrix[self.snakePos[0]][self.snakePos[1]] = 0.4
        self.gameMatrix[self.foodPos[0]][self.foodPos[1]] = 0.99

    def get_direction_from_GA(self):
        newm = self.gameMatrix.flatten()
        result = np.dot(self.gameMatrix, self.agent)
        result = np.max(result, axis=0)
        # result = np.dot(result, np.ones((20,3)))
        action = np.argmax(result)
        return self.possible_actions[action]


if __name__ == '__main__':
    numBots = 40
    popul = []  # здесь будет лежать популяция
    reward_list = []  # здесь будет сумма вознаграждений для каждого эпизода
    for i in range(numBots):
        bot = np.random.random((game_heigh, 3))
        popul.append(bot)
    nsurv = 20  # количество выживших
    nnew = numBots - nsurv  # количество новых
    epohs = 500  # количество эпох
    mut = 0.2  # коэфициент мутаций
    mutStrength = 1e-1
    curr_time = time.time()
    prevscore = 0
    snake_game = Snake(width=game_heigh, thickness=20)
    snake_game.init_interface()

    for it in range(epohs):  # создали список списков всех значений по эпохам
        reward_list = []
        for bot in popul:
            snake_game.agent = bot
            snake_game.reset()
            # if prevscore != currentscore:
            #    snake_game.mode = 'render'
            snake_game.play_game()
            reward_list.append(snake_game.score)
        newpopul, sval = getSurvPopul(popul = popul, val = reward_list, nsurv = nsurv, reverse=1)  # получили популяцию выживших, нас интересует бот с максимальным успехом, поэтому reverse = 1
        print(it, time.time() - curr_time, " ",sval[0:3])  # Выводим время на операцию, среднее значение и 20 лучших ботов
        # проходимся по новой популяции
        for k in range(nnew):
            # вытаскиваем новых родителей
            botp1, botp2 = getParents(newpopul, nsurv)
            newbot = []  # здесь будет новый бот
            for j in range(len(botp1)):  # боты-родители одинаковой длины, будем проходиться по каждому элементу родителя
                x = crossPointFrom2Parents(botp1, botp2, j)  # скрещиваем
                for t in range(botp1.shape[1]):
                    if random.random() < mut:
                        if random.random() < 0.5:
                            x[t] = random.random()
                        else:
                            x[t] += random.random() * mutStrength
                newbot.append(x)  # закидываем элемент в бота
            newpopul.append(newbot)  # добавляем бота
        popul = newpopul # вывести список на эпоху
        popul = np.array(popul)  # для того, чтобы можно было легко вытащить индексы условием, преобразуем в numpy массив
