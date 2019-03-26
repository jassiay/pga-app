import numpy as np
import random
import matplotlib.pyplot as plt

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 1
LEN = 12
WID = 4
# Consts for PGA-APP
THETA = 0.1
ETA = 0.1
XI = 1
PLENGTH = 0.1

def take_step(x,y,a):
    goal = 0
    if x == LEN - 1 and y == 0:
        goal = 1
    if a == 0:
        y += 1
    if a == 1:
        x += 1
    if a == 2:
        y -= 1
    if a == 3:
        x -= 1
        
    x = max(0,x)
    x = min(LEN-1, x)
    y = max(0,y)
    y = min(WID-1, y)

    if goal == 1:
        return x,y,-1
    if x>0 and x<LEN-1 and y==0:
        return 0,0,-100
    return x,y,-1

def epsGreedyPolicyPGA(x,y,pi,eps):
    t = random.randint(0,3)
    if random.random() < eps:
        a = t
    else:
        pi_max = pi[x][y][0]
        a_max = 0
        m = [0]
        for i in range(4):
            if pi[x][y][i] > pi_max:
                pi_max = pi[x][y][i]
                a_max = i
                m = [i]
            elif i>0 and pi[x][y][i] == pi_max:
                m.append(i)
        if len(m)==1:
            a = a_max
        else:
            a = random.choice(m)

    return a

def findMaxQ(x,y,q):
    q_max = q[x][y][0]
    a_max = 0
    m = [0]
    for i in range(4):
        if q[x][y][i] > q_max:
            q_max = q[x][y][i]
            a_max = i
            m = [i]
        elif i>0 and q[x][y][i] == q_max:
            m.append(i)
    if len(m)==1:
        a = a_max
    else:
        a = random.choice(m)
    return a
    
def policyProb(x,y,q):
    q_max = q[x][y][0]
    m = [0]
    for i in range(4):
        if q[x][y][i] > q_max:
            q_max = q[x][y][i]
            m = [i]
        elif i>0 and q[x][y][i] == q_max:
            m.append(i)
    pi_new = np.zeros(4)
    for i in m:
        pi_new[i] = 1 / len(m)
    return pi_new

def pga_app(q,pi):
    runs = 30
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            while True:
                a = epsGreedyPolicyPGA(x,y,pi,EPSILON)             
                x_next, y_next,reward = take_step(x,y,a)
                a_next = findMaxQ(x_next,y_next,q)
                reward_sum += reward
                q[x][y][a] = (1-THETA)*q[x][y][a] + THETA*(reward + XI*q[x_next][y_next][a_next]) # Q-learning
                pi_prob = policyProb(x,y,q)
                v = 0
                for action in range(4):
                    v += pi_prob[action] * q[x][y][action]
                for action in range(4):
                    if pi_prob[action] == 1:
                        delta_hat = q[x][y][action] - v
                    else:
                        delta_hat = (q[x][y][action] - v)/(1-pi_prob[action])
                    delta = delta_hat - PLENGTH* abs(delta_hat)* pi_prob[action]
                    pi[x][y][action] = pi_prob[action] + ETA* delta

                sum = 0
                for action in range(4):
                    if pi[x][y][action] >=1:
                        pi[x][y][action] = 1
                    if pi[x][y][action] <=0:
                        pi[x][y][action] = 0
                    sum += pi[x][y][action]
                for action in range(4):
                    pi[x][y][action] * (1/sum)


                # Cliff Walking termination condition
                if x == LEN - 1 and y==0:
                    break
                x = x_next
                y = y_next
            rewards[i] += reward_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards


def showOptimalPolicy(q):
    for j in range(WID-1,-1,-1):
        for i in range(LEN):
            a = findMaxQ(i,j,q)
            if a == 0:
                print(" U ",end = "")
            if a == 1:
                print(" R ",end = "")
            if a == 2:
                print(" D ",end = "")
            if a == 3:
                print(" L ",end = "")
        print("")

def showOptimalPath(q):
    x = 0
    y = 0
    path = np.zeros([LEN,WID]) - 1
    end = 0
    exist = np.zeros([LEN,WID])
    while (x != LEN-1 or y != 0) and end == 0:
        a = findMaxQ(x,y,q)
        path[x][y] = a
        if exist[x][y] == 1:
            end = 1
        exist[x][y] = 1
        x,y,r = take_step(x,y,a)
    for j in range(WID-1,-1,-1):
        for i in range(LEN):
            if i == 0 and j == 0:
                print(" S ",end = "")
                continue
            if i == LEN-1 and j == 0:
                print(" G ",end = "")
                continue
            a = path[i,j]
            if a == -1:
                print(" - ",end = "")
            elif a == 0:
                print(" U ",end = "")
            elif a == 1:
                print(" R ",end = "")
            elif a == 2:
                print(" D ",end = "")
            elif a == 3:
                print(" L ",end = "")
        print("")

pga_grid = np.zeros([12,4,4])
pi_grid = np.full([12,4,4], 0.25)
pga_app_rewards = pga_app(pga_grid, pi_grid)

plt.plot(range(len(pga_app_rewards)),pga_app_rewards,label="PGA-APP")
# plt.ylim(-100,0)
plt.legend(loc="lower right")
plt.show()

print("PGA-APP Optimal Policy")
showOptimalPolicy(pga_grid)

print("PGA-APP Optimal Path")
showOptimalPath(pga_grid)

print("PGA-APP policy grid and q-value grid")
print(pi_grid)
print("---")
print(pga_grid)
