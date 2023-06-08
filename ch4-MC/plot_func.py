import matplotlib.pyplot as plt
import numpy as np

def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_aces, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_aces].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()


# 绘制最优策略图像
def draw(policy, filepath):
    true_hit = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == True and policy[x] == 1]
    true_stick = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == True and policy[x] == 0]
    false_hit = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == False and policy[x] == 1]
    false_stick = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == False and policy[x] == 0]

    plt.figure(1)
    plt.plot([x[0] for x in true_hit],
             [x[1] for x in true_hit], 'bo', label='HIT')
    plt.plot([x[0] for x in true_stick],
             [x[1] for x in true_stick], 'rx', label='STICK')
    plt.xlabel('dealer'), plt.ylabel('player')
    plt.legend(loc='upper right')
    plt.title('Usable Ace')
    filename = 'UsabelAce.png'
    plt.savefig(filepath+filename, dpi=300)

    plt.figure(2)
    plt.plot([x[0] for x in false_hit],
             [x[1] for x in false_hit], 'bo', label='HIT')
    plt.plot([x[0] for x in false_stick],
             [x[1] for x in false_stick], 'rx', label='STICK')
    plt.xlabel('dealer'), plt.ylabel('player')
    plt.legend(loc='upper right')
    plt.title('No Usable Ace')
    filename = 'NoUsabelAce.png'
    plt.savefig(filepath+filename, dpi=300)

def draw3(policy, filepath):
    true_hit = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == True and np.argmax(policy[x]) == 1]
    true_stick = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == True and np.argmax(policy[x]) == 0]
    false_hit = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == False and np.argmax(policy[x]) == 1]
    false_stick = [(x[1], x[0]) for x in policy.keys(
    ) if x[2] == False and np.argmax(policy[x]) == 0]

    plt.figure(1)
    plt.plot([x[0] for x in true_hit],
             [x[1] for x in true_hit], 'bo', label='HIT')
    plt.plot([x[0] for x in true_stick],
             [x[1] for x in true_stick], 'rx', label='STICK')
    plt.xlabel('dealer'), plt.ylabel('player')
    plt.legend(loc='upper right')
    plt.title('Usable Ace')
    filename = 'code3-8 UsabelAce.png'
    plt.savefig(filepath+filename, dpi=300)

    plt.figure(2)
    plt.plot([x[0] for x in false_hit],
             [x[1] for x in false_hit], 'bo', label='HIT')
    plt.plot([x[0] for x in false_stick],
             [x[1] for x in false_stick], 'rx', label='STICK')
    plt.xlabel('dealer'), plt.ylabel('player')
    plt.legend(loc='upper right')
    plt.title('No Usable Ace')
    filename = 'code3-8 NoUsabelAce.png'
    plt.savefig(filepath+filename, dpi=300)