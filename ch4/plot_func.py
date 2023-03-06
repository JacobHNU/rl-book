import matplotlib.pyplot as plt

def plot(data):
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    titles = ['without ace', 'with ace']
    have_aces = [0,1]
    extent = [12, 22, 1, 11]
    for title, have_aces, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)