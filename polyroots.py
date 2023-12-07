import matplotlib.pyplot as plt
import numpy as np

import glob
import os

import cv2

from dawgz import job, schedule, after

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

PARAMS = np.arange(0.01, 15.0, 30*10)

@job(
    array=len(PARAMS),
    cpus=4,
    ram='2GB',
    time='5:00'
)
def draw_plot(i: int):
    N_ts = 3_000_000
    ts = np.exp(np.random.uniform(0, 2*np.pi, 2 * N_ts) * 1j)
    ts = ts.reshape((N_ts, 2))
    a = PARAMS[i]  # def. 8.0
    K = ts[:,1]**4 - 1.0j * ts[:,1]**2 - 1.0
    J = ts[:,0]**4 + ts[:,0]**2 - 1.0j * ts[:,0] - 1.0
    z1 = (-K + np.sqrt(K**2 - 4*a*J)) / (2*a)
    z2 = (-K - np.sqrt(K**2 - 4*a*J)) / (2*a)
    x1 = np.sqrt(z1)
    x2 = np.sqrt(z2)
    x3 = -np.sqrt(z1)
    x4 = -np.sqrt(z2)
    pts = np.column_stack([x1, x2, x3, x4]).flatten()

    fig,ax = plt.subplots(figsize=(30,30))
    fig.set_facecolor("#f4f0e7")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    for spine in ['top', 'right','left','bottom']:
        ax.spines[spine].set_visible(False)
    ax.scatter(
        x=pts.real, 
        y=pts.imag, 
        c="#262626",
        s=0.035,
        linewidths=1e-6)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title('$8.0x^4+(1.0t_2^4-1.0it_2^2-1.0)x^2+1.0t_1^4+1.0t_1^2-1.0it_1-1.0$ \n' +  r'$t_1,t_2 \in \mathbb{C}, \ \ \ |t_1|=|t_2|=1$', fontsize=50)
    
    plt.savefig('temp/polyroots_{i}.png', dpi=300, bbox_inches='tight', pad_inches=0.0)

@after(draw_plot)
@job(cpus=16, ram='32GB', time='1:00:00')
def draw_video():
    os.system('ffmpeg -y -r 30 -i temp/polyroots_%d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p polyroots.mp4')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    schedule(
        draw_video,
        name='viz',
        backend='slurm',
    )