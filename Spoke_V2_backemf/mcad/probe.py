import sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.argv = ['x']; exec(open('digitize.py').read().split('out = {}')[0])
im = load('17e55a0f-image.jpg')
Hm = homography([(422, 2448), (3928, 2502), (412, 659), (3918, 751)], [(0, -12), (340, -12), (0, 12), (340, 12)])
sub = im[680:2440, 365:4000]
chroma = sub.max(2) - sub.min(2)
g = sub.mean(2)
bg = ndimage.median_filter(g, size=31)
for name, m in (('dark', (bg - g) > 28), ('chroma', ((sub[:, :, 2] - sub[:, :, 1]) > 12) & ((bg - g) > 10))):
    ys, xs = np.nonzero(m)
    th, v = apply(Hm, xs + 365.0, ys + 680.0)
    fig, ax = plt.subplots(figsize=(14, 5)); ax.plot(th, v, ',k'); ax.set_ylim(-13, 13); ax.grid()
    fig.savefig('probe_%s.png' % name, dpi=80); plt.close(fig)
    print(name, m.sum())
