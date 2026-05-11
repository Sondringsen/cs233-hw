import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE_PLOT_DIR = Path("hmwk02_code/plots")

def draw_circle(ax, cx, cy, r, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), **kwargs)

def plot_two_circles(a, r1, r2, title, filename):
    _, ax = plt.subplots(figsize=(6, 6))
    draw_circle(ax, 0, 0, r1, label=f'Circle 1 (r={r1})')
    draw_circle(ax, a, 0, r2, label=f'Circle 2 (r={r2}, center=({a},0))')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(BASE_PLOT_DIR / Path(filename))
    plt.close()

def inscribed_triangle(cx, cy, r, angle_offset=np.pi / 2):
    """Vertices of an equilateral triangle inscribed in a circle."""
    angles = angle_offset + np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    return np.array([[cx + r * np.cos(a), cy + r * np.sin(a)] for a in angles])

def plot_non_overlapping_with_triangles(a, r1, r2, filename):
    _, ax = plt.subplots(figsize=(8, 5))
    draw_circle(ax, 0, 0, r1, color='tab:blue',   label=f'$C_1$: center $(0,0)$, $r_1={r1}$')
    draw_circle(ax, a, 0, r2, color='tab:orange', label=f'$C_2$: center $({a},0)$, $r_2={r2}$')

    for cx, r, color, prefix in [(0, r1, 'tab:blue', 'A'), (a, r2, 'tab:orange', 'B')]:
        verts = inscribed_triangle(cx, 0, r)
        tri = plt.Polygon(verts, fill=False, edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(tri)
        for i, v in enumerate(verts):
            ax.plot(*v, 'o', color=color, markersize=7, zorder=5)
            ax.annotate(f'${prefix}_{i+1}$', v, textcoords='offset points',
                        xytext=(6, 4), fontsize=11)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Non-overlapping circles with inscribed triangles')
    plt.tight_layout()
    plt.savefig(BASE_PLOT_DIR / Path(filename))
    plt.close()

def plot_isometric_embedding(epsilon=0.3, filename='isometric_embedding.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Isometric Embedding', fontsize=13)

    # ── Original space: two pairs separated vertically ──
    x1 = np.array([0.0,  1.0])
    x2 = np.array([2.0,  1.0])
    y1 = np.array([0.0, -1.0])
    y2 = np.array([2.0 + 2*epsilon, -1.0])

    ax1.plot([x1[0], x2[0]], [x1[1], x2[1]], color='royalblue', lw=1.5)
    ax1.plot([y1[0], y2[0]], [y1[1], y2[1]], color='tomato', lw=1.5)

    for pt, label, color, off in [
        (x1, r'$x_1$', 'royalblue', (-20,  6)),
        (x2, r'$x_2$', 'royalblue', (  6,  6)),
        (y1, r'$y_1$', 'tomato',    (-20, -15)),
        (y2, r'$y_2$', 'tomato',    (  6, -15)),
    ]:
        ax1.scatter(*pt, color=color, s=70, zorder=5)
        ax1.annotate(label, pt, textcoords='offset points', xytext=off, fontsize=11)

    ax1.annotate('', xy=x2, xytext=x1,
                 arrowprops=dict(arrowstyle='<->', color='royalblue', lw=1.2))
    ax1.text(1.0, 1.30, r'$d = 2$', ha='center', color='royalblue', fontsize=10)

    ax1.annotate('', xy=y2, xytext=y1,
                 arrowprops=dict(arrowstyle='<->', color='tomato', lw=1.2))
    ax1.text((y1[0]+y2[0])/2, -1.35, r'$d = 2 + 2\varepsilon$',
             ha='center', color='tomato', fontsize=10)

    ax1.set_xlim(-0.7, 3.9); ax1.set_ylim(-2.1, 2.1)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Original Space', fontsize=12)

    # ── Embedded space: everything on the y-axis ──
    x1_e = np.array([0.0, 0.0])
    x2_e = np.array([0.0, 2.0])
    y2_e = np.array([0.0, 2.0 + 2*epsilon])

    ax2.axvline(x=0, color='gray', lw=0.7, ls='--', zorder=0)

    ax2.plot([x1_e[0], x2_e[0]], [x1_e[1], x2_e[1]], color='royalblue', lw=1.5,
             label=r'$x$ pair')
    ax2.plot([x1_e[0], y2_e[0]], [x1_e[1], y2_e[1]], color='tomato', lw=1.5,
             ls='--', label=r'$y$ pair')

    # x_1 / y_1 coincide at origin
    ax2.scatter(*x1_e, color='royalblue', s=70, zorder=6)
    ax2.scatter(*x1_e, color='tomato',    s=70, marker='x', linewidths=2, zorder=7)
    ax2.scatter(*x2_e, color='royalblue', s=70, zorder=6)
    ax2.scatter(*y2_e, color='tomato',    s=70, marker='x', linewidths=2, zorder=7)

    ax2.annotate(r'$x_1 = y_1 = 0$', x1_e, textcoords='offset points',
                 xytext=(8, -12), fontsize=10)
    ax2.annotate(r'$x_2$', x2_e, textcoords='offset points',
                 xytext=(8, 0), fontsize=11, color='royalblue')
    ax2.annotate(r'$y_2$', y2_e, textcoords='offset points',
                 xytext=(8, 0), fontsize=11, color='tomato')

    ax2.annotate('', xy=x2_e + [-0.12, 0], xytext=x1_e + [-0.12, 0],
                 arrowprops=dict(arrowstyle='<->', color='royalblue', lw=1.2))
    ax2.text(-0.18, 1.0, r'$2$', ha='right', color='royalblue', fontsize=10)

    ax2.annotate('', xy=y2_e + [0.12, 0], xytext=x1_e + [0.12, 0],
                 arrowprops=dict(arrowstyle='<->', color='tomato', lw=1.2))
    ax2.text(0.18, (2+2*epsilon)/2, r'$2+2\varepsilon$',
             ha='left', color='tomato', fontsize=10)

    ax2.set_xlim(-0.8, 0.8); ax2.set_ylim(-0.4, 3.2)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Embedded Space', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(BASE_PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()



r1, r2 = 1.0, 0.8

plot_two_circles(a=1.2, r1=r1, r2=r2,
                 title='Overlapping circles',
                 filename='two_circles_overlapping.png')

plot_two_circles(a=3.0, r1=r1, r2=r2,
                 title='Non-overlapping circles',
                 filename='two_circles_non_overlapping.png')

plot_non_overlapping_with_triangles(a=3.0, r1=r1, r2=r2,
                                    filename='two_circles_inscribed_triangles.png')

plot_isometric_embedding(epsilon=0.3)



def plot_point_clouds(coord_list):
	n = len(coord_list)
	ncols = 3
	nrows = (n + ncols - 1) // ncols
	fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
	for i, pts in enumerate(coord_list):
		ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
		ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.6)
		ax.set_title(f'Shape {i}')
		ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
	plt.tight_layout()
	plt.show()
     

shape_dict = np.load('hmwk02_code/p3/shapes.npz', allow_pickle=True)['data'].item()
coord_list, dist_list = shape_dict['coord'], shape_dict['dist']
num_shapes = len(dist_list)

plot_point_clouds(coord_list)