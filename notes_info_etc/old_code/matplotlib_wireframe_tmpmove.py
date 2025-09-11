# fig.update_traces(showlegend=True)#, showscale=False)


#%% Matplotlib surface plotter.
import matplotlib.pyplot as plt

hyp_dyn_pts_grid = hyp_dyn_pts_list.reshape(GRAPH_RES, GRAPH_RES, 3)
nn_out_grid = nn_out_list.reshape(GRAPH_RES, GRAPH_RES, 3)

mfig = plt.figure(0)
mfig.clear()

ax = mfig.add_subplot(111, projection='3d')
ax.clear()
for level in range(3):
    surf = ax.plot_wireframe(
        hyp_dyn_pts_grid[..., 0], hyp_dyn_pts_grid[..., 1], nn_out_grid[..., level], label='xyz'[level],
        color="C" + str(level)
    )
surf_n = ax.plot_wireframe(
    hyp_dyn_pts_grid[..., 0], hyp_dyn_pts_grid[..., 1],
    np.linalg.norm(nn_out_grid - last_fixed_pt, axis=-1),
    color="C3", label='|d|'
)
ax.plot(*(rand_pts[:-2].T),'x-')
for i, pt in enumerate(rand_pts[:-2]):
    ax.text(x=pt[0], y=pt[1], z=pt[2], s="x" + str(i))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()

plt.show()
