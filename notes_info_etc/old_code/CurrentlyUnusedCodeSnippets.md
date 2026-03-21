# Linear Regression with custom loss

Could also do this with tensorflow. Results are slightly less accurate but way faster and way easier to code.
```py
def makeHomogenous(X_in):
    '''Add a column of 1s to the passed-in matrix.'''
    return np.concatenate((X_in, np.ones_like(X_in[:, :1])), axis=-1)

def scipyLossJAV(beta, X, Y):
    pred = X @ beta.reshape(-1, 3)
    error = np.mean(poseLossJAV(Y, pred))
    return(error)

# You must provide a starting point at which to initialize
# the parameter search space
nonco_train_homog = makeHomogenous(z_nonco_train_data)

# The below takes forever to run!
from scipy.optimize import minimize
beta_init = np.ones((nonco_train_homog.shape[1] * 3))
result = minimize(scipyLossJAV, beta_init, args=(nonco_train_homog, bcs_train.numpy()),
                  method='BFGS', options={'maxiter': 32})
lin_JAV_multiplier_weights = result.x
del beta_init

nonco_test_homog = makeHomogenous(z_nonco_test_data)[..., :-1]

loss = scipyLossJAV(lin_JAV_multiplier_weights, nonco_test_homog[s_ind_dict['skip0']], bcs_test.numpy()[s_ind_dict['skip0']])

del nonco_train_homog # Delete RAM-hungry copies that don't get used again.
del nonco_test_homog
```

# Matplotlib wireframe plotting:
```py
import matplotlib.pyplot as plt

fig = plt.figure(0)
fig.clear()

ax = fig.add_subplot(111, projection='3d')
ax.clear()
for level in range(3):
    surf = ax.plot_wireframe(
        X, Y, rand_out_vec3s[..., level], label='xyz'[level],
        color="C" + str(level)
    )
surf_n = ax.plot_wireframe(
    X, Y, np.linalg.norm(rand_out_vec3s - xyz_ins, axis=-1),
    color="C3", label='|d|'
)
ax.plot(*(rand_pts[:-2].T),'x-')
for i, pt in enumerate(rand_pts[:-2]):
    ax.text(x=pt[0], y=pt[1], z=pt[2], s="x" + str(i))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()

plt.show()
```