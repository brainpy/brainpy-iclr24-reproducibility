import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

fig, gs = bp.visualize.get_figure(4, 3, 2, 3)
xs = bm.linspace(-3, 3, 1000)
i = 0
for name in dir(bm.surrogate):
  if not name.startswith('__') and name[0].islower() and name[-1] != '2':
    ob = getattr(bm.surrogate, name)
    fig.add_subplot(gs[i // gs.ncols, i % gs.ncols])
    grads = bm.vector_grad(ob)(xs)
    plt.plot(bm.as_numpy(xs), bm.as_numpy(grads))
    plt.title(name)
    if i + 1 == gs.ncols * gs.nrows:
      break
    if i // gs.ncols + 1 == gs.nrows:
      plt.xlabel('$x$')
    if i % gs.ncols == 0:
      plt.ylabel("$g'(x)$")
    i += 1
plt.savefig('surrogate_gradient_funcs.png', dpi=300, transparent=True)
plt.show()
