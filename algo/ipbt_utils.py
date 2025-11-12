import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.kernels import MaternKernel
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch.distributions import Normal

def shrink_perturb_tensor_v2(tensor, reinit, shrink_coeff, perturb_coeff):
    if tensor.device != reinit.device:
        print(f'{tensor.device=} {reinit.device=} --- Mismatch! Moving to cuda.')
        tensor = tensor.to('cuda')
        reinit = reinit.to('cuda')

    return shrink_coeff * tensor + perturb_coeff * reinit

def shrink_perturb_state_via_named_params(state, shrink_perturb_pair, fresh_model):
    sh, pe = shrink_perturb_pair
    if np.isclose(sh, 0) and np.isclose(pe, 1):
        return fresh_model.state_dict()
    if np.isclose(sh, 1) and np.isclose(pe, 0):
        return state

    new_state = {}
    named_params_dict = dict(fresh_model.named_parameters())
    fresh_sd = fresh_model.state_dict()

    # Collect names of all Norm buffers
    norm_buffer_names = set()
    for module_name, module in fresh_model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._NormBase):
            for buffer_name, _ in module.named_buffers():
                full_buffer_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                norm_buffer_names.add(full_buffer_name)

    for k, v in state.items():
        if k not in named_params_dict:
            # Check if 'k' is a BatchNorm buffer
            if k in norm_buffer_names:
                new_state[k] = fresh_sd[k]
            else:
                new_state[k] = v
        else:
            reinit_tensor = named_params_dict[k].data # parameter -> tensor
            new_state[k] = shrink_perturb_tensor_v2(v, reinit_tensor, sh, pe)

    return new_state

def _get_all_bounds(search_space):
    bounds_cont = search_space.get_bounds_cont(treat_int_as_cont=True)
    bounds_noncont = search_space.get_bounds_noncont(treat_int_as_cont=True)
    bounds = {}
    for hp_name in search_space.get_hp_names():  # to preserve order
        if hp_name in bounds_cont:
            bounds[hp_name] = bounds_cont[hp_name]
        else:
            # don't normalize categorical hps
            bounds[hp_name] = (0, 1)
    return bounds, bounds_cont, bounds_noncont

def _adjust_categorical(search_space, current, hparams, dfnewpoint, bounds, bounds_cont, bounds_noncont):
    # for categorical hyperparameters, need to go from value to its index
    # (and same for current)
    # (no normalization should be done)
    if current is not None:
        current = pd.DataFrame(current, columns=list(bounds.keys()))
    for hp_noncont in bounds_noncont.keys():
        hparams.loc[:, hp_noncont] = hparams[hp_noncont].apply(
            lambda x: search_space.get_idx_by_value(hp_noncont, x)
        )
        if dfnewpoint is not None:
            dfnewpoint.loc[:, hp_noncont] = dfnewpoint[hp_noncont].apply(
                lambda x: search_space.get_idx_by_value(hp_noncont, x)
            )
        if current is not None:
            current.loc[:, hp_noncont] = current[hp_noncont].apply(
                lambda x: search_space.get_idx_by_value(hp_noncont, x)
            )
    if current is not None:
        # current contained categorical values (likely strings) => cont values also became strings
        # => need to convert them back to float
        for hp_cont in bounds_cont.keys():
            current[hp_cont] = current[hp_cont].astype(float)
        current = current.values
    return current, hparams, dfnewpoint

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()#LinearMean(1) # example had ConstantMean()
        # covar_kernel = gpytorch.kernels.RBFKernel()
        covar_kernel = MaternKernel(nu=2.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(covar_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_approximate_gp(train_x, train_y, inducing_points, training_iterations=50):
    model = ApproximateGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_iterations)

    # Train
    model.train()
    likelihood.train()
    for i in range(training_iterations):
        output = model(train_x)
        loss = -objective_function(output, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def train_approximate_gp_and_gauge_improvement(train_x, train_y, inducing_points, path, title='', training_iterations=50):
    model, likelihood = train_approximate_gp(train_x, train_y, inducing_points, training_iterations)

    # Test
    model.eval()
    likelihood.eval()

    viz = True

    with torch.no_grad():
        f_dist = model(inducing_points)
        mean = f_dist.mean

        if viz:
            f_lower, f_upper = f_dist.confidence_region()
            y_dist = likelihood(f_dist)
            y_lower, y_upper = y_dist.confidence_region()

    has_improved = mean[-1] > mean[-2]

    if viz:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        line, = ax.plot(inducing_points, mean, "blue")
        ax.fill_between(inducing_points, f_lower, f_upper, color=line.get_color(), alpha=0.3, label="q(f)")
        ax.fill_between(inducing_points, y_lower, y_upper, color=line.get_color(), alpha=0.1, label="p(y)")
        ax.scatter(train_x, train_y, c='k', marker='.', label="Data")
        ax.legend(loc="lower right")
        ax.set(xlabel="x", ylabel="y")
        ax.set_title(title + f' | {"improved" if has_improved else "failed"}')
        print(f'{mean=}')
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f'{title}.png')
        plt.close()

    return has_improved