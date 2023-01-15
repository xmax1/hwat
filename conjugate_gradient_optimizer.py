"""Conjugate Gradient Optimizer.
Computes the decent direction using the conjugate gradient method, and then
computes the optimal step size that will satisfy the KL divergence constraint.
Finally, it performs a backtracking line search to optimize the objective.
"""
import warnings

from dowel import logger
import numpy as np
import torch
from torch.optim import Optimizer

def unflatten_tensors(flattened, tensor_shapes):
    """Unflatten a flattened tensors into a list of tensors.
    Args:
        flattened (numpy.ndarray): Flattened tensors.
        tensor_shapes (tuple): Tensor shapes.
    Returns:
        list[numpy.ndarray]: Unflattened list of tensors.
    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    return [
        torch.reshape(pair[0], pair[1])
        for pair in zip(torch.split(flattened, tensor_sizes), tensor_shapes)
    ]


def _build_hessian_vector_product(func, params, reg_coeff=1e-5):
    """Computes Hessian-vector product using Pearlmutter's algorithm.
    `Pearlmutter, Barak A. "Fast exact multiplication by the Hessian." Neural
    computation 6.1 (1994): 147-160.`
    Args:
        func (callable): A function that returns a torch.Tensor. Hessian of
            the return value will be computed.
        params (list[torch.Tensor]): A list of function parameters.
        reg_coeff (float): A small value so that A -> A + reg*I.
    Returns:
        function: It can be called to get the final result.
    """
    param_shapes = [p.shape or torch.Size([1]) for p in params]
    f = func()
    f_grads = torch.autograd.grad(f, params, create_graph=True)

    def _eval(vector):
        """The evaluation function.
        Args:
            vector (torch.Tensor): The vector to be multiplied with
                Hessian.
        Returns:
            torch.Tensor: The product of Hessian of function f and v.
        """
        # unflatten_vector = unflatten_tensors(vector.cpu(), param_shapes)
        unflatten_vector = unflatten_tensors(vector, param_shapes)

        assert len(f_grads) == len(unflatten_vector)
        grad_vector_product = torch.sum(
            torch.stack(
                [torch.sum(g * x) for g, x in zip(f_grads, unflatten_vector)]))

        hvp = list(
            torch.autograd.grad(grad_vector_product, params,
                                retain_graph=True))
        for i, (hx, p) in enumerate(zip(hvp, params)):
            if hx is None:
                hvp[i] = torch.zeros_like(p)

        flat_output = torch.cat([h.reshape(-1) for h in hvp])
        return flat_output + reg_coeff * vector

    return _eval


def _conjugate_gradient(f_Ax, b, cg_iters, residual_tol=1e-10):
    """Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.
    Args:
        f_Ax (callable): A function to compute Hessian vector product.
        b (torch.Tensor): Right hand side of the equation to solve.
        cg_iters (int): Number of iterations to run conjugate gradient
            algorithm.
        residual_tol (float): Tolerence for convergence.
    Returns:
        torch.Tensor: Solution x* for equation Ax = b.
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    print(f'First rdotr {rdotr}')

    for _ in range(cg_iters):
        z = f_Ax(p) # Ap 
        v = rdotr / torch.dot(p, z) # alpha
        x += v * p # x = x + alpha * p
        r -= v * z # r = r - alpha * Ap
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr # beta
        p = r + mu * p # p = r + beta * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

        
    print("Last rdotr", rdotr)
    return x

class ConjugateGradientOptimizer(Optimizer):
    """Performs constrained optimization via backtracking line search.
    The search direction is computed using a conjugate gradient algorithm,
    which gives x = A^{-1}g, where A is a second order approximation of the
    constraint and g is the gradient of the loss function.
    Args:
        params (iterable): Iterable of parameters to optimize.
        max_constraint_value (float): Maximum constraint value.
        cg_iters (int): The number of CG iterations used to calculate A^-1 g
        max_backtracks (int): Max number of iterations for backtrack
            linesearch.
        backtrack_ratio (float): backtrack ratio for backtracking line search.
        hvp_reg_coeff (float): A small value so that A -> A + reg*I. It is
            used by Hessian Vector Product calculation.
        accept_violation (bool): whether to accept the descent step if it
            violates the line search condition after exhausting all
            backtracking budgets.
    """

    def __init__(self,
                 params,
                 max_constraint_value,
                 cg_iters=10,
                 max_backtracks=15,
                 backtrack_ratio=0.8,
                 hvp_reg_coeff=1e-5,
                 accept_violation=False):
        super().__init__(params, {})
        self._max_constraint_value = max_constraint_value
        self._cg_iters = cg_iters
        self._max_backtracks = max_backtracks
        self._backtrack_ratio = backtrack_ratio
        self._hvp_reg_coeff = hvp_reg_coeff
        self._accept_violation = accept_violation

    def step(self, f_loss):  # pylint: disable=arguments-differ
        """Take an optimization step.
        Args:
            f_loss (callable): Function to compute the loss.
            f_constraint (callable): Function to compute the constraint value.
        """
        # Collect trainable parameters and gradients
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))
        flat_loss_grads = torch.cat(grads)

        # Build Hessian-vector-product function
        f_Ax = _build_hessian_vector_product(f_loss, params, self._hvp_reg_coeff) # replace with f_loss instead of f_constraint

        # Compute step direction
        step_dir = _conjugate_gradient(f_Ax, flat_loss_grads, self._cg_iters)

        # Replace nan with 0.
        step_dir[step_dir.ne(step_dir)] = 0.

        # Compute step size
        step_size = torch.sqrt(2.0 * self._max_constraint_value *
                            (1. /
                             (torch.dot(step_dir, f_Ax(step_dir)) + 1e-8)))

        print(f'step_dir dot f_Ax(step_dir): {torch.dot(step_dir, f_Ax(step_dir))}')

        print(f'step_size: {step_size}')

        if torch.isnan(step_size):
            print(f'HESSIAN NOT POSITIVE DEFINITE! step_size is nan')
            step_size = 0.1
        
        step_size = 0.2
        descent_step = step_size * step_dir
        
        # Check if descent direction makes any sense
        f_grad_dot_descent_step = torch.dot(flat_loss_grads, descent_step)
        print(f'f_grad_dot_descent_step: {f_grad_dot_descent_step}')
        if f_grad_dot_descent_step > 0:
            print('CONJUGATE DIRECTION IS FAULTY! f_grad_dot_descent_step is positive')
            # choose steepest descent direction instead
            # descent_step = -1*flat_loss_grads*step_size
            # f_grad_dot_descent_step = torch.dot(flat_loss_grads, descent_step)
            # print(f'f_grad_dot_descent_step NEW: {f_grad_dot_descent_step}')

        # Update parameters using backtracking line search
        self._backtracking_line_search(params, descent_step, f_loss, f_grad_dot_descent_step)

    @property
    def state(self):
        """dict: The hyper-parameters of the optimizer."""
        return {
            'max_constraint_value': self._max_constraint_value,
            'cg_iters': self._cg_iters,
            'max_backtracks': self._max_backtracks,
            'backtrack_ratio': self._backtrack_ratio,
            'hvp_reg_coeff': self._hvp_reg_coeff,
            'accept_violation': self._accept_violation,
        }

    @state.setter
    def state(self, state):
        # _max_constraint_value doesn't have a default value in __init__.
        # The rest of thsese should match those default values.
        # These values should only actually get used when unpickling a
        self._max_constraint_value = state.get('max_constraint_value', 0.01)
        self._cg_iters = state.get('cg_iters', 10)
        self._max_backtracks = state.get('max_backtracks', 15)
        self._backtrack_ratio = state.get('backtrack_ratio', 0.8)
        self._hvp_reg_coeff = state.get('hvp_reg_coeff', 1e-5)
        self._accept_violation = state.get('accept_violation', False)

    def __setstate__(self, state):
        """Restore the optimizer state.
        Args:
            state (dict): State dictionary.
        """
        if 'hvp_reg_coeff' not in state['state']:
            warnings.warn(
                'Resuming ConjugateGradientOptimizer with lost state. '
                'This behavior is fixed if pickling from garage>=2020.02.0.')
        self.defaults = state['defaults']
        # Set the fields manually so that the setter gets called.
        self.state = state['state']
        self.param_groups = state['param_groups']

    def _backtracking_line_search(self, params, descent_step, f_loss,
                                  f_grad_dot_descent_step):
        prev_params = [p.clone() for p in params]
        ratio_list = self._backtrack_ratio**np.arange(self._max_backtracks)
        loss_before = f_loss()
        print(f'OLD LOSS: {loss_before}')

        param_shapes = [p.shape or torch.Size([1]) for p in params]
        descent_step = unflatten_tensors(descent_step, param_shapes)
        assert len(descent_step) == len(params)

        for ratio in ratio_list:
            for step, prev_param, param in zip(descent_step, prev_params, params):
                step = ratio * step
                new_param = prev_param.data - step
                param.data = new_param.data

            loss = f_loss()
            # constraint_val = f_constraint()
            # if (loss < loss_before and constraint_val <= self._max_constraint_value):
            #     print(f'Line search successful with ratio {ratio}')
            #     break

            print(f'New loss: {loss}')

            # Use Armijo condition to check if we are making sufficient progress.
            # But Armijo condition assumes f_grad_dot_descent_step is negative.
            if loss <= loss_before - 0.01 * ratio * torch.abs(f_grad_dot_descent_step):
                print(f'Line search successful with ratio {ratio}')
                # print average abs value of descent_step where descent_step is a list of tensors
                print(f'Average abs value of descent_step: {torch.mean(torch.abs(torch.cat([torch.flatten(step) for step in descent_step])))}')
                break

        if ((torch.isnan(loss) or loss >= loss_before)
                and not self._accept_violation):
            logger.log('Line search condition violated. Rejecting the step!')
            if torch.isnan(loss):
                logger.log('Violated because loss is NaN')
            if loss >= loss_before:
                logger.log('Violated because loss not improving')
            for prev, cur in zip(prev_params, params):
                cur.data = prev.data

        # if ((torch.isnan(loss) or torch.isnan(constraint_val)
        #      or loss >= loss_before
        #      or constraint_val >= self._max_constraint_value)
        #         and not self._accept_violation):
        #     logger.log('Line search condition violated. Rejecting the step!')
        #     if torch.isnan(loss):
        #         logger.log('Violated because loss is NaN')
        #     if torch.isnan(constraint_val):
        #         logger.log('Violated because constraint is NaN')
        #     if loss >= loss_before:
        #         logger.log('Violated because loss not improving')
        #     if constraint_val >= self._max_constraint_value:
        #         logger.log('Violated because constraint is violated')
        #     for prev, cur in zip(prev_params, params):
        #         cur.data = prev.data