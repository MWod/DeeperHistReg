### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import math
import torch as tc

### Internal Imports ###

########################


### Linear Splines ###
def b_01(x):
    return 1 - x

def b_11(x):
    return x

linear_splines = {0: b_01, 1: b_11}

### Cubic Splines ###

def b_03(x):
    return (1-x)**3 / 6

def b_13(x):
    return (3*x**3 - 6*x**2 + 4) / 6

def b_23(x):
    return (-3*x**3 + 3*x**2 + 3*x + 1) / 6

def b_33(x):
    return x**3 / 6

cubic_splines = {0: b_03, 1: b_13, 2: b_23, 3: b_33}

######################


def b_splines(
    template : tc.Tensor,
    control_points : tc.Tensor, 
    spacing : tuple,
    splines_type : str='cubic',
    normalize : bool=True) -> tc.Tensor:
    """
    TODO
    """
    if splines_type == "cubic":
        splines = cubic_splines
        splines_length = 4
        adder = 0
    elif splines_type == 'linear':
        splines = linear_splines
        splines_length = 2
        adder = 1
    else:
        raise ValueError("Unsupported splines type.")

    batch_size, _, y_size, x_size = template.size()
    s_x, s_y = spacing
    grid_x, grid_y = tc.meshgrid(tc.arange(x_size), tc.arange(y_size), indexing='xy')
    grid_x, grid_y = grid_x.to(template.device), grid_y.to(template.device)
    s_x, s_y = s_x * x_size, s_y * y_size
    xss, yss = grid_x / s_x, grid_y / s_y
    f_xs, f_ys = tc.floor(xss), tc.floor(yss)
    diff_xs, diff_ys = xss - f_xs, yss - f_ys
    i, j = f_xs.type(tc.int64), f_ys.type(tc.int64)
    cached_splines = tc.zeros((batch_size, y_size, x_size, 2, splines_length), device=template.device, dtype=template.dtype)
    for z in range(splines_length):
        cached_splines[:, :, :, 0, z] = splines[z](diff_xs)
        cached_splines[:, :, :, 1, z] = splines[z](diff_ys)

    if normalize:
        n_control_points = control_points.clone()
        n_control_points[:, :, :, 0] = (control_points[:, :, :, 0] / 2.0 + 0.5) * x_size
        n_control_points[:, :, :, 1] = (control_points[:, :, :, 1] / 2.0 + 0.5) * y_size

    deformation_field = tc.zeros((batch_size, y_size, x_size, 2), device=template.device, dtype=template.dtype)
    for m in range(splines_length):
        inner_result = tc.zeros((batch_size, y_size, x_size, 2), device=template.device, dtype=template.dtype)
        for n in range(splines_length):
            diff = cached_splines[:, :, :, 1, n].unsqueeze(3).repeat(1, 1, 1, 2)
            y_index, x_index = j+n+adder, i+m+adder
            t_inner = n_control_points[:, y_index, x_index, :]
            t = diff*t_inner
            inner_result += t
        diff = cached_splines[:, :, :, 0, m].unsqueeze(3).repeat(1, 1, 1, 2)
        t = diff*inner_result
        deformation_field += t
        
    displacement_field = deformation_field - tc.stack((grid_x, grid_y), dim=2)
    displacement_field[:, :, :, 0] = displacement_field[:, :, :, 0] / x_size * 2.0
    displacement_field[:, :, :, 1] = displacement_field[:, :, :, 1] / y_size * 2.0
    return displacement_field

def create_control_points(
    template : tc.Tensor,
    spacing : tuple,
    splines_type : str='cubic',
    normalize : bool=True) -> tc.Tensor:
    """
    TODO
    """
    if splines_type == 'cubic':
        adder = 3
    elif splines_type == 'linear':
        adder = 2
    else:
        raise ValueError("Unsupported splines type.")
    batch_size, _, y_size, x_size = template.size()
    s_x, s_y = spacing
    s_x, s_y = s_x * x_size, s_y * y_size
    no_px = math.ceil((x_size+1) / s_x) + adder
    no_py = math.ceil((y_size+1) / s_y) + adder
    xs = tc.arange(-s_x, (no_px-1)*s_x, s_x, device=template.device)
    ys = tc.arange(-s_y, (no_py-1)*s_y, s_y, device=template.device)
    cp_x, cp_y = tc.meshgrid(xs, ys, indexing='xy')   
    control_points = tc.zeros(batch_size, cp_x.size(0), cp_y.size(1), 2).type_as(template)
    control_points[:, :, :, 0] = cp_x
    control_points[:, :, :, 1] = cp_y
    if normalize:
        control_points[:, :, :, 0] = ((control_points[:, :, :, 0] / x_size) - 0.5) * 2.0
        control_points[:, :, :, 1] = ((control_points[:, :, :, 1] / y_size) - 0.5) * 2.0
    return control_points