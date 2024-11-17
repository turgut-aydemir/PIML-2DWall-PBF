import argparse
import torch.optim as optim
from torch.autograd import grad
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import FNN
from train import *
from util import *
import torch.nn as nn

torch.manual_seed(0)


def output_transform(X): #Data de-normalization
    X = T_range * nn.Softplus()(X) + T_ref
    return X


def input_transform(X): #Data normalization
    X = 2. * (X - X_min) / (X_max - X_min) - 1.
    return X


def PDE(x, y, t, net):
    X = torch.concat([x, y, t], axis=-1)
    T = net(X)

    T_t = grad(T, t, create_graph=True, grad_outputs=torch.ones_like(T))[0]

    T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_xx = grad(T_x, x, create_graph=True, grad_outputs=torch.ones_like(T_x))[0]

    T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_yy = grad(T_y, y, create_graph=True, grad_outputs=torch.ones_like(T_y))[0]

    Cp = a1 / 1000 * (T - 310) + b1
    k = a2 * T + b2

    h = ha * y / 1e5 + hb

    f = rho * Cp * T_t - k * (T_xx + T_yy) + 2 * h / thickness * (T - T_ref) + 2 * Rboltz * emiss / thickness * (
                T ** 4 - T_ref ** 4)
    return f


def BC(x, y, t, net, loc):
    X = torch.concat([x, y, t], axis=-1)
    T = net(X)
    k = a2 * T + b2
    h = ha * y / 1e5 + hb
    if loc == '-x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_x - h * (T - T_ref) - Rboltz * emiss * (T ** 4 - T_ref ** 4)
    if loc == '+x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return -k * T_x - h * (T - T_ref) - Rboltz * emiss * (T ** 4 - T_ref ** 4)
    if loc == '-y':
        T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_y - hc * (T - T_ref)


def generate_points(p=[], f=[]):
    t = np.linspace(0, x_max[2], 71)

    bound_x_neg, _ = sampling_uniform_2D(.5, x_min, [x_max[0], x_max[1] - 1.5, x_max[2]], '-x', t)
    bound_x_pos, _ = sampling_uniform_2D(.5, x_min, [x_max[0], x_max[1] - 1.5, x_max[2]], '+x', t)
    bound_y_neg, _ = sampling_uniform_2D(.5, x_min, [x_max[0], x_max[1] - 1.5, x_max[2]], '-y', t)

    domain_pts, _ = sampling_uniform_2D([1., .25], x_min, [x_max[0], x_max[1] - 1.5, x_max[2]], 'domain', t[1:], e=0.01)

    init_dataTimeAdjust = np.load('Final1.npy').T
    indTimeAdjust = init_dataTimeAdjust[:, 2] == 2.8666667 # Initial Condition
    init_data = init_dataTimeAdjust[indTimeAdjust, :]

    p.extend([torch.tensor(bound_x_neg, requires_grad=True, dtype=torch.float).to(device),
              torch.tensor(bound_x_pos, requires_grad=True, dtype=torch.float).to(device),
              torch.tensor(bound_y_neg, requires_grad=True, dtype=torch.float).to(device),
              torch.tensor(domain_pts, requires_grad=True, dtype=torch.float).to(device),
              torch.tensor(init_data[:, 0:3], requires_grad=True, dtype=torch.float).to(device)])
    f.extend([['BC', '-x'], ['BC', '+x'], ['BC', '-y'], ['domain'],
              ['IC', torch.tensor(init_data[:, 3:4], requires_grad=True, dtype=torch.float).to(device)]])

    return p, f

def visualize_temperature_at_time(net, time_point, test_in, test_out, T_ref, T_range, x_min, x_max, device):
    # Create a tensor for input at time 'time_point' (we assume x, y are in range [x_min, x_max])
    num_points = 50  # You can adjust this to create a denser grid for higher resolution
    x_vals = np.linspace(x_min[0], x_max[0], num_points)
    y_vals = np.linspace(x_min[1], x_max[1], num_points)
    t_vals = np.full_like(x_vals, time_point)

    # Generate a grid of points for evaluation
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_t = np.full_like(grid_x, time_point)  # Time is fixed for the plot

    # Convert the grid into tensor format
    test_input = torch.tensor(np.vstack((grid_x, grid_y, grid_t)).T, dtype=torch.float32).to(device)

    # Get predicted temperatures from the model (T_predicted)
    T_pred = net(test_input)  # T_pred will be the predicted temperatures at time 'time_point'

    # De-normalize the predicted temperatures using your output transform
    T_pred = output_transform(T_pred).cpu().detach().numpy()

    # Reshape for visualization
    T_pred_reshaped = T_pred.reshape((num_points, num_points))

    # Now, let's visualize the actual temperature at the same time point
    actual_data_at_t = test_out  # Assuming you have the actual data for comparison

    # Assuming column 3 contains actual temperature at time t
    actual_temp_at_t = actual_data_at_t[:, 3]
    actual_temp_reshaped = actual_temp_at_t.reshape((num_points, num_points))

    # Plotting Predicted and Actual Temperature
    plt.figure(figsize=(10, 8))

    # Predicted Temperature
    plt.subplot(1, 2, 1)
    plt.title(f"Predicted Temperature at t = {time_point}s")
    plt.contourf(grid_x.reshape((num_points, num_points)), grid_y.reshape((num_points, num_points)), T_pred_reshaped,
                 levels=30)
    plt.colorbar(label="Temperature (K)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    # Actual Temperature
    plt.subplot(1, 2, 2)
    plt.title(f"Actual Temperature at t = {time_point}s")
    plt.contourf(grid_x.reshape((num_points, num_points)), grid_y.reshape((num_points, num_points)),
                 actual_temp_reshaped, levels=30)
    plt.colorbar(label="Temperature (K)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    plt.tight_layout()
    plt.show()

def load_data(p=[], f=[]):
    data = np.load('Final1.npy').T

    ind = (data[:, 2] > 2.85) * (data[:, 2] < 3.0) * (data[:, 1] < 3)
    data1 = data[ind, :]

    if args.task != 'baseline':
        ind = (data[:, 2] > 2.85) * (data[:, 2] < 3.0) * (data[:, 1] > 3)
        data2 = data[ind, :]
        data1 = np.vstack((data1, data2))

    p.extend([torch.tensor(data1[:, 0:3], requires_grad=True, dtype=torch.float).to(device)])
    f.extend([['data', torch.tensor(data1[:, 3:4], requires_grad=True, dtype=torch.float).to(device)]])
    return p, f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU name')
    parser.add_argument('--iters', type=int, default=100000, help='number of iters')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--task', type=str, default='baseline', help='baseline or calibration')

    args = parser.parse_args()


device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
x_max = np.array([27., 54., 3.])
x_min = np.array([-27., 0., 0.])
X_max = torch.tensor(x_max, dtype=torch.float).to(device)
X_min = torch.tensor(x_min, dtype=torch.float).to(device)

a1 = 2.0465e-4 # to define the specific heat capacity
b1 = 3.8091e-1 # to define the specific heat capacity
a2 = 1.6702e-5 # to define the thermal conductivity
b2 = 5.5228e-3 # to define the thermal conductivity
hc = 0.0519 / 15 # scaling factor

Rboltz = 5.6704e-14
emiss = 0.72
rho = 8.19e-3

thickness = 0.1

T_ref = 310.
T_range = 1000.

net = FNN([3, 64, 64, 64, 1], nn.Tanh(), in_tf=input_transform, out_tf=output_transform)
net.to(device)

if args.task == 'baseline':
    ha = 0. # convection coefficient for baseline
    hb = 2e-5 # convection coefficient for baseline
    a1 = torch.tensor([6.256e-2], requires_grad=True, device=device)
    b1 = torch.tensor([0.4], requires_grad=True, device=device)
    inv_params = []
else:
    ha = torch.tensor([0.], requires_grad=True, device=device) # convection coefficient for non-baseline
    hb = torch.tensor([1e-5], requires_grad=True, device=device) # convection coefficient for non-baseline
    a1 = torch.tensor([2.0465e-3], requires_grad=True, device=device)
    b1 = torch.tensor([0.4], requires_grad=True, device=device)
    inv_params = [a1]

iterations = args.iters

point_sets, flags = generate_points([], [])
point_sets, flags = load_data(point_sets, flags)

lr = args.lr
info_num = 100
w = [1., 1e-4, 1., 1e-4]

##validation data
data = np.load('Final1.npy').T
ind = (data[:, 2] > 2.85) * (data[:, 2] < 3.0) * (data[:, 1] > 3)
data = data[ind, :]
test_in = torch.tensor(data[:, 0:3], requires_grad=False, dtype=torch.float).to(device)
test_out = torch.tensor(data[:, 3:4], requires_grad=False, dtype=torch.float).to(device)




l_history, err_history = train2D(net, PDE, BC, point_sets, flags, iterations, lr=lr, info_num=100,
                                 test_in=test_in, test_out=test_out, w=w,
                                 inv_params=inv_params)


torch.save(net.state_dict(), '../model/2D{}.pt'.format(args.task))
np.save('../model/2D{}.npy'.format(args.task), l_history)
np.save('../model/2D{}_err.npy'.format(args.task), err_history)
