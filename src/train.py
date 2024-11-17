import torch
import numpy as np
import torch
import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import torch.nn as nn
import time

def loss(f,target=None):
    if target == None:
        return torch.sum(torch.square(f))/f.shape[0]
    if isinstance(target,float):
        return torch.sum(torch.square(f-target))/f.shape[0]
    else:
        return nn.MSELoss()(f,target)

def rmse(pred, actual):
    return torch.sqrt(torch.mean((pred - actual)**2))

def train(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100,
         test_in = None, test_out=None,w=[1.,1.,1.,1.],inv_params=[]):
    
    if inv_params == []:
        params = net.parameters()
    else:
        params = (list(net.parameters())+inv_params)
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_bc = 0
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    if test_in != None:
        err_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],points[:,3:4],net,flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],points[:,3:4],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        
        if n_data == 0:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE)/3 #weighted
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE,elapsed))
                    start_time = time.time()
                                        
        else:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE+w[3]*l_data)/4 #weighted
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data,Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data, elapsed))
                    start_time = time.time()
                
                if inv_params!=[]:
                    for value in inv_params:
                        print(value.item())
                    
            
        cost.backward() 
        optimizer.step()
    
    return l_history,err_history


def train2D(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100,
         test_in = None, test_out=None,w=[1.,1.,1.,1.],inv_params=None):
    
    if inv_params == None:
        params = net.parameters()
    else:
        params = (list(net.parameters())+inv_params)
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_bc = 0
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    if test_in != None:
        err_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],net,flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        
        if n_data == 0:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE)/3 #weighted
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    # Compute RMSE between T_pred and T_actual
                    rMSE = rmse(T_pred, test_out).item()
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Test: %.3e, RMSE: %.3e, Time: %.2f'
                          % (epoch, cost, l_BC, l_IC, l_PDE, Test_err, rMSE, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE,elapsed))
                    start_time = time.time()
                                        
        else:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE+w[3]*l_data)/4 #weighted
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    # Compute RMSE between T_pred and T_actual
                    rMSE = rmse(T_pred, test_out).item()
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Test: %.3e, RMSE: %.3e, Time: %.2f'
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data,Test_err, rMSE, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data, elapsed))
                    start_time = time.time()
                
                if inv_params!=[]:
                    for value in inv_params:
                        print(value.item())
                    
            
        cost.backward() 
        optimizer.step()
        

    return l_history,err_history


import numpy as np
import torch
import time

import numpy as np
import torch
import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting


import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def train2DTurgut(net, PDE, BC, point_sets, flags, iterations=50000, lr=5e-4, info_num=100,
                  test_in=None, test_out=None, w=[1., 1., 1., 1.], inv_params=None, actual_temps=None,
                  cycle_name='default1'):

    if inv_params is None:
        params = net.parameters()
    else:
        params = (list(net.parameters()) + inv_params)

    optimizer = torch.optim.Adam(params, lr=lr)

    n_bc = 0
    n_ic = 0
    n_PDE = 0
    n_data = 0
    for points, flag in zip(point_sets, flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]

    start_time = time.time()

    l_history = []
    err_history = [] if test_in is not None else None

    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0

        for points, flag in zip(point_sets, flags):
            if flag[0] == 'BC':
                f = BC(points[:, 0:1], points[:, 1:2], points[:, 2:3], net, flag[1])
                l_BC += loss(f) * points.shape[0] / n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred, flag[1]) * points.shape[0] / n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred, flag[1]) * points.shape[0] / n_data
            if flag[0] == 'domain':
                f = PDE(points[:, 0:1], points[:, 1:2], points[:, 2:3], net)
                l_PDE += loss(f) * points.shape[0] / n_PDE

        if n_data == 0:
            cost = (w[0] * l_BC + w[1] * l_IC + w[2] * l_PDE) / 3  # weighted
            l_history.append([cost.item(), l_BC.item(), l_IC.item(), l_PDE.item()])
        else:
            cost = (w[0] * l_BC + w[1] * l_IC + w[2] * l_PDE + w[3] * l_data) / 4  # weighted
            l_history.append([cost.item(), l_BC.item(), l_IC.item(), l_PDE.item(), l_data.item()])

        if epoch % info_num == 0:
            if test_in is not None:
                T_pred = net(test_in)
                Test_err = loss(T_pred, test_out)
                err_history.append(Test_err.item())
                elapsed = time.time() - start_time
                # Compute RMSE between T_pred and T_actual
                rMSE = rmse(T_pred, test_out).item()
                print(f'It: {epoch}, Loss: {cost.item():.3e}, BC: {l_BC.item():.3e}, IC: {l_IC.item():.3e}, '
                      f'PDE: {l_PDE.item():.3e}, Test: {Test_err.item():.3e}, RMSE: {rMSE:.3e}, Time: {elapsed:.2f}')
                start_time = time.time()
            else:
                elapsed = time.time() - start_time
                print(f'It: {epoch}, Loss: {cost.item():.3e}, BC: {l_BC.item():.3e}, IC: {l_IC.item():.3e}, '
                      f'PDE: {l_PDE.item():.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

        cost.backward()
        optimizer.step()

    # After training, generate full-field temperature predictions at t=1, t=2, t=3
    # Create a grid for the full field (55x55 pixels)
    x = np.linspace(0, 1, 55)  # Modify the range based on your domain
    y = np.linspace(0, 1, 55)  # Modify the range based on your domain
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Generate predictions for t=1, t=2, t=3
    t_steps = [1, 2, 3]
    predictions = {}

    for t in t_steps:
        # Convert the grid to a tensor
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

        # If your model expects time as input, concatenate the time value
        grid_tensor_with_time = torch.cat([grid_tensor, torch.full((grid_tensor.shape[0], 1), float(t))], dim=1)

        # Get the model prediction for the full field
        full_field_pred = net(grid_tensor_with_time).detach().numpy()  # Convert to numpy array
        predictions[f't={t}'] = full_field_pred.reshape(55, 55)  # Reshape to 55x55 grid

        # Visualization and comparison with actual temperature
        if actual_temps is not None:
            actual_temp = actual_temps[f't={t}']  # Get the actual temperature for the given time step
            predicted_temps_reshaped = full_field_pred.reshape(55, 55)

            plt.figure(figsize=(8, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Predicted Temperature at time = {t} s")
            plt.imshow(predicted_temps_reshaped, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title(f"Actual Temperature at time = {t} s")
            plt.imshow(actual_temp, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.suptitle(f"Cycle: {cycle_name}, Epoch: {epoch + 1}")
            plt.savefig(f"{cycle_name}_epoch_{epoch + 1}_time_{t}.png")  # Save plot with cycle and epoch info
            plt.close()

            # Optionally: Compute the error (predicted - actual) for the given time step
            error = predicted_temps_reshaped - actual_temp
            plt.figure(figsize=(6, 6))
            plt.title(f"Error (Predicted - Actual) at time = {t} s")
            plt.imshow(error, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.suptitle(f"Cycle: {cycle_name}, Epoch: {epoch + 1}")
            plt.savefig(f"{cycle_name}_epoch_{epoch + 1}_error_time_{t}.png")  # Save error plot
            plt.close()

    # Save the predictions as a .npy file
    np.save(f"{cycle_name}_full_field_temperature_predictions.npy", predictions)

    return l_history, err_history


def train2DnoBC(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100,
         test_in = None, test_out=None,w=[1.,1.,1.,1.],inv_params=None):
    
    if inv_params == None:
        params = net.parameters()
    else:
        params = (list(net.parameters())+inv_params)
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    if test_in != None:
        err_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],net,flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        
        if n_data == 0:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE)/3 #weighted
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE,elapsed))
                    start_time = time.time()
                                        
        else:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE+w[3]*l_data)/4 #weighted
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data,Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data, elapsed))
                    start_time = time.time()
                
                if inv_params!=[]:
                    for value in inv_params:
                        print(value.item())
                    
            
        cost.backward() 
        optimizer.step()
        

    return l_history,err_history