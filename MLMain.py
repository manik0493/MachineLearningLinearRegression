import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import display, HTML

init_notebook_mode(connected=True)

from plotly.grid_objs import Grid, Column
import time


class training_data_set:
    x_input = float()
    y_output = float()


def start():
    training_set_x_list = list()
    training_set_y_list = list()
    training_set_list = list()

    number_of_training_set = input("Please enter number of training sets: ")

    for i in range(0,int(number_of_training_set)):
        input_training_data_point = input("Enter the "+str(i+1)+" input value: ")
        output_training_data_point = input("Enter the "+str(i+1)+" output value: ")
        training_data_set_object = training_data_set()
        training_data_set_object.x_input = float(input_training_data_point)
        training_data_set_object.y_output = float(output_training_data_point)
        training_set_list.append(training_data_set_object)

    print("\nYour training set:\n")

    for data_point in training_set_list:
        training_set_x_list.append(data_point.x_input)
        training_set_y_list.append(data_point.y_output)
        print("input: "+str(data_point.x_input)+" | output : " + str(data_point.y_output))

    theta0 = float(input("\nEnter the first assumption value of theta 0 : "))   #b
    theta1 = float(input("\nEnter the first assumption value of theta 1 : "))   #m
    alpha = float(input("\nEnter the value of alpha(Learning Rate) for minimizing function : "))
    number_of_convergence_iterations = input("\nEnter the number of iterations to run batch gradient descent algorithm: ")

    # y = theta0 + theta1 * x

    print("\nCalculating minimum thetas...\n")
    plotting_frame_list = list()
    columnlist = list()
    ylist = list()
    theta1_list=list()
    theta0_list=list()

    for J_theta_minimize_iteration in range(int(number_of_convergence_iterations)):
        output_gradient_theta0 = 0
        output_gradient_theta1 = 0
        i = 0

        for i in range(int(number_of_training_set)):

            # d/d theta0 J(theta0) = summation(0-m) ([theta0 + theta * x(i)] - y(i))
            output_gradient_theta0 += ((theta0 + (theta1 * training_set_list[i].x_input)) - training_set_list[i].y_output)

            # d/d theta1 J(theta1) = summation(0-m) ([theta0 + theta * x(i)] - y(i)) * x(i)
            output_gradient_theta1 += ((theta0 + (theta1 * training_set_list[i].x_input)) - training_set_list[i].y_output)\
                                      * training_set_list[i].x_input

        # J(theta) = theta(i) + alpha *d/d0((hx-y))^2
        new_theta0 = theta0 - (alpha * output_gradient_theta0)
        new_theta1 = theta1 - (alpha * output_gradient_theta1)

        i = 0
        ylist = list()
        for i in range(int(number_of_training_set)):
            ylist.append(new_theta0 + new_theta1*training_set_list[i].x_input)
        columnlist.append(ylist)
        theta1_list.append(new_theta1)
        theta0_list.append(new_theta0)
        theta0 = new_theta0
        theta1 = new_theta1

    i = 0

    for i in range(int(number_of_convergence_iterations)):
        plotting_frame_list.append({
            'data': [{'x': training_set_x_list, 'y': columnlist[i]}]
        })

    figure = {'data':[{'x':training_set_x_list,'y':training_set_y_list },{'x':training_set_x_list,'y':training_set_y_list,'mode':'markers','marker':{'color':'red'}}],
              'layout': {'title': 'ML Graph','hovermode':'closest',
                         'updatemenus': [{
                             'type': 'buttons',
                             'buttons': [
                                 {'args': [None],
                                  'label': 'Play',
                                  'method': 'animate'}
                             ]

                         }]},
              'frames':list(plotting_frame_list)
              }

    plot(figure, filename='basic-line.html')
    figure2 = {'data':[{'x':theta0_list,'y':theta1_list}]}
    plot(figure2,filename='theta_graph.html')

if __name__ == "__main__":
    start()

