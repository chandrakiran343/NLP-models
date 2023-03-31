import numpy as np
import random
import torch
import csv
# g = torch.Generator(696969)
weights = torch.randn((6))
impweights = torch.tensor(
    [901.2316,  696.1120,  312.4160, 1099.0256,  901.2116,  901.2444])
alpha = 0.4

file = open("realestate.csv", "r")

reader = csv.reader(file)
lines = list(reader)
lines.pop(0)
costs = []
for line in lines[:2]:
    costs.append(float(line[7]))
lines = [[float(x) for x in line[1:len(line) - 1]] for line in lines[:2]]
parameters = torch.tensor(lines)



# rip = parameters / parameters.sum(1, keepdim=True)

prices = torch.tensor(costs)
print(parameters)
print(weights)
prediction = parameters @ weights

prediction = prediction / prediction.sum()

print(prediction)
# # parameters = rip
# for iteration in range(100):
#     cumilative_error = 0
#     for row in range(len(prices)):
#         input = parameters[row]
#         goal = prices[row]

#         prediction = input.dot(weights)
#         # prediction = sum(wi*xi for wi,xi in zip(input,weights))
#         # error = (prediction - goal) ** 2
#         # error = torch.log(prediction)
#         # cumilative_error += error

#         # delta = prediction - goal
#         # weights = weights - (alpha * (input * delta))

#         print("Prediction: " + str(prediction))
#     # print("Weights: " + str(weights))
#     print("Error: " + str(cumilative_error))

# print(prices)

# print(impweights.dot(torch.tensor([2013.333,10.8,252.5822,1,24.9746,121.53046]) / parameters.sum(0)))