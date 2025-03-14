from calc_with_joblib import Calculator

c = Calculator(1.001)

x = c.getListsMP([i for i in range(35, 3500005)], 1)
print(x[:10])
print(x[-10:])
