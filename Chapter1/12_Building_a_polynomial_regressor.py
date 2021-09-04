import numpy as np

Time=np.array([1, 2, 3, 4, 5, 6])
Temp=np.array([12, 13.8, 16.5, 16, 14, 13])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(Time, Temp, 'bo')
plt.xlabel("Time")
plt.ylabel("Temp")
plt.title('Temperature / time')
plt.show()

beta=np.polyfit( Time, Temp, 2)

p=np.poly1d(beta)

xp=np.linspace(1,6,100)
plt.figure()
plt.plot(Time, Temp, 'bo', xp, p(xp), '-')#ve duong cong
plt.title('Result')
plt.show()