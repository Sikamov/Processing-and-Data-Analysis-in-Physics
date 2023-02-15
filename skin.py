import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


n='Экран отсутствует'
m='No_lg'
d=0.008
mu=1.2566650*10**(-6)  
mu0=4*3.14*10**(-7)
u=mu/mu0
sigma=38000000

#без всего
with open('/Users/kirillsikamov/Desktop/ВШЭ/2 курс/Практикум Электричество и магнетизм/Лабораторная работа №4/No_lg.txt') as file: 
 img0 = np.array([[float(z) for z in x.split(';')] for x in file.readlines()], dtype='float64')
 x1, x2=[],[]
for i in range(len(img0)):
 x1.append(img0[i][0])
 x2.append(img0[i][1])
 
#экран
with open('/Users/kirillsikamov/Desktop/ВШЭ/2 курс/Практикум Электричество и магнетизм/Лабораторная работа №4/'+m+'.txt') as file: 
 img = np.array([[float(z) for z in x.split(';')] for x in file.readlines()], dtype='float64')
 y1, y2=[],[]
for i in range(len(img)):
 y1.append(img[i][0])
 y2.append(img[i][1])
 
#сигнал без экрана
a0 =[]
U0=3
for i in range(len(img0)):
 a0.append(U0*10**(x2[i]/20))

 
#сигнал с экраном
a =[]
U0=3
for i in range(len(img)):
 a.append(U0*10**(y2[i]/20))
plt.xscale('log')
plt.plot(y1, a, color='black', label=n)
plt.legend()
plt.ylabel('U1 (В)')
plt.xlabel('w (Гц)')
plt.grid()
plt.show()


#логарифм отношения
z =[]
for i in range(min(len(img),len(img0))):
 z.append((y2[i]-x2[i])/20)
 
plt.xscale('log')
plt.plot(x1, z, color='black', label='Log'+m)
plt.legend()
plt.ylabel('lg(U1/U) ')
plt.xlabel('w (Гц)')
plt.grid()
plt.show()

#квадрат логарифма отношения
z2 =[]
for i in range(min(len(img),len(img0))):
 z2.append(((y2[i]-x2[i])/20)**2)
 
plt.xscale('log')
plt.plot(x1, z2, color='black', label='Log^2'+m)
plt.legend()
plt.ylabel('lg^2(U1/U) ')
plt.xlabel('w (Гц)')
plt.grid()
plt.show()


#теория коэффицент поглащения
with open('/Users/kirillsikamov/Desktop/ВШЭ/2 курс/Практикум Электричество и магнетизм/Лабораторная работа №4/'+m+'.txt') as file: 
 img = np.array([[float(z) for z in x.split(';')] for x in file.readlines()], dtype='float64')
 k,w=[],[]
for i in range(len(img)):
 w.append(img[i][0])
 k.append(131.4*d*np.sqrt(w[i]*u*sigma))

plt.xscale('log')
plt.plot(w, k, color='y', label=n)
plt.legend()
plt.ylabel('K')
plt.xlabel('w (Гц)')
plt.grid()
plt.show()

#теория коэффицент поглащения
with open('/Users/kirillsikamov/Desktop/ВШЭ/2 курс/Практикум Электричество и магнетизм/Лабораторная работа №4/'+m+'.txt') as file: 
 img = np.array([[float(z) for z in x.split(';')] for x in file.readlines()], dtype='float64')
 k,w=[],[]
for i in range(len(img)):
 w.append(img[i][0])
 k.append(1/np.sqrt(w[i]*mu*sigma*3.14*10**(-6)))

plt.xscale('log')
plt.plot(w, k, color='black', label=n)
plt.legend()
plt.ylabel('d')
plt.xlabel('w (Гц)')
plt.grid()
plt.show()


