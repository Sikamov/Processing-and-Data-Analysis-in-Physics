
import numpy as np
import scipy
import json
from numpy import math
from scipy import integrate
import matplotlib.pyplot as plt
from opt import gauss_newton
from opt import lm

#f-целевая функция
def f(z, omega, H0):
 f=np.empty(z.shape[0])
 i=0
 for i in range(z.shape[0]):
    f[i]=5*np.log10((3*10**11/H0) * (1+z[i]) * scipy.integrate.quad(lambda x:((1-omega)*(1+x)**3+omega)**(-1/2), 0, z[i])[0])-5
 return f

#j-якобиан
def j(z, omega, H0):
 j=np.empty((z.shape[0], 2))
 i=0
 for i in range(z.shape[0]):
  j[i,0]=-(5/np.log(10)) * (scipy.integrate.quad(lambda x:(1-(1+x)** 3)/(((1-omega)*(1+x)**3+omega)**(3/2)), 0, z[i])[0] / scipy.integrate.quad(lambda x:((1-omega)*(1+x)**3+omega)**(-1/2), 0, z[i])[0])
 j[:,1]=-5/(H0 * np.log(10))
 return j

#mu,z-измеренные данные
data=np.loadtxt(fname='jla_mub.txt')
z=data[:, 0]
mu=data[:, 1]


#Ньтон-Гаусс
result=gauss_newton(y=mu, f=lambda *x:f(z,*x), j=lambda *x:j(z,*x), x0=np.array([0.5, 50]))
print(result)


#Параметры H0 и omega для НГ
omega1=result[3][0]
H01=result[3][1]
mu_1=np.empty(z.shape[0])
for i in range(z.shape[0]):
 mu_1[i]=5*np.log10((3*10**11/H01)*(1+z[i])* scipy.integrate.quad(lambda x:((1-omega1)*(1+x)**3+omega1)**(-1/2), 0, z[i])[0])-5


#Итерации НГ
nfev1=result[0]
count1=np.empty(nfev1)
cost1=np.empty(nfev1)
i=0
for i in range(nfev1):
 count1[i]=int(i+1)
 cost1 [i]=result[1][i]


#Левенберг-Марквардт
result=lm(y=mu, f=lambda *x:f(z,*x), j=lambda *x:j(z,*x), x0=np.array([0.5, 50]), lmbd0=1e-2, nu=2, tol=1e-4)
print(result)


#Параметры H0 и omega для ЛМ
omega2=result[3][0]
H02=result[3][1]
mu_2=np.empty(z.shape[0])
for i in range(z.shape[0]):
 mu_2[i]=5*np.log10((3*10**11/H02)*(1+z[i])* scipy.integrate.quad(lambda x:((1-omega2)*(1+x)**3+omega2)**(-1/2), 0, z[i])[0])-5


#Итерации ЛМ
nfev2=result[0]
m=nfev2//2
count2=np.empty(m)
cost2=np.empty(m)
i=0
for i in range(m):
 count2[i]=int(i+1)
 cost2 [i]=result[1][i]


#График ЛМ и НГ
fig, ax = plt.subplots(figsize=(10,5),dpi=200)
plt.title('mu(z)')
plt.xlabel('z')
plt.ylabel('mu')
ax.plot(z, mu_1, label='mu_1(z)-модель Ньютон-Гаусс', linewidth=6.0)
ax.plot(z, mu_2, label='mu_2(z)-модель Левенберг-Марквардт', linewidth=4.0)
ax.plot(z, mu, label='mu(z)-измеренные', linewidth=2.0)
plt.legend()
fig.savefig('mu-z.png')


#Функция потерь от итерационного шага 
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
plt.xlabel('Итерационный шаг')
plt.ylabel('Функция потерь')
plt.title('Функция потерь от итерационного шага')
ax.plot(count2, cost2, label='Левенберг-Марквардт', linewidth=4.0) 
ax.plot(count1, cost1, label='Ньютон-Гаусс', linewidth=2.0)
fig.savefig('cost.png')


#Сохраняем данные в json
dic = {
  "Gauss-Newton": {"H0": H01, "Omega": omega1, "nfev": nfev1},
  "Levenberg-Marquardt": {"H0": H02, "Omega": omega2, "nfev": nfev2}
}
with open('parameters.json', 'w') as file:
    json.dump(dic, file, indent=4)