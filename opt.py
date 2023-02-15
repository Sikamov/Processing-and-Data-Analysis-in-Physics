from collections import namedtuple
import numpy as np

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""
#Ньютон-Гаусс
def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
 x=np.array(x0, dtype=np.float)
 cost=[]
 i=0
 while True:
  i += 1
  jac=j(*x)
  K=np.dot(jac.T, jac)
  d=y-f(*x)
  cost.append(0.5*np.dot(d, d))
  b=np.dot(jac.T, d)
  dx=np.linalg.solve(K, b)
  x += k*dx
  if np.linalg.norm(dx)<=tol:
   break
 gradnorm = np.linalg.norm(np.dot(jac.T, d))
 cost = np.array(cost)
 return Result(nfev=i, cost=cost, gradnorm=gradnorm, x=x)

#Левенберг-Марквардт
def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
 x=np.array(x0, dtype=np.float)
 lmbd=lmbd0
 cost=[]
 i=0
 dx=1e+10
 while True:
  i += 2
  jac=j(*x)
  K0= np.dot(jac.T, jac)+lmbd*np.identity(jac[1, :].size)
  Knu = np.dot(jac.T, jac)+lmbd/nu*np.identity(jac[1, :].size)
  d=y-f(*x)
  cost.append(0.5*np.dot(d, d))
  b=np.dot(jac.T, d)
  dx0=np.linalg.solve(K0, b)
  dxnu=np.linalg.solve(Knu, b)
  Fi=lambda t: 0.5*np.linalg.norm(y - f(*(x + t)))**2
  if Fi(dxnu)<= 0.5*np.dot(d, d):
    lmbd=lmbd/nu
    dx=dxnu
    x += dx
  elif Fi(dxnu)<=Fi(dx0):
    dx=dx0
    x += dx
  else:
   lmbd = nu*lmbd
  if np.linalg.norm(dx)<=tol:
    break
 gradnorm = np.linalg.norm(np.dot(jac.T, d))
 cost = np.array(cost)
 return Result(nfev=i, cost=cost, gradnorm=gradnorm, x=x)