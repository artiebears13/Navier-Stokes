import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Navier:
    def __init__(self, N, eps, nu):
        '''

        :param N: size of mesh for one direction (square mesh)
        :param eps: error
        :param nu:  1/Re (Re - Reynolds number)
        '''
        self.A = np.zeros((N * N, N * N))
        self.eps = eps
        self.h = 1 / N
        self.N = N
        # self.nu = 8.9 * (pow(10, -4))
        self.nu = nu
        for i in range(N):
            for j in range(N):
                ij = i * N + j
                counter = 0
                if ij > N - 1:
                    self.A[ij, ij - N] = -1
                    counter += 1
                if ij % N != 0:
                    self.A[ij, ij - 1] = -1
                    counter += 1
                if (ij + 1) % N != 0:
                    self.A[ij, ij + 1] = -1
                    counter += 1
                if ij < N * (N - 1):
                    self.A[ij, ij + N] = -1
                    counter += 1
                self.A[ij, ij] = counter
        self.u_prev = np.zeros((N, N + 1))
        # self.u_prev[0] = np.ones(N + 1)
        # print(self.u_prev)
        self.v_prev = np.zeros((N + 1, N))
        self.p_prev = np.zeros(N * N)
        # self.u = np.zeros((N, N + 1))
        # self.v = np.zeros((N + 1, N))
        # self.p = np.zeros(N * N)
        self.b = np.zeros(N * N)

    def solve_P(self, b):
        sol = np.linalg.solve(self.A, b)
        # print(sol)
        return sol

    def div(self, u, v, dt):
        b = np.zeros(self.N * self.N)
        # print(f'u: {u.shape}')
        # print(f'v: {v.shape}')
        # print(f'p: {p.shape}')
        for i in range(self.N):
            for j in range(self.N):
                # print(f'i: {i}  j: {j}')
                b[i * self.N + j] = -self.h * (u[i, j + 1] - u[i, j] + v[i + 1, j] - v[i, j])

        return b

    def solveUV(self, p, dt):
        '''
         du/dt = (u*grad)*u - nu (laplace(u)) + grad(p)

        TWO STEPS:
        du/dt = u*(du/dx)+v*(du/dy) - nu*(d2u/dx2+d2u/dy2) + px
        dv/dt = u*(dv/dx)+v*(dv/dy) - nu*(d2v/dx2+d2v/dy2) + px

        :param p:  pressure on previous iteration
        :param dt:  step by time
        :return:  u - x-velocity component ,v - y-velocity component
        '''
        N = self.N
        u = np.zeros((N, N + 1))
        v = np.zeros((N + 1, N))

        for i in range(self.N):  # i = [0,N)
            for j in range(1, self.N):  #

                uw = self.u_prev[i, j - 1]
                uc = self.u_prev[i, j]
                ue = self.u_prev[i, j + 1]
                if i == 0:  # upper boundary
                    un = 2 - uc
                else:
                    un = self.u_prev[i - 1, j]
                if i == (self.N - 1):  # lower boundary
                    # print('i1:   ',i)
                    us = -uc
                else:

                    us = self.u_prev[i + 1, j]
                vnw = self.v_prev[i, j - 1]
                vne = self.v_prev[i, j]
                vsw = self.v_prev[i + 1, j - 1]
                vse = self.v_prev[i + 1, j]
                # print(f'i: {i}, j: {j}')
                pe = p[i * N + j]
                pw = p[i * N + j - 1]

                gradU = 0.25 / self.h * (
                        (uc + ue) * (uc + ue) - (uw + uc) * (uw + uc) - (vnw + vne) * (un + uc) + (
                        vsw + vse) * (us + uc)
                )

                u[i, j] = uc - dt * (
                        gradU + self.nu / (self.h ** 2) * (4 * uc - uw - ue - us - un) + (pe - pw) / self.h)

        # --------------------------------------------------------------------------------

        for i in range(1, N):
            for j in range(N):
                vc = self.v_prev[i, j]
                vn = self.v_prev[i - 1, j]
                vs = self.v_prev[i + 1, j]
                if j == N - 1:  # right boundary
                    ve = -vc
                else:
                    ve = self.v_prev[i, j + 1]
                if j == 0:  # left boundary
                    vw = -vc
                else:
                    vw = self.v_prev[i, j - 1]
                une = self.u_prev[i - 1, j + 1]
                unw = self.u_prev[i - 1, j]
                use = self.u_prev[i, j + 1]
                usw = self.u_prev[i, j]
                pn = p[(i - 1) * N + j]
                ps = p[i * N + j]
                gradV = 0.25 / self.h * (
                        (une + use) * (ve + vc) - (unw + usw) * (vc + vw) - ((vn + vc)) ** 2 + (
                    (vs + vc)) ** 2
                )
                v[i, j] = vc - dt * (gradV + self.nu / self.h ** 2 * (4 * vc - vw - ve - vs - vn) + (ps - pn) / self.h)

        return u, v

    def norm_u(self, u):
        norm = 0
        for i in range(self.N):
            for j in range(self.N):
                if np.abs(u[i, j] - self.u_prev[i, j]) > norm:
                    norm = np.abs(u[i, j] * 10000 - self.u_prev[i, j] * 10000)
        return norm

    def solver(self, dt, stop=0.01):
        '''

        :param dt: step by time
        :param stop: difference of velocity on two time layers (system equilibrium condition)
        :return:
        '''
        N = self.N
        eps = self.eps
        t = 0
        cont = True
        self.u_prev = np.zeros((N, N + 1))
        # self.u_prev[0] = np.ones(N + 1)
        # print(self.u_prev)
        self.v_prev = np.zeros((N + 1, N))
        self.p_prev = np.zeros(N * N)

        # while t < T+dt/2:
        while cont and t < 30:  # limit for t
            b_check = False
            # print(t)
            norm = self.eps + 1
            p = self.p_prev
            # while not b_check:
            it = -1
            while norm > eps:
                it += 1
                # b_check = True
                u, v = self.solveUV(p, dt)
                b = self.div(u, v, dt)
                print('t: ', t, ' iter: ', it, ' norm: ', np.linalg.norm(b), ' norm_p', np.mean(p))
                # print(p)
                norm = np.linalg.norm(b)
                if norm > self.eps:
                    p_correction = self.solve_P(b / dt)
                    p = p + p_correction * self.h
                    # print('p_corr: ', np.mean(p_correction))
                    p -= np.mean(p)
                    # print('----------------------------------')
                    # print(p)
                    # print('-------------------------------------')
                    # print(p_correction)

                else:
                    b_check = True
            # print('t: ', t, ' delta_u: ', np.linalg.norm(u - self.u_prev), ' delta_v: ', np.linalg.norm(v - self.v_prev, 2))
            if np.linalg.norm(u - self.u_prev) < stop and np.linalg.norm(v - self.v_prev) < stop and t > 1:
                cont = False
            self.u_prev = u
            self.v_prev = v
            self.p_prev = p

            t += dt
        clear_output()
        self.plot_solution(self.u_prev, self.v_prev, self.p_prev, dt)  # break
        print('t: ', t)
        # t+=

    def plot_solution(self, u, v, p, dt, streamplot=True):

        u = (u[:, :-1] + u[:, 1:]) / 2
        v = (v[1:, :] + v[:-1, :]) / 2
        N = self.N

        print(u.shape)
        u = u[::-1, ::]
        v = -v[::-1, ::]
        p = p.reshape((N, N))[::-1, ::]
        x = np.arange(self.h / 2, 1, self.h)
        y = np.arange(self.h / 2, 1, self.h)
        grid_x, grid_y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.streamplot(grid_x, grid_y, u, v, color='black')
        plt.contourf(grid_x, grid_y, p.reshape((self.N, self.N)))
        plt.title(f"N = {self.N}, Eps = {self.eps}, Nu = {self.nu}, dt = {dt}", fontsize=20)
        plt.savefig('output.png')


A = Navier(N=16, eps=0.01, nu=0.01)
A.solver( 0.01, stop=0.01)
# print('u:',A.u_prev)
# print('v: ',A.v_prev)
# print(A.p_prev)
# print(A.A)
