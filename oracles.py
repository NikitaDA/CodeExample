import numpy as np
import scipy


from scipy.misc import logsumexp
from scipy.sparse import csr_matrix
from scipy.special import expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        k = self.X.shape[0]
        if len(w.shape) == 1:
            log = np.logaddexp(0, -self.y * self.X.dot(w))
            loss = log.sum(axis=0) / k + (self.l2_coef * np.dot(w, w) / 2)
        else:
            alphas = self.X.dot(w.T) - np.amax(self.X.dot(w.T), axis=1)[:, np.newaxis]
            softmax = np.exp(alphas) / np.sum(np.exp(alphas), axis=1)[:, np.newaxis]
            loss = -np.log(softmax[range(k), self.y]).sum() / k + (self.l2_coef * np.sum(np.diag(w.dot(w.T))) / 2)
        return loss

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        k = self.X.shape[0]
        if len(w.shape) == 1:
            if isinstance(self.X, csr_matrix):
                vec = - self.X.multiply(self.y[:, np.newaxis]).multiply(expit(- self.y * self.X.dot(w))[:, np.newaxis])
                grad = np.array(vec.sum(axis=0))[0] / k + self.l2_coef * w
            else:
                vec = - self.y[:, np.newaxis] * (self.X * expit(- self.y * self.X.dot(w))[:, np.newaxis])
                grad = vec.sum(axis=0) / k + self.l2_coef * w
        else:
            alphas = self.X.dot(w.T) - np.amax(self.X.dot(w.T), axis=1)[:, np.newaxis]
            softmax = np.exp(alphas) / np.sum(np.exp(alphas), axis=1)[:, np.newaxis]
            mask = np.zeros((self.class_number, k), dtype=bool)
            mask[np.unique(self.y)] = self.y == np.unique(self.y)[:, np.newaxis]
            if isinstance(self.X, csr_matrix):
                c1 = (self.X.T * softmax).T
                c = (c1 - self.X.T.dot(mask.T).T) / k
            else:
                c = ((self.X[:, :, np.newaxis] * softmax[:, np.newaxis, :]).sum(axis=0).T - self.X.T.dot(mask.T).T) / k
            grad = c + self.l2_coef * w
        return grad

        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        
        self.X = X
        self.y = y
        return super().func(w)
        
    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        self.X = X
        self.y = y
        return super().grad(w)
    
    
class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """
    
    def __init__(self, class_number=3, l2_coef=1):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.l2_coef = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        self.X = X
        self.y = y
        return super().func(w)
        
    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        self.X = X
        self.y = y
        return super().grad(w)
