import numpy as np
import scipy.signal as signal

class AdaptiveFilter:
    def __init__(self, b, a, k=None, update_func=None) -> None:
        self.b = b
        self.a = a
        self.y = []
        self.k = 1
        if k is not None:
            self.k = float(k)
        self.update_func = None
        self.num_taps = max(len(b), len(a))
        self.__reset_buffers()
        self.__update_coeffs(self.b, self.a)
        if update_func is not None:
            self.update_func = update_func

    def __reset_buffers(self):
        self.x_buffer = np.zeros(self.num_taps)
        self.y_buffer = np.zeros(self.num_taps)

    def __update_coeffs(self, b, a, k=None):
        b = np.array(b); a = np.array(a)
        assert max(len(b), len(a)) <= self.num_taps
        self.b = b
        self.a = a
        if len(b) < self.num_taps:
            self.b = np.append(b, np.zeros(self.num_taps - len(b)))
        if len(a) < self.num_taps:
            self.a = np.append(a, np.zeros(self.num_taps - len(a)))
        if k is not None:
            self.k = k

    def filter_sample(self, x):
        self.x_buffer = np.roll(self.x_buffer, -1)
        self.x_buffer[-1] = x
        self.y_buffer = np.roll(self.y_buffer, -1)
        self.y_buffer[-1] = 0
        y = np.dot(self.b, np.flip(self.x_buffer)) - np.dot(self.a, np.flip(self.y_buffer))
        self.y_buffer[-1] = y
        return y

    def forward_filter(self, x):
        # Identical to lfilterbut with adaptive capability
        for xi in x:
            self.y.append(self.filter_sample(xi))
            if self.update_func is not None:
                self.__update_coeffs(*self.update_func(y=self.y[-1], x=xi))
        self.__reset_buffers()
        return self.y

    def forward_backward_filter(self, x, zero_phase=False):
        # Similar to filtfilt
        y_fwd = self.forward_filter(x)
        self.y = y_fwd
        if zero_phase:
            # Forward - Backward Filter
            # Known issues with zero-initial conditions
            padlen = len(self.b)
            input = np.flip(y_fwd)
            input = np.pad(input, (padlen, 0), mode='constant')
            y_bwd = np.flip(self.forward_filter(input))
            self.y = y_bwd[:-padlen]
        return self.y

    def lfilter(self, x, **kwargs):
        return self.k*signal.lfilter(self.b, self.a, x, *kwargs)
    
    def filtfilt(self, x):
        return self.k*signal.filtfilt(self.b, self.a, x)


class FilterRLS(AdaptiveFilter):

    def __init__(self, d, N_fir, lambda_=1, eps=0.5, **kwargs) -> None:
        self.N_fir = N_fir
        self.lambda_ = lambda_
        self.d = d  # desired input
        self.d_next = 0
        self.d_buffer = np.zeros(N_fir)
        self.e = [] # y - d
        self.f = np.zeros(shape=(N_fir, 1))  # input buffer
        self.kman = np.zeros(shape=(N_fir, 1))  # kalman gain
        self.R = np.eye(N_fir) / eps
        self.b = np.zeros(self.N_fir)
        self.a = [1]
        self.num_taps = max(len(self.b), len(self.a))
        super(FilterRLS, self).__init__(self.b, self.a, update_func=self.adapt)
        self.update_func = self.adapt
        if'error_filter' in kwargs:
            if kwargs['error_filter'] == 'triangle':
                # Triangle filter error buffer
                self.ew = np.arange(self.N_fir, 0, -1)
                self.ew = self.ew/np.sum(self.ew)
            assert self.ew.shape == (self.N_fir,)
            assert round(np.sum(self.ew), 8) == 1
    
    # step1 and step2 covered in parent filter
    # i.e. buffer input and apply filters
    
    def step3_error_calc(self):
        self.d_buffer = np.roll(self.d_buffer, -1)
        self.d_buffer[-1] = self.d[self.d_next]
        self.d_next += 1
        e_buffer = self.d_buffer - self.y_buffer
        e = e_buffer[-1]
        if hasattr(self, 'ew'):
            e = np.dot(self.ew, np.flip(e_buffer))
        self.e.append(e)
    
    def step4_kalman_gain_update(self):
        self.kman = (self.R @ self.x_buffer.T[:,None])/(self.lambda_ + self.x_buffer[None,:] @ self.R @ self.x_buffer.T[:,None])
        assert self.kman.shape == (self.N_fir, 1)
    
    def step5_covariance_update(self):
        self.R = 1/self.lambda_ * (self.R - self.kman @ self.x_buffer[None,:] @ self.R)
        assert self.R.shape == (self.N_fir, self.N_fir)
    
    def step6_fir_update_func(self):
        self.b = self.b + (self.kman*self.e[-1]).flatten()
        assert self.b.shape == (self.N_fir, )
    
    def adapt(self, **kwargs):
        if not "y" in kwargs:
            raise Exception("Filter output not found")
        assert self.x_buffer.shape == (self.N_fir,)
        assert self.y_buffer.shape == (self.N_fir,)
        # need to flip because buffer fills from back
        self.x_buffer = np.flip(self.x_buffer)
        self.step3_error_calc()
        self.step4_kalman_gain_update()
        self.step5_covariance_update()
        self.step6_fir_update_func()
        # undo the flip before forward filter continues
        self.x_buffer = np.flip(self.x_buffer)
        return self.b, self.a
    
    def run(self, f):
        y = self.forward_filter(f)
        # f     : input signal + noise
        # d     : noise ref
        # y     : FIR estimate of noise ref (d) from input
        # f-y   : input - noise est
        # d-y   : noise - noise est
        # e     : error term
        return y, f - y, self.e
