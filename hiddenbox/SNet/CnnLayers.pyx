import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
	np.float32_t
	np.float64_t

@cython.boundscheck(False)
def conv2dForward(np.ndarray[DTYPE_t, ndim=4] out, 
	np.ndarray[DTYPE_t, ndim=4] padX,
	np.ndarray[DTYPE_t, ndim=4] w,
	np.ndarray[DTYPE_t, ndim=1] b, int stride):
	# - out: Output data, of shape (N, F, H', W') where H' and W' are given by
	#   H' = 1 + (H - HH) / stride
	#   W' = 1 + (W - WW) / stride
	# - x: Input data of shape (N, C, H, W)
	# - w: Filter weights of shape (F, C, HH, WW)
	# - b: Biases, of shape (F,)

	cdef int N = padX.shape[0]
	cdef int C = padX.shape[1]
	cdef int H = padX.shape[2]
	cdef int W = padX.shape[3]
	cdef int F = w.shape[0]
	cdef int HH = w.shape[2]
	cdef int WW = w.shape[3]
	cdef int Hpai = 1 + (H - HH) / stride
	cdef int Wpai = 1 + (W - HH) / stride
	for n in range(N):
		padXn = np.reshape(padX[n], (C, H, W ))
		for f in range(F):
			wf = np.reshape(w[f], (C, HH, WW))
			bf = np.reshape(b[f], (1))
			for i in range(Hpai):
				si = i * stride
				for j in range(Wpai):
					sj = j * stride
					out[n,f,i,j] = np.sum(padXn[:,si:si+HH, sj:sj+WW] * wf ) + bf

def conv2dBackward(
	np.ndarray[DTYPE_t, ndim=4] padX,
	np.ndarray[DTYPE_t, ndim=4] w,
	np.ndarray[DTYPE_t, ndim=1] b,
	np.ndarray[DTYPE_t, ndim=4] dout,
	int pad, int stride):

	cdef int N = padX.shape[0]
	cdef int C = padX.shape[1]
	cdef int H = padX.shape[2]
	cdef int W = padX.shape[3]
	cdef int F = w.shape[0]
	cdef int HH = w.shape[2]
	cdef int WW = w.shape[3]
	cdef int Hpai = dout.shape[2]
	cdef int Wpai = dout.shape[3]

	dxPad = np.zeros_like(padX)
	dw = np.zeros_like(w)

	sii = [(i,stride*i) for i in range(Hpai)]
	jii = [(j,stride*j) for j in range(Wpai)]
	
	for n in range(N):
		for f in range(F):
			for i,si in sii:
				for j,sj in jii:
					dxPad[n,:,si:si+HH, sj:sj+WW] += w[f] * dout[n,f,i,j]
					dw += padX[n,:,si:si+HH, sj:sj+WW] * dout[n,f,i,j]
	dw = dw / F
	db = np.sum(dout, (0,2,3))
	dx = dxPad if pad == 0 else dxPad[:,:,pad:-pad,pad:-pad]
		

	return dx, dw, db
