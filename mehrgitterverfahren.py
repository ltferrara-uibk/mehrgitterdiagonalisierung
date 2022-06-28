# Mehrgitterverfahren fur die Warmeleitungsgleichung

import numpy as np
import numpy.linalg as la
import numpy.random as rand
# PIL - Python Imaging Library
import PIL.Image as Image
import sys
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
	print("Usage: " + sys.argv[0] + " <filename.png>")
	sys.argv = sys.argv + ["Gitarrenkorper.png"]
	print("Assuming default file " + sys.argv[1])
	print()

# Gitarrenkorper finden
image = Image.open(sys.argv[1])
korper_RGB = np.asarray(image, dtype = np.int64)
korper = np.zeros(korper_RGB.shape[0:2], dtype = np.int64)
if korper.shape != (512, 512):
	print("Warning: Image is not 512x512 pixels.")
	print()

background_color = korper_RGB[0, 0, :]
#print("background_color", background_color)

korper = la.norm(korper_RGB - background_color, axis = 2) >= 10
pixel_count = 0
pixel_count = sum(sum(korper))

print("Body:")
print("#pixels", korper.shape[0] * korper.shape[1])
print("#true  ", pixel_count)
print("ratio", pixel_count / (korper.shape[0] * korper.shape[1]))
print()

# Mehrgitterverfahren

def plot(u, title = ' '):
	plt.imshow(u)
	plt.colorbar()
	plt.title(title)
	plt.show()

# prolongation
# (v, h) -> (w, h)
def prolongation (v, h):
	# mehr raumpunkte
	w = np.zeros(tuple(np.array(v.shape) * 2))
	w[0::2, 0::2] = v[::1, ::1]
	w[1::2, 0::2] = v[::1, ::1]
	w[0::2, 1::2] = v[::1, ::1]
	w[1::2, 1::2] = v[::1, ::1]
	return w, (1/2) * h

def clamp(n, smallest, largest):
	return max(smallest, min(n, largest))

# Wendet den Laplace-Operator auf eine Funktion an.
def L (v, h):
	# h : gitterpunktabstand (euklidisch)
	# H : gitterpunktabstand (in der matrix)
	H = int(h * korper.shape[0])
	# N : seitenlange der matrix (angenommen N*N)
	N = v.shape[0]
	# w = Laplace v
	w = np.zeros(v.shape)
	# Verwende Dirichlet-Randbedingungen:
	# Randpunkte werden auf 0 gesetzt.
	w[0:N  , 0:N  ] = -4 * v[0:N, 0:N]
	w[0:N  , 0:N-1] = w[0:N  , 0:N-1] + v[0:N  , 1:N  ]
	w[0:N  , 1:N  ] = w[0:N  , 1:N  ] + v[0:N  , 0:N-1]
	w[0:N-1, 0:N  ] = w[0:N-1, 0:N  ] + v[1:N  , 0:N  ]
	w[1:N  , 0:N  ] = w[1:N  , 0:N  ] + v[0:N-1, 0:N  ]

	#w = 1 / h**2 * w # schrittweite berucksichtigen
	return w

# Glattung
# v : Feld
# h : Gitterpunktabstand
def Glattung (v, h):
	# Tatsachlich hangt die Glattung nicht vom Gitterpunktabstand
	# ab. siehe:

	# lambda_N ~ lambda_1 * N^2 ~ 1/(N*h)^2 * N^2
	# ~ 1 / h^2, N ... v.shape[0]
	# dt * lambda_N ~ 1
	# dt = 1 / lambda_N ~ h^2
	# return v + dt * Laplace(v)
	# return v + Laplace(v, h = 1)

	return v + 1/8 * L(v, 1)

def Restriktion (v, h = None, do_return_h = False):
	# weniger raumpunkte
	w = np.zeros((int(v.shape[0] / 2), int(v.shape[1] / 2)))

	w[::1, ::1] = ( 0
	+ v[ ::2,  ::2]
	+ v[1::2,  ::2]
	+ v[ ::2, 1::2]
	+ v[1::2, 1::2]
	)
	if do_return_h:
		return w, 2 * h
	else:
		return w

def Gram_Schmidt (Basis, v):
	if len(Basis) == 0:
		return v
	#print("Gram_Schmidt", len(Basis))
	#print(Basis[0].shape[0], v.shape[0])
	while Basis[0].shape[0] > v.shape[0]:
		Basis = [Restriktion(Basis[k]) for k in range(len(Basis))]
	#print()

	for k in range(len(Basis)):
		#inner = np.inner(Basis[k], v) # v ist matrix
		inner = np.einsum('ij,ij', Basis[k], v)
		inner_b = np.einsum('ij,ij', Basis[k], Basis[k])
		inner = inner / inner_b
		#print("k", k, "inner Nr.1", inner)
		v = v - inner * Basis[k]
		#inner = np.einsum('ij,ij', Basis[k], v)
		#print("k", k, "inner Nr.2", inner)
	return v

def normieren (u):
	norm = la.norm(u, ord = 'fro')
	if norm < 0.00001:
		u = u + 0.001 + rand.random(u.shape)
		norm = la.norm(u, ord = 'fro')
	return (1 / norm) * u.shape[0] * u.shape[1] * u

# Feld mit korper schneiden
def schneiden (korper, v):
	T = int(korper.shape[0] / v.shape[0])
	# T ... schrittweite
	v[::1, ::1] = v[::1, ::1] * korper[::T, ::T]
	return v

def Mehrgitterdiagonalisierung (korper, K = 5):
	# K ... Anzahl Eigenvektoren
	K = K
	# N ... Anzahl Raumpunkte (Seitenlange)
	N = 128
	# Basis ... Orthonormalbasis
	Basis = np.zeros((K, N, N))
	# k ... Arbeitsindex
	k = 0

	for k in range(K):
		n = 2 ** int(np.ceil(np.log2(k + 1)))
		h = 1/n
		u = rand.random((n, n))

		while n < N:
			n = 2 * n
			u, h = prolongation(u, h)

			for l in range(96*(k+1)):
				u = normieren(u)
				u = schneiden(korper, u)
				u = Gram_Schmidt(Basis[:k], u)
				u = Glattung(u, h)
				#if k < 3 and n >= 8 and l == 0:
				#	plt.subplot(3, 5, int(5*k + np.log2(n)-3 + 1))
				#	plt.imshow(u)
				#	plt.axis('off')

		Basis[k] = u
		k = k + 1

	#plt.show()

	return Basis

Basis = Mehrgitterdiagonalisierung(korper, 15)

print("Basis", len(Basis))

K = len(Basis)
for k in range(K):
	plt.subplot(3, 5, k + 1)
	plt.imshow(np.ma.masked_array(Basis[k], mask = Restriktion(Restriktion(korper == False))))
	plt.axis('off')
plt.savefig(sys.argv[1].replace(".png", "_moden.png"))
print()
print("Saved to " + sys.argv[1].replace(".png", "_moden.png"))
plt.show()

K = len(Basis)
for k in range(K):
	print("k", k, np.einsum('ij,ij', Basis[k], L(Basis[k], 1)) / np.einsum('ij,ij', Basis[k], Basis[k]))
plt.plot([np.einsum('ij,ij', Basis[k], L(Basis[k], 1)) / np.einsum('ij,ij', Basis[k], Basis[k]) for k in range(K)])
plt.title("Eigenwerte")
plt.xlabel("k")
plt.ylabel("lambda_k")
plt.savefig(sys.argv[1].replace(".png", "_eigenwerte.png"))
print()
print("Saved to " + sys.argv[1].replace(".png", "_eigenwerte.png"))
plt.show()

