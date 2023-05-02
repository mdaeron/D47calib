"""
Generate, combine, display and apply Δ47 calibrations

.. include:: ../../docpages/calibs.md
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2023 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__docformat__ = "restructuredtext"
__date__      = '2023-04-30'
__version__   = '0.1.0'


import ogls as _ogls
import numpy as _np
from scipy.linalg import block_diag as _block_diag
from scipy.interpolate import interp1d as _interp1d


class D47calib(_ogls.InverseTPolynomial):

	def __init__(self,
		samples, T, D47,
		sT = None, sD47 = None,
		degrees = [0,2], xpower = 2,
		color = (.5,.5,.5),
		label = '',
		description = '',
		regress_now = False,
		**kw):

		self.samples = samples[:]
		self.color = color
		self.label = label
		self.description = description


		self.D47 = _np.asarray(D47, dtype = 'float')
		self.N = self.D47.size

		if sD47 is None:
			self.sD47 = _np.zeros((self.N, self.N))
		else:
			self.sD47 = _np.asarray(sD47)
			if len(self.sD47.shape) == 1:
				self.sD47 = _np.diag(self.sD47**2)
			elif len(self.sD47.shape) == 0:
				self.sD47 = _np.eye(self.D47.size) * self.sD47**2

		_ogls.InverseTPolynomial.__init__(self, T=T, Y=D47, sT=sT, sY=sD47, degrees = degrees, xpower = xpower, **kw)
		
		if regress_now:
			self.regress()
		
		self.bff_deriv = lambda x: _np.array([k * self.bfp[f'a{k}'] * x**(k-1) for k in degrees if k > 0]).sum(axis = 0)
		
		xi = _np.linspace(0,200**-1,1001)
		self.inv_bff = _interp1d(self.bff(xi), xi)

		self.D47_from_T = lambda T: self.bff((T+273.15)**-1)
		self.T_from_D47 = lambda D47: self.inv_bff(D47)**-1 - 273.15
		self.D47_from_T_deriv = lambda T: -(T+273.15)**-2 * self.bff_deriv((T+273.15)**-1)
		self.T_from_D47_deriv = lambda D47: self.D47_from_T_deriv(self.T_from_D47(D47))**-1
		
	def invT_xaxis(self, ylabel = 'Δ$_{47}$ (‰)', **kw):
		return _ogls.InverseTPolynomial.invT_xaxis(self, ylabel = ylabel, **kw)


	def plot_data(self, label = False, **kw):
		if 'mec' not in kw:
			kw['mec'] = self.color
		if label is not False:
			kw['label'] = self.label if label is True else label
		return _ogls.InverseTPolynomial.plot_data(self, **kw)


	def plot_error_bars(self, **kw):
		if 'ecolor' not in kw:
			kw['ecolor'] = self.color
		return _ogls.InverseTPolynomial.plot_error_bars(self, **kw)


	def plot_error_ellipses(self, **kw):
		if 'ec' not in kw:
			kw['ec'] = self.color
		return _ogls.InverseTPolynomial.plot_error_ellipses(self, **kw)


	def plot_bff(self, label = False, **kw):
		if 'color' not in kw:
			kw['color'] = self.color
		if label is not False:
			kw['label'] = self.label if label is True else label
		return _ogls.InverseTPolynomial.plot_bff(self, **kw)


	def plot_bff_ci(self, **kw):
		if 'color' not in kw:
			kw['color'] = self.color
		return _ogls.InverseTPolynomial.plot_bff_ci(self, **kw)

	def new_T47(self,
		D47 = None,
		sD47 = None,
		T=None,
		sT = None,
		error_from = 'both',
		return_covar = False,
		):
		'''
		Convert from Δ47 to T (if `D47` is specified as input)
		or from T to Δ47 (if `T` is specified as input).

		Only one of either `D47` or `T` may be specified.

		`T` input may be specified as a scalar, or as a 1-D array.
		`D47` output will then have the same type as `T`.

		`D47` input may be specified as a scalar, or as a 1-D array.
		`T` output will then have the same type as `D47`.
		
		**Arguments:**		

		* `D47`: Δ47 value(s) to convert into temperature (`float` or 1-D array)
		* `sD47`: Δ47 uncertainties, which may be:
		  - `None` (default)
		  - `float` or `int`
		  - 1-D array (standard errors on `D47`)
		  - 2-D array (covariance matrix for `D47`)
		* `T`: T value(s) to convert into Δ47 (`float` or 1-D array), in degrees C
		* `sT`: T uncertainties, which may be:
		  - `None` (default)
		  - `float` or `int`
		  - 1-D array (standard errors on `T`)
		  - 2-D array (variance-covariance matrix for `T`)
		* `error_from`: if set to `'both'` (default), returned errors take into account
		  input uncertainties (`sT` or `sD47`) as well as calibration uncertainties;
		  if set to `'calib'`, only calibration uncertainties are accounted for;
		  if set to `'sT'` or `'sD47'`, calibration uncertainties are ignored
		* `return_covar`: (False by default) whether to return the full covariance matrix
		  for returned `T` or `D47` values, otherwise return standard errors for the returned
		  `T` or `D47` values instead.
		  
		**Returns (if `D47` was specified):**
		
		* `T` (temperatures computed from `D47`)
		* `sT` (uncertainties on `T`, whether as standard errors or covariance matrix)

		**Returns (if `T` was specified):**
		
		* `D47` (temperatures computed from `D47`)
		* `sD47` (uncertainties on `D47`, whether as standard errors or covariance matrix)
		'''

		if D47 is None and T is None:
			raise ValueError('Either D47 or T must be specified, but both are undefined.')

		if D47 is not None and T is not None:
			raise ValueError('Either D47 or T must be specified, but not both.')

		if T is not None:
			
			D47 = self.D47_from_T(T)
			Np = len(self.degrees)
			N = D47.size

			### Compute covariance matrix of (*bfp, *T):
			CM = _np.zeros((Np+N, Np+N))

			if error_from in ['calib', 'both']:
				CM[:Np, :Np] = self.bfp_CM[:,:]

			if (sT is not None) and error_from in ['sT', 'both']:
				_sT = _np.asarray(sT)
				if _sT.ndim == 0:
					for k in range(N):
						CM[Np+k, Np+k] = _sT**2
				elif _sT.ndim == 1:
					for k in range(N):
						CM[Np+k, Np+k] = _sT[k]**2
				elif _sT.ndim == 2:
					CM[-N:, -N:] = _sT[:,:]

			### Compute Jacobian of D47(T) relative to (*bfp, *T):
			_T = _np.asarray(T)
			if _T.ndim == 0:
				_T = _np.expand_dims(_T, 0)
			J = _np.zeros((N, Np+N))
			if (sT is not None) and error_from in ['sT', 'both']:
				for k in range(N):
					J[k, Np+k] = (self.D47_from_T(_T[k] + .01) - self.D47_from_T(_T[k] - .01)) / .02
			if error_from in ['calib', 'both']:
				for k in range(Np):
					p1 = {_: self.bfp[_] for _ in self.bfp}
					p1[f'a{self.degrees[k]}'] += 0.001 * self.bfp_CM[k,k]**.5
					p2 = {_: self.bfp[_] for _ in self.bfp}
					p2[f'a{self.degrees[k]}'] -= 0.001 * self.bfp_CM[k,k]**.5
					J[:, k] = (self.model_fun(p1, (_T+273.15)**-1) - self.model_fun(p2, (_T+273.15)**-1)) / (0.002 * self.bfp_CM[k,k]**.5)

			### Error propagation:
			CM_D47 = J @ CM @ J.T

			if return_covar:
				return D47, CM_D47
			else:
				return D47, _np.diag(CM_D47)**.5

		if D47 is not None:
			# todo: implement new_T47(D47)
			pass


	def T47(self, D47 = None, sD47 = None, T=None, sT = None):
		if D47 is not None:
			if T is not None:
				raise ValueError("both D47 and T are specified.")
			T = self.T_from_D47(D47)
			local_deriv = self.T_from_D47_deriv(D47)
			if sD47 is None:
				sT = None
			else:
				sT = abs(sD47 * local_deriv)
			if isinstance(T, _np.ndarray):
				sTcalib = abs(self.bff_se((T+273.15)**-1) * local_deriv)
				return T, sT, sTcalib
			else:
				sTcalib = abs(self.bff_se(_np.array([(T+273.15)**-1])) * local_deriv)
				return T, sT, float(sTcalib)
		else:
			if T is None:
				raise ValueError("neither D47 nor T are specified.")
			D47 = self.D47_from_T(T)
			local_deriv = self.D47_from_T_deriv(T)
			if sT is None:
				sD47 = None
			else:
				sD47 = abs(sT * local_deriv)
			if isinstance(D47, _np.ndarray):
				sD47calib = self.bff_se((T+273.15)**-1)
				return D47, sD47, sD47calib
			else:
				sD47calib = self.bff_se(_np.array([(T+273.15)**-1]))
				return D47, sD47, float(sD47calib)
	

	def export(self, name, filename):
		with open(filename, 'w') as f:
			f.write(f'''
{name} = D47calib(
	samples = {self.samples},
	T = {list(self.T)},
	D47 = {list(self.D47)},
	sT = {[list(l) for l in self.sT]},
	sD47 = {[list(l) for l in self.sD47]},
	description = {repr(self.description)},
	label = {repr(self.label)},
	color = {self.color},
	degrees = {list(self.degrees)},
	bfp = {self.bfp},
	bfp_CM = {[list(l) for l in self.bfp_CM]},
	chisq = {self.chisq},
	Nf = {self.Nf},
	)
''')

	def plot_T47_errors(
		self,
		calibname = None,
		rD47 = 0.010,
		Nr = [2,4,8,12,20],
		Tmin = 0,
		Tmax = 120,
		colors = [(1,0,0),(1,.5,0),(.25,.75,0),(0,.5,1),(0.5,0.5,0.5)],
		yscale = 'lin',
		):

		from matplotlib import pyplot as ppl

		if calibname is None:
			calibname = self.label

		Nr = _np.array(Nr)
		if len(colors) < Nr.size:
			print('WARNING: Too few colors to plot different numbers of replicates; generating new colors.')
			from colorsys import hsv_to_rgb
			hsv = [(x*1.0/Nr.size, 1, .9) for x in range(Nr.size)]
			colors = [hsv_to_rgb(*x) for x in hsv]

		Ti = _np.linspace(Tmin, Tmax)
		D47, sD47, sD47_calib  = self.T47(T = Ti)

		ymax, ymin = 0, 1e6
		for N,c in zip(Nr, colors):
			sD47 = rD47 / N**.5
			T, sT, sT_calib = self.T47(D47 = D47, sD47 = sD47)
			ppl.plot(Ti, sT, '-', color = c, label=f'Analytical SE for {N} replicates')
			ymin = min(ymin, min(sT))
			ymax = max(ymax, max(sT))
		
		ppl.plot(Ti, sT_calib, 'k--', label='Independent SE from calibration alone')

		ppl.legend(fontsize=9)
		ppl.xlabel("T (°C)")

		ppl.ylabel("Standard error on reconstructed T (°C)")

		# yticks([0,.5,1,1.5,2])
		ppl.title(f"{calibname},\nassuming external Δ$_{{47}}$ repeatability of {rD47:.3f} ‰", size = 9)
		ppl.grid( alpha = .25)
		if yscale == 'lin':
			ppl.axis([Ti[0], Ti[-1], 0, ymax*1.05])
			t1, t2 = self.T.min(), self.T.max()
			ppl.plot([t1, t2], [0, 0], 'k-', alpha = .25, lw = 8, solid_capstyle = 'butt', clip_on = False)
			ppl.text((t1+t2)/2, 0, 'range of observations\n', alpha = .4, size = 7, ha = 'center', va = 'bottom', style = 'italic')
		else:
			ymin /= 2
			ppl.axis([Ti[0], Ti[-1], ymin, ymax*1.05])
			ppl.yscale('log')
			t1, t2 = self.T.min(), self.T.max()
			ppl.plot([t1, t2], [ymin, ymin], 'k-', alpha = .25, lw = 8, solid_capstyle = 'butt', clip_on = False)
			ppl.text((t1+t2)/2, ymin, 'range of observations\n', alpha = .4, size = 7, ha = 'center', va = 'bottom', style = 'italic')

	def export_data(self, csvfile, sep = ',', label = False, T_correl = False, D47_correl = False):
		'''
		Write calibration data to a csv file.
		'''
		n = len(str(self.N))

		with open(csvfile, 'w') as f:
			f.write(sep.join(['ID', 'Sample', 'T', 'SE_T', 'D47', 'SE_D47']))

			if label:
				f.write(f'{sep}Dataset')

			if T_correl:
				inv_diag_sT = _np.diag(_np.diag(self.sT)**-.5)
				Tcorrel = inv_diag_sT @ self.sT @ inv_diag_sT
				f.write(sep.join(['']+[f'Tcorrel_{k+1:0{n}d}' for k in range(self.N)]))

			if D47_correl:
				inv_diag_sD47 = _np.diag(_np.diag(self.sD47)**-.5)
				D47correl = inv_diag_sD47 @ self.sD47 @ inv_diag_sD47
				f.write(sep.join(['']+[f'D47correl_{k+1:0{n}d}' for k in range(self.N)]))

			for k, (s, T, sT, D47, sD47) in enumerate(zip(
				self.samples,
				self.T,
				_np.diag(self.sT)**.5,
				self.D47,
				_np.diag(self.sD47)**.5,
				)):
				f.write('\n' + sep.join([f'{k+1:0{n}d}', s, f'{T:.2f}', f'{sT:.2f}', f'{D47:.4f}', f'{sD47:.4f}']))
				if label:
					f.write(f'{sep}{self.label}')
				if T_correl:
					f.write(sep.join(['']+[
						f'{Tcorrel[k,_]:.0f}'
						if f'{Tcorrel[k,_]:.6f}'[-6:] == '000000'
						else f'{Tcorrel[k,_]:.6f}'
						for _ in range(self.N)]))
				if D47_correl:
					f.write(sep.join(['']+[
						f'{D47correl[k,_]:.0f}'
						if f'{D47correl[k,_]:.6f}'[-6:] == '000000'
						else f'{D47correl[k,_]:.6f}'
						for _ in range(self.N)]))
				



def combine_D47calibs(calibs, degrees = [0,2], same_T = []):
	'''
	Combine `D47calib` instances.
	
	Example of valid `same_T` parameter:
	`[{'LGB-2', DHC-2-8'}, {'ETH-1-1100-SAM', 'ETH-1-1100'}]`.
	'''

	samples = [s for c in calibs for s in c.samples]
	T = [t for c in calibs for t in c.T]
	D47 = [x for c in calibs for x in c.D47]
	sD47 = _block_diag(*[c.sD47 for c in calibs])
	sT = _block_diag(*[c.sT for c in calibs])

	for i in range(len(samples)):
		for j in range(len(samples)):
			if i != j:
				if samples[i] == samples[j] or {samples[i], samples[j]} in same_T:
					sT[i,j] = (sT[i,i] * sT[j,j])**.5
	
	calib = D47calib(
		samples = samples,
		T = T,
		D47 = D47,
		sT = sT,
		sD47 = sD47,
		degrees = degrees,
		regress_now = True,
		)

	return calib

from ._calibs import *