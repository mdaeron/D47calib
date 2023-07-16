"""
Generate, combine, display and apply Δ47 calibrations

.. include:: ../../docpages/calibs.md
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2023 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
# __docformat__ = "restructuredtext"
__date__      = '2023-04-30'
__version__   = '0.1.0'


import typer
import ogls as _ogls
import numpy as _np
from scipy.linalg import block_diag as _block_diag
from scipy.interpolate import interp1d as _interp1d
from matplotlib import pyplot as _ppl


class D47calib(_ogls.InverseTPolynomial):
	"""
	Δ47 calibration class based on OGLS regression
	of Δ47 as a polynomial function of inverse T.
	"""

	def __init__(self,
		samples, T, D47,
		sT = None,
		sD47 = None,
		degrees = [0,2],
		xpower = 2,
		name = '',
		label = '',
		description = '',
		**kwargs,
		):
		"""
		### Parameters
		
		+ **samples**: a list of N sample names.
		+ **T**: a 1-D array (or array-like) of temperatures values (in degrees C), of size N.
		+ **D47**: a 1-D array (or array-like) of Δ47 values (in permil), of size N.
		+ **sT**: uncertainties on `T`. If specified as:
		  + a scalar: `sT` is treated as the standard error applicable to all `T` values;
		  + a 1-D array-like of size N: `sT` is treated as the standard errors of `T`;
		  + a 2-D array-like of size (N, N): `sT` is treated as the (co)variance matrix of `T`.
		+ **sD47**: uncertainties on `D47`. If specified as:
		  + a scalar: `sD47` is treated as the standard error applicable to all `D47` values;
		  + a 1-D array-like of size N: `sD47` is treated as the standard errors of `D47`;
		  + a 2-D array-like of size (N, N): `sD47` is treated as the (co)variance matrix of `D47`.
		+ **degrees**: degrees of the polynomial regression, e.g., `[0, 2]` or `[0, 1, 2, 3, 4]`.
		+ **name**: a human-readable, short name assigned to the calibration.
		+ **label**: a short description of the calibration, e.g., to be used in legends.
		+ **description**: a longer description, including relevant references/DOIs.
		This is not necessary when `bfp` and `CM_bfp` are specified at instantiation time.
		+ **kwargs**: keyword arguments passed to the underlying `ogls.InverseTPolynomial()` call.
		
		### Notable attributes

		+ **N**:
		The total number of observations (samples) in the calibration data.
		+ **samples**:
		The list sample names.
		+ **T**:
		1-D `ndarray` of temperatures in degrees C.
		+ **D47**:
		1-D `ndarray` of Δ47 values in permil.
		+ **sT**:
		2-D `ndarray` equal to the full (co)variance matrix for `T`.
		+ **D47**:
		2-D `ndarray` equal to the full (co)variance matrix for `D47`.
		+ **xpower**:
		By default, all `D47calib` graphical methods plot Δ47 as a function of 1/T<sup>2</sup>.
		It is possible to change this behavior to use a different power of 1/T.
		This is done by redefining the `xpower` attribute to a different, non-zero `int` value
		(e.g. `foo.xpower = 1` to plot as a function of 1/T instead of 1/T<sup>2</sup>).
		+ **bfp**:
		The best-fit parameters of the regression.
		This is a `dict` with keys equal to the polynomial coefficients (see `bff` definition below)
		+ **bff()**:
		The best-fit polynomial function of inverse T, defined as:
		`bff(x) = sum(bfp[f'a{k}'] * x**k for k in degrees)`
		Note that `bff` takes `x = 1/(T+273.15)` (instead of `T`) as input.

		
		### Examples
		
		A very simple example:
		
		````py
		.. include:: ../../code_examples/D47calib_init/example.py
		````
		
		Should yield:

		````
		.. include:: ../../code_examples/D47calib_init/output.txt
		````
		
		"""

		self.samples = samples[:]
		self.name = name
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

		_ogls.InverseTPolynomial.__init__(self, T=T, Y=D47, sT=sT, sY=sD47, degrees = degrees, xpower = xpower, **kwargs)
		
		if self.bfp is None:
			self.regress()
		
		self._bff_deriv = lambda x: _np.array([k * self.bfp[f'a{k}'] * x**(k-1) for k in degrees if k > 0]).sum(axis = 0)
		
		xi = _np.linspace(0,200**-1,1001)
		self._inv_bff = _interp1d(self.bff(xi), xi)

		self._D47_from_T = lambda T: self.bff((T+273.15)**-1)
		self._T_from_D47 = lambda D47: self._inv_bff(D47)**-1 - 273.15
		self._D47_from_T_deriv = lambda T: -(T+273.15)**-2 * self._bff_deriv((T+273.15)**-1)
		self._T_from_D47_deriv = lambda D47: self._D47_from_T_deriv(self._T_from_D47(D47))**-1
	
	def __repr__(self):
		return f'<D47calib: {self.name}>'
		
	def invT_xaxis(self,
		xlabel = None,
		Ti = [0,20,50,100,250,1000],
		):
		"""
		Create and return an `Axes` object with X values equal to 1/T<sup>2</sup>,
		but labeled in degrees Celsius.
		
		### Parameters
		
		+ **xlabel**:
		Custom label for X axis (`r'$1\,/\,T^2$'` by default)
		+ **Ti**:
		Specify tick locations for X axis, in degrees C.

		### Returns

		+ an `matplotlib.axes.Axes` instance

		### Examples

		````py
		.. include:: ../../code_examples/D47calib_invT_xaxis/example_1.py
		````
		
		This should result in something like this:

		<img align="center" src="example_invT_xaxis_1.png">

		It is also possible to define the X axis using a different power of 1/T
		by first redefining the `xpower` attribute:
		
		````py
		.. include:: ../../code_examples/D47calib_invT_xaxis/example_2.py
		````
		
		This should result in something like this:

		<img align="center" src="example_invT_xaxis_2.png">
		"""
		if xlabel is None:
			xlabel = f'$1\\,/\\,T^{self.xpower}$' if self.xpower > 1 else '1/T'
		_ppl.xlabel(xlabel)
		_ppl.xticks([(273.15 + t) ** -self.xpower for t in sorted(Ti)[::-1]])
		ax = _ppl.gca()
		ax.set_xticklabels([f"${t}\\,$°C" for t in sorted(Ti)[::-1]])
		ax.tick_params(which="major")

		return ax
		

	def plot_data(self, label = False, **kwargs):
		"""
		Plot Δ47 value of each sample as a function of 1/T<sup>2</sup>.
		
		### Parameters
		
		+ **label**:
		  + If `label` is a string, use this string as `label` for the underlyig
		  `matplotlib.pyplot.plot()` call.
		  + If `label = True`, use the caller's `label` attribute instead.
		  + If `label = False`, no label is specified (default behavior).
		+ **kwargs**:
		keyword arguments passed to the underlying `matplotlib.pyplot.plot()` call.

		### Returns

		+ the return value(s) of the underlying `matplotlib.pyplot.plot()` call.

		### Example
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import huyghe_2022 as calib

		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis(Ti = [0,10,25])
		calib.plot_data(label = True)
		ppl.ylabel('$Δ_{47}$ (‰ I-CDES)')
		ppl.legend()
		ppl.savefig('example_plot_data.png', dpi = 100)
		`````

		This should result in something like this:

		<img align="center" src="example_plot_data.png">
		"""
# 		if 'mec' not in kwargs:
# 			kwargs['mec'] = self.color
		if label is not False:
			kwargs['label'] = self.label if label is True else label
		return _ogls.InverseTPolynomial.plot_data(self, **kwargs)


	def plot_error_bars(self, **kwargs):
		"""
		Plot Δ47 error bars (±1.96 SE) of each sample as a function of 1/T<sup>2</sup>.
		
		### Parameters
		
		+ **kwargs**:
		keyword arguments passed to the underlying `matplotlib.pyplot.errrobar()` call.

		### Returns

		+ the return value(s) of the underlying `matplotlib.pyplot.errorbar()` call.

		### Example
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import huyghe_2022 as calib

		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis(Ti = [0,10,25])
		calib.plot_error_bars(alpha = .4)
		calib.plot_data(label = True)
		ppl.ylabel('$Δ_{47}$ (‰ I-CDES)')
		ppl.legend()
		ppl.savefig('example_plot_error_bars.png', dpi = 100)
		`````

		This should result in something like this:

		<img align="center" src="example_plot_error_bars.png">
		"""
# 		if 'ecolor' not in kwargs:
# 			kwargs['ecolor'] = self.color
		return _ogls.InverseTPolynomial.plot_error_bars(self, **kwargs)


	def plot_error_ellipses(self, **kwargs):
		"""
		Plot Δ47 error ellipses (95 % confidence) of each sample as a function of 1/T<sup>2</sup>.
		
		### Parameters
		
		+ **kwargs**:
		keyword arguments passed to the underlying `matplotlib.patches.Ellipse()` call.

		### Returns

		+ the return value(s) of the underlying `matplotlib.patches.Ellipse()` call.

		### Example
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import huyghe_2022 as calib

		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis(Ti = [0,10,25])
		calib.plot_error_ellipses(alpha = .4)
		calib.plot_data(label = True)
		ppl.ylabel('$Δ_{47}$ (‰ I-CDES)')
		ppl.legend()
		ppl.savefig('example_plot_error_ellipses.png', dpi = 100)
		`````

		This should result in something like this:

		<img align="center" src="example_plot_error_ellipses.png">
		"""
# 		if 'ec' not in kwargs:
# 			kwargs['ec'] = self.color
		return _ogls.InverseTPolynomial.plot_error_ellipses(self, **kwargs)


	def plot_bff(self, label = False, **kwargs):
		"""
		Plot best-fit regression of Δ47 as a function of 1/T<sup>2</sup>.
		
		### Parameters
		
		+ **label**:
		  + If `label` is a string, use this string as `label` for the underlyig
		  `matplotlib.pyplot.plot()` call.
		  + If `label = True`, use the caller's `label` attribute instead.
		  + If `label = False`, no label is specified (default behavior).
		+ **kwargs**:
		keyword arguments passed to the underlying `matplotlib.pyplot.plot()` call.

		### Returns

		+ the return value(s) of the underlying `matplotlib.pyplot.plot()` call.

		### Example
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import huyghe_2022 as calib

		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis(Ti = [0,10,25])
		calib.plot_bff(label = True, dashes = (8,2,2,2))
		calib.plot_data()
		ppl.ylabel('$Δ_{47}$ (‰ I-CDES)')
		ppl.legend()
		ppl.savefig('example_plot_bff.png', dpi = 100)
		`````

		This should result in something like this:

		<img align="center" src="example_plot_bff.png">
		"""
# 		if 'color' not in kwargs:
# 			kwargs['color'] = self.color
		if label is not False:
			kwargs['label'] = self.label if label is True else label
		return _ogls.InverseTPolynomial.plot_bff(self, **kwargs)


	def plot_bff_ci(self, **kwargs):
		"""
		Plot 95 % confidence region for best-fit regression of Δ47 as a function of 1/T<sup>2</sup>.
		
		### Parameters
		
		+ **label**:
		+ **kwargs**:
		keyword arguments passed to the underlying `matplotlib.pyplot.fill_between()` call.

		### Returns

		+ the return value(s) of the underlying `matplotlib.pyplot.fill_between()` call.

		### Example
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import huyghe_2022 as calib

		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis(Ti = [0,10,25])
		calib.plot_bff_ci(alpha = .15)
		calib.plot_bff(label = True, dashes = (8,2,2,2))
		calib.plot_data()
		ppl.ylabel('$Δ_{47}$ (‰ I-CDES)')
		ppl.legend()
		ppl.savefig('example_plot_bff_ci.png', dpi = 100)
		`````

		This should result in something like this:

		<img align="center" src="example_plot_bff_ci.png">
		"""
# 		if 'color' not in kwargs:
# 			kwargs['color'] = self.color
		return _ogls.InverseTPolynomial.plot_bff_ci(self, **kwargs)

	def T47(self,
		D47 = None,
		sD47 = None,
		T=None,
		sT = None,
		error_from = 'both',
		return_covar = False,
		):
		'''
		When `D47` is input, computes corresponding T value(s).
		`D47` input may be specified as a scalar, or as a 1-D array.
		`T` output will then have the same type and size as `D47`.

		When `T` is input, computes corresponding Δ47 value(s).
		`T` input may be specified as a scalar, or as a 1-D array.
		`D47` output will then have the same type and size as `T`.
		
		Only one of either `D47` or `T` may be specified as input.

		**Arguments:**		

		* `D47`: Δ47 value(s) to convert into temperature (`float` or 1-D array)
		* `sD47`: Δ47 uncertainties, which may be:
		  - `None` (default)
		  - `float` or `int` (uniform standard error on `D47`)
		  - 1-D array (standard errors on `D47`)
		  - 2-D array (covariance matrix for `D47`)
		* `T`: T value(s) to convert into Δ47 (`float` or 1-D array), in degrees C
		* `sT`: T uncertainties, which may be:
		  - `None` (default)
		  - `float` or `int` (uniform standard error on `T`)
		  - 1-D array (standard errors on `T`)
		  - 2-D array (variance-covariance matrix for `T`)
		* `error_from`: if set to `'both'` (default), returned errors take into account
		  input uncertainties (`sT` or `sD47`) as well as calibration uncertainties;
		  if set to `'calib'`, only calibration uncertainties are accounted for;
		  if set to `'sT'` or `'sD47'`, calibration uncertainties are ignored.
		* `return_covar`: (False by default) whether to return the full covariance matrix
		  for returned `T` or `D47` values, otherwise return standard errors for the returned
		  `T` or `D47` values instead.
		  
		**Returns (with `D47` input):**
		
		* `T`: temperature value(s) computed from `D47`
		* `sT`: uncertainties on `T` value(s), whether as standard error(s) or covariance matrix

		**Returns (with `T` input):**
		
		* `D47`: Δ47 value(s) computed from `D47`
		* `sD47`: uncertainties on `D47` value(s), whether as standard error(s) or covariance matrix

		### Example
		
		````py
		import numpy as np
		from matplotlib import pyplot as ppl
		from D47calib import combined_2023 as calib

		X = np.linspace(1473**-2, 270**-2)
		D47, sD47 = calib.T47(T = X**-0.5 - 273.15)
		
		fig = ppl.figure(figsize = (5,3))
		ppl.subplots_adjust(bottom = .25, left = .15)
		calib.invT_xaxis()
		ppl.plot(X, 1000 * sD47, 'r-')
		ppl.ylabel('Calibration SE on $Δ_{47}$ values (ppm)')
		ppl.savefig('example_SE47.png', dpi = 100)
		`````

		This should result in something like this:
		
		<img src="example_SE47.png">
		'''

		if D47 is None and T is None:
			raise ValueError('Either D47 or T must be specified, but both are undefined.')

		if D47 is not None and T is not None:
			raise ValueError('Either D47 or T must be specified, but not both.')

		if T is not None:
			
			D47 = self._D47_from_T(T)
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
					J[k, Np+k] = self._D47_from_T_deriv(_T[k])

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
				return D47, float(_np.diag(CM_D47)**.5) if D47.ndim == 0 else _np.diag(CM_D47)**.5

		if D47 is not None:

			T = self._T_from_D47(D47)
			Np = len(self.degrees)
			N = T.size

			### Compute covariance matrix of (*bfp, *T):
			CM = _np.zeros((Np+N, Np+N))

			if error_from in ['calib', 'both']:
				CM[:Np, :Np] = self.bfp_CM[:,:]

			if (sD47 is not None) and error_from in ['sD47', 'both']:
				_sD47 = _np.asarray(sD47)
				if _sD47.ndim == 0:
					for k in range(N):
						CM[Np+k, Np+k] = _sD47**2
				elif _sD47.ndim == 1:
					for k in range(N):
						CM[Np+k, Np+k] = _sD47[k]**2
				elif _sD47.ndim == 2:
					CM[-N:, -N:] = _sD47[:,:]

			### Compute Jacobian of T(D47) relative to (*bfp, *D47):
			_D47 = _np.asarray(D47)
			if _D47.ndim == 0:
				_D47 = _np.expand_dims(_D47, 0)
			J = _np.zeros((N, Np+N))
			if (sD47 is not None) and error_from in ['sD47', 'both']:
				for k in range(N):
					J[k, Np+k] = self._T_from_D47_deriv(_D47[k])
			if error_from in ['calib', 'both']:
				
				xi = _np.linspace(0,200**-1,1001)[1:]
				for k in range(Np):
				
					p1 = {_: self.bfp[_] for _ in self.bfp}
					p1[f'a{self.degrees[k]}'] += 0.001 * self.bfp_CM[k,k]**.5
					T_from_D47_p1 = _interp1d(self.model_fun(p1, xi), xi**-1 - 273.15)

					p2 = {_: self.bfp[_] for _ in self.bfp}
					p2[f'a{self.degrees[k]}'] -= 0.001 * self.bfp_CM[k,k]**.5
					T_from_D47_p2 = _interp1d(self.model_fun(p2, xi), xi**-1 - 273.15)

					J[:, k] = (T_from_D47_p1(_D47) - T_from_D47_p2(_D47)) / (0.002 * self.bfp_CM[k,k]**.5)

			### Error propagation:
			CM_T = J @ CM @ J.T

			if return_covar:
				return T, CM_T
			else:
				return T, float(_np.diag(CM_T)**.5) if T.ndim == 0 else _np.diag(CM_T)**.5
	

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
		"""
		Plot SE of T reconstructed using the calibration as a function of T for various
		combinations of analytical precision and number of analytical replicates.

		**Arguments**		

		+ **calibname**:
		Which calibration name to display. By default, use `label` attribute.
		+ **rD47**:
		Analytical precision of a single analysis.
		+ **Nr**:
		A list of lines to plot, each corresponding to a given number of replicates.
		+ **Tmin**:
		Minimum T to plot.
		+ **Tmax**:
		Maximum T to plot.
		+ **colors**:
		A list of colors to distinguish the plotted lines.
		+ **yscale**:
		  + If `'lin'`, the Y axis uses a linear scale.
		  + If `'log'`, the Y axis uses a logarithmic scale.
		  
		**Example**
		
		````py
		from matplotlib import pyplot as ppl
		from D47calib import devils_laghetto_2023 as calib

		fig = ppl.figure(figsize = (3.5,4))
		ppl.subplots_adjust(bottom = .2, left = .15)
		calib.plot_T47_errors(
			calibname = 'Devils Laghetto calibration',
			Nr = [1,2,4,16],
			Tmin  =0,
			Tmax = 40,
			)
		ppl.savefig('example_SE_T.png', dpi = 100)
		````

		This should result in something like this:
		
		<img src="example_SE_T.png">
		"""

		if calibname is None:
			calibname = self.label

		Nr = _np.array(Nr)
		if len(colors) < Nr.size:
			print('WARNING: Too few colors to plot different numbers of replicates; generating new colors.')
			from colorsys import hsv_to_rgb
			hsv = [(x*1.0/Nr.size, 1, .9) for x in range(Nr.size)]
			colors = [hsv_to_rgb(*x) for x in hsv]

		Ti = _np.linspace(Tmin, Tmax)
		D47i, _  = self.T47(T = Ti)
		_, sT_calib = self.T47(D47 = D47i, error_from = 'calib')

		ymax, ymin = 0, 1e6
		for N,c in zip(Nr, colors):
			_, sT = self.T47(D47 = D47i, sD47 = rD47 / N**.5, error_from = 'sD47')
			_ppl.plot(Ti, sT, '-', color = c, label=f'SE for {N} replicate{"s" if N > 1 else ""}')
			ymin = min(ymin, min(sT))
			ymax = max(ymax, max(sT))
		
		_ppl.plot(Ti, sT_calib, 'k--', label='SE from calibration')

		_ppl.legend(fontsize=9)
		_ppl.xlabel("T (°C)")

		_ppl.ylabel("Standard error on reconstructed T (°C)")

		# yticks([0,.5,1,1.5,2])
		_ppl.title(f"{calibname},\nassuming external Δ$_{{47}}$ repeatability of {rD47:.3f} ‰", size = 9)
		_ppl.grid( alpha = .25)
		if yscale == 'lin':
			_ppl.axis([Ti[0], Ti[-1], 0, ymax*1.05])
			t1, t2 = self.T.min(), self.T.max()
			_ppl.plot([t1, t2], [0, 0], 'k-', alpha = .25, lw = 8, solid_capstyle = 'butt', clip_on = False)
			_ppl.text((t1+t2)/2, 0, 'range of observations\n', alpha = .4, size = 7, ha = 'center', va = 'bottom', style = 'italic')
			_ppl.axis([None, None, None, _ppl.axis()[-1]*1.25])
		elif yscale == 'log':
			ymin /= 2
			_ppl.axis([Ti[0], Ti[-1], ymin, ymax*1.05])
			_ppl.yscale('log')
			t1, t2 = self.T.min(), self.T.max()
			_ppl.plot([t1, t2], [ymin, ymin], 'k-', alpha = .25, lw = 8, solid_capstyle = 'butt', clip_on = False)
			_ppl.text((t1+t2)/2, ymin, 'range of observations\n', alpha = .4, size = 7, ha = 'center', va = 'bottom', style = 'italic')

	def export_data(self, csvfile, sep = ',', label = False, T_correl = False, D47_correl = False):
		"""
		Write calibration data to a csv file.
		
		### Parameters
		
		+ **csvfile**:
		The filename to write data to.
		+ **sep**:
		The separator between CSV fields.
		+ **label**:
		  + If specified as `True`, include a `Dataset` column with the calibration's `label` attribute.
		  + If specified as a `str`, include a `Dataset` column with that string.
		  + If specified as `False`, do not include a `Dataset` column.
		+ **T_correl**:
		  + If `True`, include correlations between all `T` values.
		+ **D47_correl**:
		  + If `True`, include correlations between all `D47` values.
		
		### Example

		````py
		D47calib.huyghe_2022.export_data(
			csvfile = 'example_export_data.csv',
			T_correl = True,
			D47_correl = True,
			)
		````

		This should result in something like this ([link](example_export_data.csv)):
		
		.. include:: ../../docs/example_export_data.md

		"""
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
					if label is True:
						f.write(f'{sep}{self.label}')
					else:
						f.write(f'{sep}{label}')
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
				

	def export(self, name, filename):
		"""
		Save `D47calib` object as an importable file.
		
		### Parameters
		
		+ **name**:
		The name of the variable to export.
		+ **filename**:
		The filename to write to.
		
		### Example

		````py
		D47calib.anderson_2021_lsce.export('foo', 'bar.py')
		````

		This should result in a `bar.py` file with the following contents:
		
		````py
		foo = D47calib(
			samples = ['LGB-2', 'DVH-2'],
			T = [7.9, 33.7],
			D47 = [0.6485720997671647, 0.5695972909966959],
			sT = [[0.04000000000000001, 0.0], [0.0, 0.04000000000000001]],
			sD47 = [[8.72797097773764e-06, 2.951894073404263e-06], [2.9518940734042614e-06, 7.498611746762038e-06]],
			description = 'Devils Hole & Laghetto Basso from Anderson et al. (2021), processed in I-CDES',
			label = 'Slow-growing calcites from Anderson et al. (2021)',
			color = (0, 0.5, 0),
			degrees = [0, 2],
			bfp = {'a0': 0.1583220210575451, 'a2': 38724.41371782721},
			bfp_CM = [[0.00035908667755871876, -30.707016431538836], [-30.70701643153884, 2668091.396598919]],
			chisq = 6.421311854486162e-27,
			Nf = 0,
			)
		````
		"""
		with open(filename, 'w') as f:
			f.write(f'''
{name} = D47calib(
	samples = {self.samples},
	T = {list(self.T)},
	D47 = {list(self.D47)},
	sT = {[list(l) for l in self.sT]},
	sD47 = {[list(l) for l in self.sD47]},
	degrees = {self.degrees},
	description = {repr(self.description)},
	name = {repr(self.name)},
	label = {repr(self.label)},
	bfp = {self.bfp},
	bfp_CM = {[list(l) for l in self.bfp_CM]},
	chisq = {self.chisq},
	cholesky_residuals = {list(self.cholesky_residuals)},
	aic = {self.aic},
	bic = {self.bic},
	ks_pvalue = {self.ks_pvalue},
	)
''')

def combine_D47calibs(calibs, degrees = [0,2], same_T = []):
	'''
	Combine data from several `D47calib` instances.
	
	### Parameters
	
	+ **calibs**:
	A list of `D47calib` instances
	+ **degrees**:
	The polynomial degrees of the combined regression.
	+ **same_T**:
	Use this `list` to specify when samples from different calibrations are known/postulated
	to have formed at the same temperature (e.g. `DVH-2` and `DHC2-8` from the `fiebig_2021`
	and `anderson_2021_lsce` data sets). Each element of `same_T` is a `list` with the names
	of two or more samples formed at the same temperature.
	
	For example, the `combined_2023` calibration is computed with:
	
	`same_T = [['DVH-2', DHC-2-8'], ['ETH-1-1100-SAM', 'ETH-1-1100']]`

	Note that when samples from different calibrations have the same name,
	it is not necessary to explicitly list them in `same_T`.
	
	Also note that the regression will fail if samples listed together in `same_T`
	actually have different `T` values specified in the original calibrations.

	### Example
	
	The `devils_laghetto_2023` calibration is computed using the following code:
	
	````py
	K = [fiebig_2021.samples.index(_) for _ in ['LGB-2', 'DVH-2', 'DHC2-8']]

	fiebig_temp = D47calib(
		samples = [fiebig_2021.samples[_] for _ in K],
		T = fiebig_2021.T[K],
		D47 = fiebig_2021.D47[K],
		sT = fiebig_2021.sT[K,:][:,K],
		sD47 = fiebig_2021.sD47[K,:][:,K],
		)

	devils_laghetto_2023 = combine_D47calibs(
		calibs = [
			anderson_2021_lsce,
			fiebig_temp,
			],
		degrees = [0,2],
		same_T = [
			{'DVH-2', 'DHC2-8'},
			],
		)
	````
	'''

	samples = [s for c in calibs for s in c.samples]
	T = [t for c in calibs for t in c.T]
	D47 = [x for c in calibs for x in c.D47]
	sD47 = _block_diag(*[c.sD47 for c in calibs])
	sT = _block_diag(*[c.sT for c in calibs])

	for i in range(len(samples)):
		for j in range(len(samples)):
			if i != j:
				if (samples[i] == samples[j] or
					any([samples[i] in _ and samples[j] in _ for _ in same_T])):

					sT[i,j] = (sT[i,i] * sT[j,j])**.5
	
	calib = D47calib(
		samples = samples,
		T = T,
		D47 = D47,
		sT = sT,
		sD47 = sD47,
		degrees = degrees,
		)

	return calib

from ._calibs import *

try:
	def _cli(
		input,
		calib = 'combined_2023',
		input_delimiter = '\t',
		output_delimiter = '\t',
		T_precision = 2,
		D47_precision = 4,
		):
		from numpy import loadtxt

		calib = globals()[calib]

		with open(input) as f:
			fields = f.readlines()[0].strip().split(input_delimiter)

		data = loadtxt(input, delimiter = input_delimiter, skiprows = 1)
		if len(data.shape) == 1:
			kwargs = {fields[0]: data[:]}
		elif len(data.shape) == 2:
			kwargs = {
				fields[k]: data[:,k]
				for k in range(len(fields))
				}
		
		output_error_from_both = calib.T47(**kwargs, error_from = 'both')
		output_error_from_calib = calib.T47(**kwargs, error_from = 'calib')
		if 'sT' in kwargs:
			output_error_from_sT = calib.T47(**kwargs, error_from = 'sT')		
		if 'sD47' in kwargs:
			output_error_from_sD47 = calib.T47(**kwargs, error_from = 'sD47')		
		
		if 'T' in kwargs:
			kwargs['D47'] = output_error_from_both[0]
			kwargs['sD47_from_calib'] = output_error_from_calib[1]
			if 'sT' in kwargs:
				kwargs['sD47_from_sT'] = output_error_from_sT[1]
				kwargs['sD47_from_both'] = output_error_from_both[1]
		elif 'D47' in kwargs:
			kwargs['T'] = output_error_from_both[0]
			kwargs['sT_from_calib'] = output_error_from_calib[1]
			if 'sD47' in kwargs:
				kwargs['sT_from_sD47'] = output_error_from_sD47[1]
				kwargs['sT_from_both'] = output_error_from_both[1]

		print(output_delimiter.join(kwargs))
		for k in range(len(kwargs['T'])):
			print(output_delimiter.join([f'{kwargs[f][k]:.{T_precision if "T" in f[:2] else D47_precision}f}' for f in kwargs]))
		

except NameError:
	pass

def cli():
	typer.run(_cli)