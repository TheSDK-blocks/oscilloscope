"""
============
Oscilloscope
============

Plots continuous or sampled time-domain signals.

"""
import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

import numpy as np
import tempfile
import matplotlib.pyplot as plt

from thesdk import *
#from vhdl import *

import pdb

class oscilloscope(thesdk):
    """

    Attributes
    ----------
    IOS.Members['in'].Data: ndarray, list(ndarray)
        Time-domain input signal to use for plotting. If the number of columns
        is 1, the data is assumed to be sampled. If the number of columns is 2,
        the first column is assumed to be the time-vector, and the second
        column is assumed to be the signal. Multiple input signals can be
        plotted to the same figure by passing a list of 2-column ndarrays.
    signames: list(str), default []
        List containing names/handles for individual signals. The names are
        added to a legend in the figure. The length of signames should match
        the length of input signal list.
    title: str, default ''
        Title for the produced figure.
    scale_x: bool, default True
        Scale the x-axis (change the unit for more clearer scale of the time).
        WARNING: Can not be done if xlabel is changed manually
    nsamp: int, default None
        Number of samples to be plotted. Used for truncating the plotted data
        for sampled signals. When nsamp is None, the full signal is plotted.
    plot : bool, default True
        Should the figure be drawn or not? True -> figure is drawn, False ->
        figure not drawn.
    export : (bool,str), default (False,'')
        Should the figure(s), as well as the FFT datapoints, be exported to pdf
        and csv files or not? The filetypes .csv and .pdf are automatically
        appended to the 'filepath/filename' -string given.

        For example::

            export = (True,'./figures/result')

        would create 'result.pdf' and 'result.csv' (provided export_csv is True)
        in the directory called 'figures'.
    export_csv: bool, default True
        Export datapoints to CSV files? Enabled by default.

    """
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Initializing %s' %(__name__)) 
        self.proplist = ['Rs','nsamp','plot']
        
        self.nsamp = None 

        self.plot = True
        self.export = (False,"filepath")
        self.export_csv = True
        self.signames = []
        self.title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.xlim = None
        self.scale_x = True

        self.Rs = False
        self.draw_eye = False

        # Used internally to determine signal properties:
        self.xdata=None
        self.maxlen = 0

        self.IOS=Bundle()
        self.IOS.Members['in']=IO()

        self.model='py'
        self.par= False
        self.queue= []

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        ### Lets fix this later on
        if self.model=='vhdl':
            self.print_log(type='F', msg='VHDL simulation is not supported with v1.2\n Use v1.1')


    def float_to_si_string(self, num, precision=6):
        """Converts the given floating point number to a SI prefix string and divider.

        Parameters
        ----------
        num : float
            the number to convert.
        precision : int
            number of significant digits, defaults to 6.

        Returns
        -------
        x_scale : str
            the SI string of the value.
        x_scaler : float
            the scaler (divider) that can be used to normalize the signal to the
            given SI unit.
        """
        si_mag = [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
        si_pre = ['a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']

        if abs(num) < 1e-21:
            return '',1
        exp = np.log10(abs(num))

        pre_idx = len(si_mag) - 1
        for idx in range(len(si_mag)):
            if exp < si_mag[idx]:
                pre_idx = idx - 1
                break

        res = 10.0 ** (si_mag[pre_idx])
        return si_pre[pre_idx],res

    def sanitize_input(self, signal):
        '''
        This module assumes:

        1. ) Data is given as column vector
        2. ) Singular vector is flat (e.g. shape returns (n,) and not (n,1)) 

        Additionally, this function determines whether the data contains x,y value pairs
        or just y-values and the maximum length used for stacking the signals.

        '''
        sig_list = [] # List of sanitized signals
        sig_len = [] # List of sanitized signal lengths
        if self.xdata == None:
            self.xdata = False 
        if not isinstance(signal, list):
            if isinstance(signal, np.ndarray):
                signal = [signal]
            else:
                self.print_log(type='F', msg='Invalid input data type %s!' % type(signal))
        for i in range(len(signal)): # Loop over data
            sig = signal[i]
            if len(sig.shape) > 1: # Sanitize input data
                if sig.shape[1] > sig.shape[0]: # Was row vector, make column
                    sig = sig.T
                if sig.shape[1] == 1: # If the other dimention is empty, flatten (e.g. only magnitude data)
                    sig = sig.flatten()
                elif sig.shape[1] == 2: # The other dimensions contains time data
                    self.xdata=True
                else:
                    self.print_log(type='F',
                        msg='Input must be list of vectors, containing either (x,y)-value pairs or y-values!')
            elif len(sig.shape)==1: # Input signal contains only magnitude data
                self.xdata=False
            sig_len.append(len(sig))
            sig_list.append(sig)
        # Get maximum possible length, used for stacking purposes.
        if len(sig_len) < 1:
            self.print_log(type='F', msg='Input signal vector should contain at least 1 signal!')
        else:
            self.maxlen = max(sig_len)
        if not sig_len.count(sig_len[0]) == len(sig_len):
            maxlen=max(sig_len)
            for i, sig in enumerate(sig_list):
                if len(sig) != maxlen:
                    num_pad=maxlen - len(sig)
                    if self.xdata:
                        sig=np.r_['0', sig, np.ones((num_pad, 2))*np.nan]
                    else:
                        sig=np.r_['0', sig, np.ones((num_pad, 1))*np.nan]
                    sig_list[i] = sig 
        num_sig = len(sig_list)
        self.print_log(type='I', msg='Plotting %d input signals.' % (num_sig))
        if num_sig == 1:
            return sig_list[0]
        return sig_list

    def stack_and_save(self, signal):
        '''
        Stacks the signal into a single matrix for plotting, if the input type was a list.
        '''
        # Stacking the input signals to a matrix
        if self.xdata:
            sigstack = np.empty((self.maxlen,2*len(signal)))
        else:
            sigstack = np.empty((self.maxlen,len(signal)))
        sigstack[:] = np.nan
        for i in range(len(signal)):
            sig = signal[i]
            # This should be calculated for all signals (now only returns last one)
            self.peak_to_peak=self.find_peaktopeak(sig)
            if self.xdata:
                sigstack[:sig.shape[0],2*i:2*i+2] = sig
            else:
                sigstack[:sig.shape[0],i] = sig
        # Finally, save to file
        if self.export[0] and self.export_csv:
            for i in range(len(signal)):
                if len(self.signames) > 0:
                    np.savetxt("%s_%s.csv"%(self.export[1],self.signames[i]),signal[i],delimiter=",")        
                else:
                    np.savetxt("%s_%d.csv"%(self.export[1],i),signal[i],delimiter=",")        
        return sigstack


    def main(self):
        #TODO: Maybe we could split the main function into helpers
        if isinstance(self.IOS.Members['in'].Data,list):
            signal = [s.copy() for s in self.IOS.Members['in'].Data]
        else: # Just numpy array
            signal = np.copy(self.IOS.Members['in'].Data) # Use np.copy in order to skip pointer value change issues
        signal = self.sanitize_input(signal)
        if isinstance(signal,list): # We plot multiple signals
            self.is_stack=True
            signal=self.stack_and_save(signal)
        else: # We plot only one signal
            self.is_stack=False
            if self.xdata == None: # If user didn't specify, assume only y-data is given
                if len(signal.shape) > 1:
                    self.xdata=True
                else:
                    self.xdata=False
            if len(signal.shape) > 1 and signal.shape[1] > signal.shape[0]:
                signal = signal.T
            self.peak_to_peak=self.find_peaktopeak(signal)
            if self.export[0] and self.export_csv:
                np.savetxt("%s.csv"%(self.export[1]),signal,delimiter=",")        
        
        if self.draw_eye:
            self.eye_diagram(signal)
        # Check signal type:
        # Case 1: signal matrix contains data in x,y value pairs
        if isinstance(signal.shape,tuple) and self.xdata: 
            x_scaler = 1
            if self.nsamp is not None:
                signal=signal[-self.nsamp:,:]
            ncol = len(signal[0,:])
            if ncol % 2 != 0:
                self.print_log(type='W',msg='Missing a time-vector?')
            figure=plt.figure()
            if self.scale_x and self.xlabel=='':
                if self.xlim is not None:
                    # Compute the x-axis scale
                    x_scale,x_scaler=self.float_to_si_string(self.xlim[1])
                else:
                    # Compute the x-axis scale
                    x_scale,x_scaler=self.float_to_si_string(signal[:,0][-1])
            else:
                x_scale=''
                x_scaler=1
            for i in range(int(ncol/2)):
                signal[:,2*i]=signal[:,2*i]/x_scaler
                if len(self.signames) > 0 and len(self.signames) == int(ncol/2):
                    plt.plot(signal[:,2*i],signal[:,2*i+1],label=self.signames[i])
                    plt.legend()
                else:
                    plt.plot(signal[:,2*i],signal[:,2*i+1])
            if self.xlabel == '':
                self.xlabel=f'Time ({x_scale}s)'
            if self.ylabel == '':
                self.ylabel='Voltage (V)'
        # Case 2: signal matrix contains only y-values (sampled data)
        elif len(signal.shape) > 1 and not self.xdata:
            x_scaler = 1
            if self.nsamp is not None:
                signal=signal[-self.nsamp:,:]
            ncol = signal.shape[1]
            if len(self.signames) > 0:
                if len(self.signames) != int(ncol):
                    self.print_log(type='W', msg='Number of label must be equal to number of signals!')
            figure=plt.figure()
            for i in range(0, ncol):
                if len(self.signames) == ncol:
                    plt.step(x=np.arange(0,len(signal[:,i]), 1), 
                        y=signal[:,i],
                        where='post',
                        label=self.signames[i])
                    plt.legend()
                else:
                    plt.step(x=np.arange(0,len(signal[:,i]), 1), 
                        y=signal[:,i],
                        where='post')
            if self.xlabel == '':
                self.xlabel='Sample'
            if self.ylabel == '':
                self.ylabel='Code'
        # Case 3: Input was single vector, containing only y-values (sampled data)
        else:
            x_scaler = 1
            if self.nsamp is not None:
                signal=signal[-self.nsamp:]
            figure=plt.figure()
            if len(self.signames) == 1:
                plt.step(x=np.arange(0,len(signal), 1), 
                    y=signal,
                    where='post',
                    label=self.signames[0])
                plt.legend()
            else:
                plt.step(x=np.arange(0,len(signal), 1), 
                    y=signal,
                    where='post')
            if self.xlabel == '':
                self.xlabel='Sample'
            if self.ylabel == '':
                self.ylabel='Code'
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title != '':
            plt.title(self.title)
        if self.xlim is not None:
            plt.xlim(self.xlim[0]/x_scaler,self.xlim[1]/x_scaler)
        else:
            plt.autoscale(True,'x',tight=True)
        if self.plot:
            plt.show(block=False)
        if self.export[0]:
            fname = "%s.pdf"%self.export[1]
            self.print_log(type='I',msg='Saving figure to %s.' % fname)
            figure.savefig(fname,format='pdf')
        if not self.plot:
            plt.close()
        plt.pause(0.5)


    def find_peaktopeak(self,signal):
        if isinstance(signal.shape, tuple) and (len(signal.shape) > 1):
            tmpsig = signal[:,1]
        else:
            tmpsig = signal
        tmpsig=tmpsig[-int(0.5*len(tmpsig)):] #allow output to settle
        ptp=max(tmpsig)-min(tmpsig)
        return ptp

    def eye_diagram(self,signal):

        # signal.shape must be tuple and include samples and timestamps! 
        if not (isinstance(signal.shape, tuple) and (len(signal.shape) > 1)):
            self.print_log(type='F',msg="Invalid signal shape: signal.shape must be tuple.")
            return None
         
        period=1/self.Rs
        symbols=int((signal[-1,0]) / period)

        figure=plt.figure()
        axis=plt.gca()

        for i in range(symbols - 1):
            # Plot 2T intervals of the signal while increasing start time by 1T every iteration
            interval=np.where(np.logical_and(i*period < signal[:,0], signal[:,0] < (i+2)*period))[0]
            if len(interval) > 0:
                timestamps=signal[interval,0]-i*period
                x_scale,x_scaler=self.float_to_si_string(timestamps[-1])
                y_scale,y_scaler=self.float_to_si_string(signal[interval,1][-1])
                plt.xlabel(f"Time ({x_scale}s)")
                plt.ylabel(f"Voltage ({y_scale})V")
                plt.plot(timestamps/x_scaler,signal[interval,1]/y_scaler,'-',alpha=0.1,color='black')

        if self.plot:
            plt.show(block=False)
        if self.export[0]:
            fname = "%s_eyediagram.pdf"%self.export[1]
            self.print_log(type='I',msg='Saving eyediagram to %s.' % fname)
            figure.savefig(fname,format='pdf')
        if not self.plot:
            plt.close()
        plt.pause(0.5)

    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()
        else: 
          if self.model=='sv':
              self.vlogparameters=dict([ ('g_Rs',self.Rs),]) #Defines the sample rate
              self.run_verilog()
                            
              #This is for parallel processing
              if self.par:
                  self.queue.put(self.IOS.Members[Z].Data)
              del self.iofile_bundle #Large files should be deleted

          elif self.model=='vhdl':
              self.print_log(type='F', msg='VHDL simulation is not supported with v1.2\n Use v1.1')

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  oscilloscope import *
    #from  oscilloscope.controller import controller as oscilloscope_controller
    import pdb

    try:
        from plot_format import *
    except:
        self.print_log(type='W',msg='Module \'plot_format\' not in path. Plot formatting might look incorrect.')

    tvec = np.linspace(0,5e-9,200).reshape(-1,1)
    sine1 = np.sin(2*np.pi*1e9*tvec+0*np.pi/2).reshape(-1,1)
    sine2 = np.sin(2*np.pi*1e9*tvec-1*np.pi/2).reshape(-1,1)
    sine3 = np.sin(2*np.pi*1e9*tvec-2*np.pi/2).reshape(-1,1)
    sine4 = np.sin(2*np.pi*1e9*tvec-3*np.pi/2).reshape(-1,1)

    sig1 = np.hstack((tvec,sine1))
    sig2 = np.hstack((tvec,sine2))
    sig3 = np.hstack((tvec,sine3))
    sig4 = np.hstack((tvec,sine4))
    data = [[sig1, sig2, sig3, sig4], # Test stacked x,y values
            [sine1, sine2, sine3, sine4], # Test stacked y values
            sig1, # Test vector of x,y values
            sine1, # Test vector of y values
            ]
    signames = [
            ['0 deg','90 deg','180 deg','270 deg'],
            ['0 deg','90 deg','180 deg','270 deg'],
            [],
            [],
            ]

    duts=[oscilloscope() for i in range(len(data)) ]
    duts[0].model='py'
    for i in range(len(duts)): 
        duts[i].IOS.Members['in'].Data=data[i]
        duts[i].signames = signames[i]
        duts[i].init()
        duts[i].run()
        input()
