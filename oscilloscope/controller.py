# Written by Marko kosunen, Marko.kosunen@aalto.fi 20190530
# The right way to do the unit controls is to write a controller class here
import os

import numpy as np
from thesdk import *
from verilog import *
from verilog.module import *

class controller(verilog):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.proplist = [ 'Rs' ];    #properties that can be propagated from parent
        self.Rs = 100e6;                   # Sampling frequency
        self.step=int(1/(self.Rs*1e-12))   #Time increment for control
        self.time=0
        self.IOS=Bundle()
        self.IOS.Members['control_write']= IO()        #We use this for writing
        _=verilog_iofile(self, name='control_write', dir='in', iotype='event')        
        #Permanent pointer assignment to write io
        self.IOS.Members['control_write'].Data=self.iofile_bundle.Members['control_write']
 
        self.IOS.Members['control_read']= IO()        #We use this for reading
        _=verilog_iofile(self, name='control_read', dir='out', iotype='event', datatype='int')        

        self.model='py';             #can be set externally, but is not propagated
        self.par= False              #By default, no parallel processing
        self.queue= []               #By default, no parallel processing

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;


        # We now where the verilog file is. 
        # Let's read in the file to have IOs defined
        self.dut=verilog_module(file=self.vlogsrcpath 
                + '/oscilloscope.sv')

        # Define the signal connectors associated with this 
        # controller
        # These are signals of tb driving several targets
        # Not present in DUT
        self.connectors=verilog_connector_bundle()

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        #These are signals not in dut
        self.newsigs_write=[
                 'initdone',
                ]

        # Selected signals controlled with this file with init values
        # These are tuples defining name init value pair
        self.signallist_write=[
            ('reset', 1),
            ('initdone',0),
        ]

        #These are signals not in dut
        self.newsigs_read=[
                ]
        self.signallist_read=[
        ]
        self.init()

    def init(self):
        self._vlogparameters =dict([('Rs',self.Rs)])
        # This gets interesting
        # IO is a file data stucture
        self.define_control()

    def reset_control_sequence(self):
        f=self.iofile_bundle.Members['control_write']
        self.time=0
        f.Data= np.array([])
        f.set_control_data(init=0) # Initialize to zeros at time 0
        self.assign_io()


    # First we start to control Verilog simulations with 
    # This controller. I.e we pass the IOfile definition
    def step_time(self,**kwargs):
        self.time+=kwargs.get('step',self.step)

    def define_control(self):
        # This is a bit complex way of passing the data,
        # But eventually we pass only the data , not the file
        # Definition. File should be created in the testbench
        scansigs_write=[]
        for name, val in self.signallist_write:
            # We manipulate connectors as verilog_iofile operate on those
            if name in self.newsigs_write:
                self.connectors.new(name=name, cls='reg')
            else:
                self.connectors.Members[name]=self.dut.io_signals.Members[name]
                self.connectors.Members[name].init=''
            scansigs_write.append(name) 

        f=self.iofile_bundle.Members['control_write']
        #define connectors controlled by this file in order of the list provided 
        f.verilog_connectors=self.connectors.list(names=scansigs_write)
        f.set_control_data(init=0) # Initialize to zeros at time 0

    #Methods to reset and to start datafeed
    def reset(self):
        #start defining the file
        f=self.iofile_bundle.Members['control_write']
        for name in [ 'reset', ]:
            f.set_control_data(time=self.time,name=name,val=1)

        # After awhile, switch off reset 
        self.step_time(step=15*self.step)

        for name in [ 'reset', ]:
            f.set_control_data(time=self.time,name=name,val=0)

    def start_datafeed(self):
        f=self.iofile_bundle.Members['control_write']
        for name in [ 'initdone', ]:
            f.set_control_data(time=self.time,name=name,val=1)
        self.step_time()

