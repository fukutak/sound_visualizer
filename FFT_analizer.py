
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys
import pyaudio
import struct
import datetime as dt
from scipy import signal

class PlotWindow:
    def __init__(self):
        # window settings
        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle(u"Real time plotting")
        self.plt=self.win.addPlot()
        self.lim = 2 # 0.8
        self.plt.setYRange(-self.lim,self.lim)
        self.curve=self.plt.plot(pen = 'y')
        self.plt.setLabel('left', "Amplitude")
        self.plt.setLabel('bottom', "Sec", units='s')
        self.win.nextRow()
        self.plt2=self.win.addPlot()
        self.curves = [self.plt2.plot(pen='g') for i in range(4)]
        self.plt2.setLogMode(True, False)
        self.lim2=0.8 # 0.15
        self.plt2.setYRange(0,self.lim2)
        self.plt2.setLabel('left', "Spectrum")
        self.plt2.setLabel('bottom', "Freqency", units='Hz')
        text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">Where is </span><span style="color: #FF0; font-size: 16pt;">MAX peak??</span></div>',
                           anchor=(0,0), angle=-5, border='w', fill=(0, 0, 255, 100))
        self.plt2.addItem(text)
        text.setPos(3.5, 0.7)
        self.text2 = pg.TextItem(anchor=(0,0), angle=-5, border='w', fill=(0, 0, 255, 100))
        self.text2.setPos(3.5, 0.55)
        self.plt2.addItem(self.text2)
        self.delta_no = 0
        self.per = 1
        self.time = dt.datetime.now()

        # mic settings
        self.CHUNK= 2048
        self.RATE=48000 # 44100
        self.audio=pyaudio.PyAudio()
        self.hostAPICount = self.audio.get_host_api_count()
        self.default = self.audio.get_default_input_device_info()
        self.count = self.audio.get_device_count()
        self.SPEAKERS = self.audio.get_default_output_device_info()["hostApi"]
        print(self.default)
        print(self.count)
        print(self.hostAPICount)
        for i in range(self.count):
            print(self.audio.get_device_info_by_index(i))

        for cnt in range(self.hostAPICount):
            print(self.audio.get_host_api_info_by_index(cnt))

        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    input_host_api_specific_stream_info=self.SPEAKERS,
                                    frames_per_buffer=self.CHUNK)

        # recall
        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        self.data=np.zeros(self.CHUNK)

    def update(self):
        self.data=np.append(self.data, self.AudioInput())
        self.a = 3
        if len(self.data)/self.CHUNK > self.a:
            self.data=self.data[self.CHUNK:]
        self.fft_data=self.FFT_AMP(self.data)[0]
        self.freq=self.FFT_AMP(self.data)[1]
        self.t = np.linspace(0,len(self.data)/self.RATE,len(self.data))
        self.plotset()
        
    def plotset(self):
        self.peaks = signal.find_peaks(self.fft_data[10:1024],threshold = 0.05)
        self.id = self.peaks[0]+10
        self.curve.setData(self.t[0:2048], self.data[0:2048], pen='g')
        self.peak = np.argmax(self.fft_data[7:])
        self.curves[0].setData(self.freq[7:], self.fft_data[7:],pen="g")
        self.curves[1].setData(self.freq[self.id], self.fft_data[self.id],pen=None, symbol='t', symbolPen=None, symbolSize=10,)
        self.text2.setText('→ '+str(int(self.freq[self.peak+7])) + ' [Hz]')
        self.plt.setYRange(-self.lim,self.lim)
        if np.max(self.data) > self.lim:
            self.plt.autoRange()

    def AudioInput(self):
        try:
            ret=self.stream.read(self.CHUNK, exception_on_overflow = False)
            ret=np.frombuffer(ret, dtype="int16")/2**12
            return ret
        except Exception as e:
            print('You need to switch input devices.')
            print(e)
    def FFT_AMP(self,data):
        data=np.hamming(len(data))*data
        data=np.fft.fft(data)
        data=np.abs(data)/len(data)*4
        self.freq=np.fft.fftfreq(len(data),1/(self.RATE))
        return data[:int(len(data)/2)], self.freq[:int(len(self.freq)/2)]
        
if __name__=="__main__":
    plotwin=PlotWindow()
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


