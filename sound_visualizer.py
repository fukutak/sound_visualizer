# coding:utf-8

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys
import colorsys
import pyaudio
from scipy import signal
import csv


class PlotWindow:
    def __init__(self):
        # window settings
        self.app = QtGui.QApplication([])

        # Create window with GraphicsView widget
        self.win = pg.GraphicsLayoutWidget(border=(0, 0, 0))
        self.win.show()  # show widget alone in its own window
        self.win.setWindowTitle("What's Color You Are Listening?")
        self.plt0 = self.win.addPlot(row=1, col=1, colspan=1)
        self.curves = [self.plt0.plot() for i in range(4)]
        self.brushes = [(0, 0, 0), (0, 0, 0)]
        self.fill1 = pg.FillBetweenItem(self.curves[0], self.curves[1], self.brushes[0])
        self.fill2 = pg.FillBetweenItem(self.curves[2], self.curves[3], self.brushes[1])
        self.plt0.addItem(self.fill1)
        self.lim = 50
        self.plt0.setYRange(-self.lim, self.lim)
        self.plt0.setXRange(-self.lim, self.lim)
        self.plt0.setAspectLocked(True)
        self.plt0.hideAxis('bottom')
        self.plt0.hideAxis('left')

        # plot settings
        self.sec = 0
        self.x1 = np.cos(np.linspace(0, np.pi, 1024))
        self.y1 = np.sin(np.linspace(0, np.pi, 1024))
        self.x2 = np.cos(np.linspace(np.pi, 2*np.pi, 1024))
        self.y2 = np.sin(np.linspace(np.pi, 2*np.pi, 1024))
        self.mx = np.cos(np.linspace(0, 2*np.pi, 2048))
        self.my = np.sin(np.linspace(0, 2*np.pi, 2048))
        self.fft_peaks = np.zeros(2048)
        self.fft_peaks0 = np.zeros(2048)
        self.fft_peaks1 = np.zeros(2048)
        self.zero = np.zeros(1000)
        self.key = ['　ド　', '　ド＃', '　レ　', '　レ＃', '　ミ　', 'ファ　',
                    'ファ＃', '　ソ　', '　ソ＃', '　ラ　', '　ラ＃', '　シ　']
        self.rgb_bar = np.array([
                      [187, 85, 133],  # ra
                      [145, 106, 153],  # ra_#
                      [219, 163, 176],  # si
                      [224, 31, 32],  # do
                      [232, 184, 97],  # do_#
                      [216, 178, 19],  # re
                      [141, 177, 69],  # re_＃
                      [62, 169, 53],  # mi
                      [187, 131, 56],  # fa
                      [210, 93, 40],  # fa_#
                      [0, 176, 204],  # so
                      [38, 133, 179],  # so_#
                      ])/255
        self.up = np.exp(1/12*np.log(2))
        self.up = np.exp(1/12*np.log(2))
        self.key_new = ['　ラ　', '　ラ＃', '　シ　', '　ド　', '　ド＃', '　レ　', '　レ＃', '　ミ　', 'ファ　',
                        'ファ＃', '　ソ　', '　ソ＃'] 
        self.hsv_bar = np.zeros((17, 3))

        for rgb in range(len(self.rgb_bar)):
            self.hsv_bar[rgb, :] = colorsys.rgb_to_hsv(self.rgb_bar[rgb, 0], self.rgb_bar[rgb, 1], self.rgb_bar[rgb, 2])

        # prepare Wave plot
        self.win2 = self.win.addLayout(row=1, col=1, colspan=1)
        self.plt = self.win2.addPlot()
        self.lim = 2
        self.plt.setYRange(-self.lim, self.lim)
        self.curve = self.plt.plot(pen='y') 
        self.plt.setLabel('left', "Amplitude")
        self.plt.setLabel('bottom', "Sec", units='s')
        self.plt.hideAxis('bottom')
        self.plt.hideAxis('left')

        # Text item
        self.f1 = 0
        self.h = '<div style="text-align: center"><span style="color: #000; font-size: 30pt;">str(self.f1)</span></div>'
        self.text = pg.TextItem(html=self.h,
                                anchor=(0.5, 0.5), angle=0, color='w', border='k', fill=(0, 0, 0, 255))
        self.plt0.addItem(self.text)
        self.text.setPos(50, 40)
        self.text.setText('0 [Hz]')

        # init ploting data
        self.data0 = np.zeros(6144)
        self.data1 = np.zeros(6144)
        self.bla = np.zeros((1, 3))
        self.bla0 = np.zeros((1, 3))
        self.RGB0 = np.zeros((1, 3))
        self.RGB1 = np.zeros((1, 3))
        self.RGB2 = [0, 0, 0]
        self.r = 2
        self.r0 = 0
        self.r1 = 0
        self.f0 = 0
        self.f1 = 0
        self.RGB = np.array((0.0, 0.0, 0.0))
        self.piano = 0
        # import polygon data
        self.Poly_x0 = np.zeros(2048)
        self.Poly_y0 = np.zeros(2048)
        self.n0 = 0
        self.gold = (1+5**0.5)/2
        with open('PolyData2.csv', 'r') as f:
            self.reader = csv.reader(f)
            self.i = 0
            self.Polys = np.zeros((58, 2048))
            for row in self.reader:
                self.Polys[self.i, :] = row
                self.i += 1

        self.st = 0
        self.count = 0
        self.count1 = 10

        # mic settings
        self.CHUNK = 2048
        self.RATE = 48000  # 44100
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)

        # update settings
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)

        self.data = np.zeros(self.CHUNK)

    def update(self):
        self.data = np.append(self.data, self.audio_input())
        self.a = 3
        if len(self.data)/self.CHUNK > self.a:
            self.data = self.data[self.CHUNK:]
        self.fft_data = self.fft_amp(self.data)[0]
        self.freq = self.fft_amp(self.data)[1]
        self.t = np.linspace(0, len(self.data)/self.RATE, len(self.data))
        self.plotset()

    def plotset(self):
        # param: freq, data, t, fft_data
        self.index = np.argmax(self.fft_data)
        self.peaks = signal.find_peaks(self.fft_data[10:], threshold=0.004)
        self.id = self.peaks[0]+10
        self.peaks_f = np.array(self.freq[self.id])
        self.f = self.freq[self.index]
        self.spec = self.fft_data[self.index]

        # judge polygon num
        self.n1 = len(self.id)
        self.nB = (2*self.n1+self.n0)/3
        if 0 <= self.nB < 0.005:
            self.nB1 = int(0)
        elif 0.005 <= self.nB < 2:  # 1
            self.nB1 = int(1)  # triangle
        elif self.nB < 28:
            self.nB1 = int(self.nB)  # recangle and more
        else:
            self.nB1 = int(0)  # circle

        # calc radius
        self.r = 5*(np.log2(self.spec*self.nB1/0.003+0.0001))+4
        if len(self.data) == self.a*self.CHUNK:
            if self.r < 10:
                self.r = 10
            elif self.r > 38:
                self.r = 33

            # determin color
            [self.RGB, self.piano] = self.color()
            if len(self.peaks_f) >= 2:
                self.RGBs = self.colors()
                self.RGB_origin = self.RGB
                self.RGB = (1*self.RGB + np.sum(self.RGBs, axis=0)/len(self.RGBs))/2

            # averaging to smooth between flames
            self.r_beau = (1*self.r+3*self.r0+1*self.r1)/5
            self.r_spec = (1*self.r+3*self.r0)/5
            self.RGB_beau = (4*self.RGB + 3*self.RGB0 + 0*self.RGB1)/7  # color
            self.bla_beau = (self.bla*5 + 2*(self.RGB_beau+2*self.RGB1)/3)/7  # color in brank
            self.f_beau = (self.f+1*self.f0+2*self.f1)/4
            self.data_beau = (3*self.data+2*self.data0+1*self.data1)/6

            # waves on circle
            self.b = 40  # magnification
            self.data_beau[2047] = self.data_beau[0]
            self.wave = self.r_beau+self.b*(self.data_beau[0:2048])  # +50*self.fft_data[10:2058])
            # self.wave0 = (self.r_beau-0.7)+self.b*(self.data_beau[0:2048])
            self.xx1 = self.wave[0:1024]*self.x1  # upper half
            self.yy1 = self.wave[0:1024]*self.y1  # upper half
            self.xx2 = self.wave[1024:2048]*self.x2  # bottom half
            self.yy2 = self.wave[1024:2048]*self.y2  # bottom half
            self.trans_bla = [int(self.bla_beau[0, 0]), int(self.bla_beau[0, 1]), int(self.bla_beau[0, 2])]  # transform types for pg
            self.brushes[0] = int(self.bla_beau[0, 0]), int(self.bla_beau[0, 1]), int(self.bla_beau[0, 2])  # transform types for pg
            # transform types to mkPen
            self.trans = [int(self.RGB_beau[0, 0]), int(self.RGB_beau[0, 1]), int(self.RGB_beau[0, 2])]
            self.trans0 = [int(self.RGB0[0, 0]), int(self.RGB0[0, 1]), int(self.RGB0[0, 2])]
            # new mkPen
            # self.trans1 = [int(self.RGB_new[0]),int(self.RGB_new[1]),int(self.RGB_new[2])]
            # self.trans2 = [int(self.RGB_new[0]*0.8),int(self.RGB_new[1]*0.8),int(self.RGB_new[2]*0.8)]

            self.Poly_x = (self.r_spec+0)/self.gold*self.Polys[self.nB1*2, :]
            self.Poly_y = (self.r_spec+0)/self.gold*self.Polys[self.nB1*2+1, :]
            self.Poly_xB = (2*self.Poly_x+3*self.Poly_x0)/5
            self.Poly_yB = (2*self.Poly_y+3*self.Poly_y0)/5
            # setting data
            self.fill1.setBrush(self.brushes[0])
            self.curves[0].setData(self.xx1, self.yy1, pen=pg.mkPen(self.trans, width=2.5))
            self.curves[1].setData(self.xx2, self.yy2, pen=pg.mkPen(self.trans, width=2.5))
            # self.fill2.setBrush(self.brushes[1])
            # polygon
            self.curves[2].setData(self.Poly_xB, self.Poly_yB, pen=pg.mkPen(self.trans, width=2))
            # self.curves[3].setData(self.Poly_xB,self.Poly_yB, pen=pg.mkPen(self.trans0,width=2))
            # self.curve.setData(self.t, self.data, pen=pg.mkPen(self.trans,width=1))#self.RGB_beau.tolist)   #プロットデータを格納
            # $self.text.setText(str(int(self.f_beau)) + ' [Hz]：'+ self.key_new[int(self.piano)],color=self.RGB_origin) # 表示器

            # update params
            self.f0 = self.f1  # for display
            self.f1 = self.f  # for display
            self.RGB0 = self.RGB_beau
            self.RGB1 = self.RGB0
            # self.RGB2 = self.RGB
            self.bla0 = self.bla_beau
            self.r0 = self.r_beau
            self.r1 = self.r0
            self.data0 = self.data_beau
            self.data1 = self.data0
            self.fft_peaks0 = self.fft_peaks
            self.fft_peaks1 = self.fft_peaks0
            self.fft_peaks = np.zeros(2048)
            self.Poly_x0 = self.Poly_x
            self.Poly_y0 = self.Poly_y
            self.n0 = self.nB

    def color(self):  # using self.f
        if self.f > 27.5:
            # calc every 27.5Hz
            self.half_up0 = np.log((self.f/27.5))/np.log(self.up)
            self.half_up = np.floor(self.half_up0)
            self.oct, self.doremi = divmod(self.half_up, 12)  # quotient：Octave, surplus: Doremi information
            self.oct = int(self.oct)
            self.doremi = int(self.doremi)

            # determin a color
            self.piano = np.round(self.half_up0) % 12
            self.RGB = ((self.rgb_bar[self.doremi] - self.rgb_bar[self.doremi-1])*(self.half_up0 - self.half_up) + self.rgb_bar[self.doremi-1])*255  #*self.oct*0.3
        else:
            self.RGB = self.RGB
            self.piano = self.piano
        return self.RGB, self.piano

    def colors(self):  # determin color using self.peaks_f
        # calc every 27.5Hz
        self.half_ups0 = np.log((self.peaks_f/27.5))/np.log(self.up)
        self.half_ups = np.floor(self.half_ups0)
        self.octs, self.doremis = np.array(divmod(self.half_ups, 12))  # quotient：Octave, surplus: Doremi information

        self.octs = self.octs.astype('int')
        self.doremis = self.doremis.astype('int')
        self.RGBs = ((self.rgb_bar[self.doremis] - self.rgb_bar[self.doremis-1])*(self.half_ups0 - self.half_ups).reshape(-1, 1) + self.rgb_bar[self.doremis-1])*255  # *self.oct*0.3

        return self.RGBs

    def audio_input(self):
        ret = self.stream.read(self.CHUNK, exception_on_overflow=False)  # read binary data 
        ret = np.frombuffer(ret, dtype="int16")/2**12 # normalization
        return ret

    def fft_amp(self, data):
        data = np.hamming(len(data))*data
        data = np.fft.fft(data)
        data = np.abs(data)/len(data)*4
        self.freq = np.fft.fftfreq(len(data), 1/self.RATE)
        return data[:int(len(data)/2)], self.freq[:int(len(self.freq)/2)]


if __name__ == "__main__":
    plotwin = PlotWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
