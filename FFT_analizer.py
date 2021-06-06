# 音声ファイルから画像生成を行う。
# 流れの夢コンテスト向けのコード
"""
参考サイト：http://takeshid.hatenadiary.jp/entry/2015/11/27/045123
参考サイト：https://qiita.com/yukky-k/items/0d18ec22420e8b35d0ac
ピーク検出付き
マイクの変更で変えたもの：各グラフの範囲、ピーク検出のしきい値
"""
#プロット関係のライブラリ
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys

#音声関係のライブラリ
import pyaudio
import struct

# 表示関係のライブラリ
#import pyautogui as pag # キーボード操作をしてくれるライブラリ
import datetime as dt
from scipy import signal # ピーク検出

class PlotWindow:
    def __init__(self):
        #プロット初期設定
        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle(u"Real time plotting")
        self.plt=self.win.addPlot() #プロットのビジュアル関係
        self.lim = 2 # 0.8
        self.plt.setYRange(-self.lim,self.lim)    #y軸の上限、下限の設定
        self.curve=self.plt.plot(pen = 'y')  #プロットデータを入れる場所
        self.plt.setLabel('left', "Amplitude")
        self.plt.setLabel('bottom', "Sec", units='s')
        self.win.nextRow() # 上下の分離点
        self.plt2=self.win.addPlot()
        self.curves = [self.plt2.plot(pen='g') for i in range(4)]
        self.plt2.setLogMode(True, False)
        self.lim2=0.8 # 0.15
        self.plt2.setYRange(0,self.lim2)    #y軸の制限
        self.plt2.setLabel('left', "Spectrum")
        self.plt2.setLabel('bottom', "Freqency", units='Hz')
        text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">Where is </span><span style="color: #FF0; font-size: 16pt;">MAX peak??</span></div>',
                           anchor=(0,0), angle=-5, border='w', fill=(0, 0, 255, 100))
        self.plt2.addItem(text)
        text.setPos(3.5, 0.7)
        self.text2 = pg.TextItem(anchor=(0,0), angle=-5, border='w', fill=(0, 0, 255, 100))
        self.text2.setPos(3.5, 0.55)
        self.plt2.addItem(self.text2)
        self.delta_no = 0 # mxの要素合わせに必要な分岐
        self.per = 1
        self.time = dt.datetime.now()
        print('開始時刻：' + str(self.time))

        #マイクインプット設定
        self.CHUNK= 2048             #1度に読み取る音声のデータ幅
        self.RATE=48000 # 44100             #サンプリング周波数
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

        #アップデート時間設定
        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)    #10msごとにupdateを呼び出し

        #音声データの格納場所(プロットデータ)
        self.data=np.zeros(self.CHUNK)
        print('開始時刻：' + str(self.time))

    def update(self):
        self.data=np.append(self.data, self.AudioInput()) #これがプロットデータ？？
        self.a = 3
        if len(self.data)/self.CHUNK > self.a:     #CHUNK点を超えたらCHUNK点を吐き出し
            self.data=self.data[self.CHUNK:]
        self.fft_data=self.FFT_AMP(self.data)[0]
        self.freq=self.FFT_AMP(self.data)[1]
        self.t = np.linspace(0,len(self.data)/self.RATE,len(self.data))
        self.plotset()
        #print(self.freq[9])
        
    def plotset(self):
        self.peaks = signal.find_peaks(self.fft_data[10:1024],threshold = 0.05) # 10:1024の間で検出
        self.id = self.peaks[0]+10
#        print(self.peaks)
        #print(self.freq[self.id])
        self.curve.setData(self.t[0:2048], self.data[0:2048], pen='g')   #プロットデータを格納
        self.peak = np.argmax(self.fft_data[7:])
        self.curves[0].setData(self.freq[7:], self.fft_data[7:],pen="g") # FFT結果
        self.curves[1].setData(self.freq[self.id], self.fft_data[self.id],pen=None, symbol='t', symbolPen=None, symbolSize=10,) # FFT結果
        self.text2.setText('→ '+str(int(self.freq[self.peak+7])) + ' [Hz]') # 表示器の結果表示
        self.plt.setYRange(-self.lim,self.lim)    #y軸の上限、下限の設定
        if np.max(self.data) > self.lim:
            # Y軸のレンジをautoにするやつだけど、おもすぎて却下。
            self.plt.autoRange()

    def AudioInput(self):
        try:
            ret=self.stream.read(self.CHUNK, exception_on_overflow = False)  #音声の読み取り(バイナリ)
            #バイナリ → 数値(int16)に変換
            #32768.0=2^15で割ってるのは正規化(絶対値を1以下にすること)
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


