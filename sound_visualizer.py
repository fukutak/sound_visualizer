# coding:utf-8
#音から動的に画像を生成
"""
from imageAudio_no_MyDesk2.py
四角の大きさ：スペクトルの強さ、色：最大スペクトルの周波数で画像生成
円の中に円を入れて、内側の円はスペクトルの個数の正多角形を描写
なめらかに多角形が動くように2048点の周方向データに変更して描写

onMyDesk3の変更点
1. 高速化のためにコードを簡略化する。
"""
#プロット関係のライブラリ

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys
import colorsys

#音声関係のライブラリ
import pyaudio

# 表示関係のライブラリ
from scipy import signal , interpolate# ピーク検出と、内挿
import datetime
import csv


class PlotWindow:
    def __init__(self):
        #プロット初期設定
        self.app = QtGui.QApplication([])

        ## Create window with GraphicsView widget
        self.win = pg.GraphicsLayoutWidget(border=(0,0,0))
        self.win.show()  ## show widget alone in its own window
        self.win.setWindowTitle("What's Color You Are Listening?")
        self.plt0 = self.win.addPlot(row=1, col=1,colspan=1)
        self.curves = [self.plt0.plot() for i in range(4)]
        self.brushes = [(0, 0, 0),(0, 0, 0)]
        self.fill1 = pg.FillBetweenItem(self.curves[0], self.curves[1], self.brushes[0])
        self.fill2 = pg.FillBetweenItem(self.curves[2], self.curves[3], self.brushes[1])
        self.plt0.addItem(self.fill1)
        self.lim = 50
        self.plt0.setYRange(-self.lim,self.lim)    #y軸の上限、下限の設定
        self.plt0.setXRange(-self.lim,self.lim)    #y軸の上限、下限の設定
        self.plt0.setAspectLocked(True)
        self.plt0.hideAxis('bottom')
        self.plt0.hideAxis('left')

        # 初期設定：周方向データは2048点
        self.sec = 0 # 内側の多角形を回転させるときに使用
        self.x1 = np.cos(np.linspace(0, np.pi, 1024)) # 上半分のデータのx方向
        self.y1 = np.sin(np.linspace(0, np.pi, 1024)) # 上半分のデータのy方向
        self.x2 = np.cos(np.linspace(np.pi, 2*np.pi, 1024)) # 下半分のデータのx方向
        self.y2 = np.sin(np.linspace(np.pi, 2*np.pi, 1024)) # 下半分のデータのy方向 間を色で塗るために分割している、やりようによってはしなくてもよいかも
        self.mx = np.cos(np.linspace(0, 2*np.pi, 2048)) # 一周のx方向
        self.my = np.sin(np.linspace(0, 2*np.pi, 2048)) # 一周のy方向
        self.fft_peaks = np.zeros(2048)
        self.fft_peaks0 = np.zeros(2048)
        self.fft_peaks1 = np.zeros(2048)
        self.zero = np.zeros(1000)
        self.key = ['　ド　','　ド＃','　レ　','　レ＃','　ミ　','ファ　','ファ＃','　ソ　','　ソ＃','　ラ　','　ラ＃','　シ　']
        # 新しい色の定義 [153, 114, 35], # re_b [0, 129, 65], # mi_b [27, 125, 160], # so_b [99, 55, 140], # ra_b [163, 116, 162], # si_b
        self.rgb_bar = np.array([
                      [187, 85, 133], # ra
                      [145, 106, 153], # ra_#
                      [219, 163, 176], # si
                      [224, 31, 32], # do
                      [232, 184, 97], # do_#
                      [216, 178, 19], # re
                      [141, 177, 69], # re_＃
                      [62, 169, 53], # mi
                      [187, 131, 56], # fa
                      [210, 93, 40], # fa_#
                      [0, 176, 204], # so
                      [38, 133, 179], # so_#
                      ])/255
        self.up = np.exp(1/12*np.log(2)) # 半音上がるときの係数
        self.key_new = ['　ラ　','　ラ＃','　シ　','　ド　','　ド＃','　レ　','　レ＃','　ミ　','ファ　','ファ＃','　ソ　','　ソ＃'] # ラスターとのほうが都合が良い
        self.hsv_bar = np.zeros((17,3))

        for rgb in range(len(self.rgb_bar)): # hsv色覚空間に変換　明暗とか調節できる。
            self.hsv_bar[rgb,:] = colorsys.rgb_to_hsv(self.rgb_bar[rgb,0],self.rgb_bar[rgb,1],self.rgb_bar[rgb,2])

        # Waveプロットの用意
        self.win2 = self.win.addLayout(row=1, col=1,colspan=1)
        self.plt=self.win2.addPlot() #プロットのビジュアル関係
        self.lim = 2
        self.plt.setYRange(-self.lim,self.lim)    #y軸の上限、下限の設定
        self.curve=self.plt.plot(pen = 'y')  #プロットデータを入れる場所
        self.plt.setLabel('left', "Amplitude")
        self.plt.setLabel('bottom', "Sec", units='s')
        self.plt.hideAxis('bottom')
        self.plt.hideAxis('left')

        ## Text item
        self.f1 = 0
        self.h = '<div style="text-align: center"><span style="color: #000; font-size: 30pt;">str(self.f1)</span></div>'
        self.text = pg.TextItem(html=self.h,
                           anchor=(0.5,0.5), angle=0,color='w', border='k', fill=(0, 0, 0, 255))
        self.plt0.addItem(self.text)
        self.text.setPos(50, 40)
        self.text.setText('0 [Hz]') # 表示器の結果表示

        # dataの初期設定
        self.data0 = np.zeros(6144)
        self.data1 = np.zeros(6144)
        self.bla = np.zeros((1,3))
        self.bla0 = np.zeros((1,3))
        self.RGB0 = np.zeros((1,3))
        self.RGB1 = np.zeros((1,3))
        self.RGB2 = [0, 0, 0]
        self.r = 2
        self.r0 = 0
        self.r1 = 0
        self.f0 = 0
        self.f1 = 0
        self.RGB = np.array((0.0,0.0,0.0))
        self.piano = 0
        # 多角形のデータをインポート
        self.Poly_x0 = np.zeros(2048)
        self.Poly_y0 = np.zeros(2048)
        self.n0 = 0
        self.gold = (1+5**0.5)/2
        with open('PolyData2.csv', 'r') as f:
            self.reader = csv.reader(f) # readerオブジェクトを作成
            # 行ごとのリストを処理する
            self.i = 0
            self.Polys = np.zeros((58,2048))
            for row in self.reader:
                self.Polys[self.i,:] = row
                self.i += 1

        # 終了シークエンスの初期値
        self.st = 0
        self.count = 0 # 表示機のカウント
        self.count1 = 10 # 終了シークエンスの時間

        #マイクインプット設定
        self.CHUNK= 2048             #1度に読み取る音声のデータ幅
        self.RATE=48000 # 44100             #サンプリング周波数
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)

        #アップデート時間設定
        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)    #10msごとにupdateを呼び出し

        #音声データの格納場所(プロットデータ)
        self.data=np.zeros(self.CHUNK)

    def update(self):
        self.data=np.append(self.data, self.AudioInput()) #これがプロットデータ？？
        self.a = 3
        if len(self.data)/self.CHUNK > self.a:     #CHUNK点の3倍を超えたらCHUNK点を吐き出し
            self.data=self.data[self.CHUNK:]
        self.fft_data=self.FFT_AMP(self.data)[0]
        self.freq=self.FFT_AMP(self.data)[1]
        self.t = np.linspace(0,len(self.data)/self.RATE,len(self.data))
        self.plotset()

    def plotset(self):
        # 画像生成するための準備　変数；freq, data, t, fft_data
        self.index = np.argmax(self.fft_data)
        self.peaks = signal.find_peaks(self.fft_data[10:],threshold = 0.004) # 10:の間で検出
        self.id = self.peaks[0]+10 # ピークたちのインデックス
        self.peaks_f = np.array(self.freq[self.id]) # ピークを持つ周波数のベクトル
        self.f = self.freq[self.index] # fft_dataの最大値のインデックスをfreqに入れて周波数の特定
        self.spec = self.fft_data[self.index]

        # 多角形の個数の判定
        self.n1 = len(self.id)
        self.nB = (2*self.n1+self.n0)/3
        if self.nB >= 0 and self.nB < 0.005:
            self.nB1 = int(0) # 円
        elif self.nB >= 0.005 and self.nB < 2: # 1
            self.nB1 = int(1) # 三角形
        elif self.nB < 28:
            self.nB1 = int(self.nB) # 四角形以降の多角形
        else:
            self.nB1 = int(0) # 円

        # 半径の計算式 log2にしているのは10にすると変化が少なすぎてしまうから。
        self.r = 5*(np.log2(self.spec*self.nB1/0.003+0.0001))+4
        if len(self.data) == self.a*self.CHUNK:
            if self.r < 10:
                self.r = 10
            elif self.r > 38:
                self.r = 33

            # 色の決定
            [self.RGB,self.piano] = self.color()
            if len(self.peaks_f) >= 2:
                self.RGBs = self.colors()
                self.RGB_origin = self.RGB
                self.RGB = (1*self.RGB + np.sum(self.RGBs, axis=0)/len(self.RGBs))/2
                #print(len(self.peaks_f))
                #print(self.RGB)

            # 描写するものをなめらかにさせるために平均化処理
            self.r_beau = (1*self.r+3*self.r0+1*self.r1)/5
            self.r_spec = (1*self.r+3*self.r0)/5
            self.RGB_beau = (4*self.RGB + 3*self.RGB0 + 0*self.RGB1)/7# 急激な色の変化を防ぐ
            self.bla_beau = (self.bla*5 + 2*(self.RGB_beau+2*self.RGB1)/3)/7 # ブランク部分の色味
            self.f_beau = (self.f+1*self.f0+2*self.f1)/4
            self.data_beau = (3*self.data+2*self.data0+1*self.data1)/6

            #self.RGB_new = (self.RGB + self.RGB2)/2
            # 生波形を円周上に配置
            self.b = 40 # 生波形の倍率
            self.data_beau[2047] = self.data_beau[0]
            self.wave = self.r_beau+self.b*(self.data_beau[0:2048])#+50*self.fft_data[10:2058])
            #self.wave0 = (self.r_beau-0.7)+self.b*(self.data_beau[0:2048])
            self.xx1 = self.wave[0:1024]*self.x1 # 上半分のプロットデータ
            self.yy1 = self.wave[0:1024]*self.y1 # 上半分のプロットデータ
            self.xx2 = self.wave[1024:2048]*self.x2 # 下半分のプロットデータ
            self.yy2 = self.wave[1024:2048]*self.y2 # 下半分のプロットデータ
            self.trans_bla = [int(self.bla_beau[0,0]),int(self.bla_beau[0,1]),int(self.bla_beau[0,2])] # pgに渡すために型を変換(ぬる色)
            self.brushes[0] = int(self.bla_beau[0,0]),int(self.bla_beau[0,1]),int(self.bla_beau[0,2]) # pgに渡すために型を変換
            # mkPenの型に変換
            self.trans = [int(self.RGB_beau[0,0]),int(self.RGB_beau[0,1]),int(self.RGB_beau[0,2])]
            self.trans0 = [int(self.RGB0[0,0]),int(self.RGB0[0,1]),int(self.RGB0[0,2])]
            # 新しくmkPenの色を追加
            #self.trans1 = [int(self.RGB_new[0]),int(self.RGB_new[1]),int(self.RGB_new[2])]
            #self.trans2 = [int(self.RGB_new[0]*0.8),int(self.RGB_new[1]*0.8),int(self.RGB_new[2]*0.8)]

            self.Poly_x = (self.r_spec+0)/self.gold*self.Polys[self.nB1*2,:] # CSVデータのやつ
            self.Poly_y = (self.r_spec+0)/self.gold*self.Polys[self.nB1*2+1,:] # CSVデータのやつ
            self.Poly_xB = (2*self.Poly_x+3*self.Poly_x0)/5
            self.Poly_yB = (2*self.Poly_y+3*self.Poly_y0)/5
            # 描写curvesにそれぞれデータセット
            self.fill1.setBrush(self.brushes[0])
            self.curves[0].setData(self.xx1, self.yy1, pen=pg.mkPen(self.trans,width=2.5))
            self.curves[1].setData(self.xx2, self.yy2, pen=pg.mkPen(self.trans,width=2.5))
            #self.fill2.setBrush(self.brushes[1])
            # 多角形
            self.curves[2].setData(self.Poly_xB,self.Poly_yB, pen=pg.mkPen(self.trans,width=2))
            #self.curves[3].setData(self.Poly_xB,self.Poly_yB, pen=pg.mkPen(self.trans0,width=2))
            #self.curve.setData(self.t, self.data, pen=pg.mkPen(self.trans,width=1))#self.RGB_beau.tolist)   #プロットデータを格納
            #$self.text.setText(str(int(self.f_beau)) + ' [Hz]：'+ self.key_new[int(self.piano)],color=self.RGB_origin) # 表示器

            # 更新
            self.f0 = self.f1 # 表示器用
            self.f1 = self.f # 表示器用
            self.RGB0 = self.RGB_beau
            self.RGB1 = self.RGB0
            #self.RGB2 = self.RGB # 前回のRGB値の引き継ぎ
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

    def color(self): # self.fが入ってきている。
        if self.f > 27.5:
            # それぞれ27.5Hzに対して半音がいくつ上がるのかを計算
            self.half_up0 = np.log((self.f/27.5))/np.log(self.up)
            self.half_up = np.floor(self.half_up0)
            self.oct, self.doremi = divmod(self.half_up,12) # 切り捨て、商：オクターブ、余剰：ドレミの情報
            self.oct = int(self.oct)
            self.doremi = int(self.doremi)

            # 対応する色の決定
            self.piano = np.round(self.half_up0)%12
            self.RGB = ((self.rgb_bar[self.doremi] - self.rgb_bar[self.doremi-1])*(self.half_up0 - self.half_up) + self.rgb_bar[self.doremi-1])*255 #*self.oct*0.3
        else:
            self.RGB = self.RGB
            self.piano = self.piano
        return self.RGB, self.piano

    def colors(self): # self.peaks_fを使った色決定を行う
        # それぞれ27.5Hzに対して半音がいくつ上がるのかを計算
        self.half_ups0 = np.log((self.peaks_f/27.5))/np.log(self.up)
        self.half_ups = np.floor(self.half_ups0)
        self.octs, self.doremis = np.array(divmod(self.half_ups,12)) # 切り捨て、商：オクターブ、余剰：ドレミの情報

        self.octs = self.octs.astype('int')
        self.doremis = self.doremis.astype('int')

        # 対応する色の決定
        # 一次元配列には行と列の区別がないため、二次元配列として明確に列ベクトルをreshapeメソッドで定義する。
        self.RGBs = ((self.rgb_bar[self.doremis] - self.rgb_bar[self.doremis-1])*(self.half_ups0 - self.half_ups).reshape(-1,1) + self.rgb_bar[self.doremis-1])*255 #*self.oct*0.3

        return self.RGBs

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK, exception_on_overflow = False)  #音声の読み取り(バイナリ)
        #バイナリ → 数値(int16)に変換
        #32768.0=2^15で割ってるのは正規化(絶対値を1以下にすること)
        ret=np.frombuffer(ret, dtype="int16")/2**12
        return ret

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
