import numpy as np
import matplotlib.pyplot as plt
from function2 import _Affine
from utils import args
#from common.gradient import numerical_gradient

#func = "sigmoid"
func = "affine"
mode = ""
#mode = "center"
#mode = "reverse"
b = 64
num = 10
name = "%s_%db_%s"%(func, b, mode)
plot = True
args = args()

if b==64:X = np.linspace(-5, 5, num)
elif b==32: X = np.linspace(-5, 5, num).astype("float32")

if func=="affine" and b==64:X = np.random.randn(num, 8)
elif func=="affine" and b==32: X = np.random.randn(num, 8).astype("float32")

if func=="sigmoid":
    f = _Sigmoid()
    g = lambda x: (1-f(x))*f(x)
elif func == "relu":
    f = lambda x: np.maximum(x,0)
    g = lambda x: 0 if x <= 0 else 1


print("func is ", name)

N = np.zeros(11)
df = np.zeros(11)
err = np.zeros(11)
if func=="sigmoid" or func=="relu":
    for x in X:
        if func=="relu" and x <= 0: continue
        min_err = (0, 1000) #k, min
        #各hの数値における導関数の値を格納
        for k in range(1, 11):
            N[k] = 10 ** k
            if b == 64: h = np.array(1 / N[k])
            elif b == 32: h = np.array(1 / N[k]).astype("float32")
            
            if mode=="center": df[k] = (f(x+h) - f(x-h))/(2*h)
            elif mode=="reverse": df[k] = (f(x) - f(x-h))/h
            else: df[k] = (f(x+h) - f(x))/h
            #df[k] = numerical_gradient(f, x)
            #誤差の計算
            diff = abs(df[k] - g(x))
            if(diff < min_err[1]): min_err = (k,diff)
            err[k] = diff

        print("best result: x={:.2f}, h=1e-{}, min error={:e}".format(x, min_err[0], min_err[1]))

        #両対数プロット
        if plot:
            plt.plot(N, err, label=name+"_x={:.3f}".format(x))
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.xlabel('N')
            plt.ylabel('Error')
            #plt.ylim(ymax=1e-2)
            plt.legend()
    plt.savefig("numerical_analysis/{}.png".format(name))
else:
    for i in range(X.shape[0]):
        min_err = (0, 1000) #k, min
        W = np.random.normal(0, 0.4, (8,8))
        bias = np.random.normal(0, 0.4, 8)
        for k in range(1, 11):
            N[k] = 10 ** k
            if b == 64: h = np.array(1 / N[k])
            elif b == 32: h = np.array(1 / N[k]).astype("float32")
            args.eps = h
            f = _Affine(8,8, args=args)
            f2 = _Affine(8,8, args=args)
            f.weight = W
            f.bias = bias
            f2.weight = W
            f2.bias = bias
            dout = np.random.normal(0,0.4, (1,8))
            f(X[i])
            f2(X[i])
            b1 = f.backward(dout)
            b2 = f2.backward2(dout)
            print(i, "ana\n", b1, "\nnum\n", b2)
            
            #誤差の計算
            diff = np.mean(abs(b1-b2))
            if(diff < min_err[1]): min_err = (k,diff)
            err[k] = diff

        print("best result: i={}, h=1e-{}, min error={:e}".format(i, min_err[0], min_err[1]))

        #両対数プロット
        if plot:
            plt.plot(N, err, label=name+"_{}".format(i))
            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.xlabel('N')
            plt.ylabel('Error')
            #plt.ylim(ymax=1e-2)
            plt.legend()
        plt.savefig("numerical_analysis/{}.png".format(name))
