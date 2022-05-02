# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import pandas as pd
import numpy as np
import os
#Get the experimental groups(n)

for i in os.listdir(os.getcwd()):
    if i.endswith("xlsx"):
        data =  pd.read_excel(i)
a = data["Unnamed: 6"]
a = np.array(a)
# print(a)
#print(data)
import re
a = str(a)
a2 = re.findall( "\d+\.\d+",a)
# print(a2)
n=len(a2)/2
# print(n)
# print("sys.argv:",sys.argv)

#Get the The fluorescent intensity ratio (FIR) of 488nm/440nm excitation spectrums for calibration curve
df = pd.DataFrame()
for i in os.listdir(os.getcwd()):
    if i.endswith("xlsx"):
        # print(i)
        f =  pd.read_excel(i)
        a=f[f.columns[6]]
        a=a[9:9+int(2*n)]
        a1=np.array(a[::2])  #pick the numbers in odd positions
        a2=np.array(a[1::2]) #pick the numbers in even positions
        a3=a2/a1
        # print(a3)
        aa = pd.Series(a3)
        df = df.append(aa, ignore_index = True)
        #df.append(aa,ignore_index=True) 注意只使用df.append不改变原来的df
# print(df)


#Get the pHi values for the calibration curve
os.chdir('./PH')
path=os.getcwd()
for i in os.listdir(path):
    # print(i)
    if i.endswith("xlsx"):
        # print(i)
        df2 =  pd.read_excel(i)
df.columns=df2.columns

os.chdir(os.path.dirname(path)) ##return to the parent directory
# print(df)
# print(df2)

##Constructing the linear calibration curve equation about the relationship between FIR versus pHi
import statsmodels.api as sm
pv=[]
cv=[]
bv=[]

for i in df.columns:
    # print(i)
    x=df[i]
    y=df2[i]
    # #print(type(x))
    # #print(type(y))
    X=sm.add_constant(x)
    results = sm.OLS(y, X).fit()
    pv1=  float(np.array(results.pvalues)[1])
    cv1 = np.array(results.params)[0]
    bv1 = np.array(results.params)[1]
    # print(bv1)
    pv.append(pv1)
    cv.append(cv1)
    bv.append(bv1)
calibration_df = pd.DataFrame({
                    'beta': bv,
                    'const': cv,
                    'p_values': pv
})
df.to_csv('calibration_df.csv')

##Caculation of pHi recovery rate according to the calibration curve equations
os.chdir("./NDPR")
dfa = pd.DataFrame()
dfb = pd.DataFrame()
for i in os.listdir(os.getcwd()):
    if i.endswith("xlsx"):
        # print(i)
        f =  pd.read_excel(i)
        af=f[f.columns[6]]
        a=af[9:9+int(2*n)]
        a1=np.array(a[::2])  #pick the numbers in odd positions
        a2=np.array(a[1::2]) #pick the numbers in even positions
        a3=a2/a1     #FIR for pHi recovery rate
        aa = pd.Series(a3)
        dfa = dfa.append(aa, ignore_index = True)  #The FIR matrix for pHi recovery experiment

        bf = f[f.columns[7]]
        b=bf[9:9+int(2*n)]
        b1=np.array(b[::2])
        bb=pd.Series(b1)
        dfb = dfb.append(bb,ignore_index=True)  #Time matrix for pHi recovery experiment
# print(dfa)
# print(dfb)
# print(calibration_df)
import matplotlib.pyplot as plt
count = 0
const=[]
slope=[]
index=[]
def plott(ss=0):
    if ss>0:
        vv = pd.DataFrame({
            "t": t,
            "pHi": y
        }, index=np.arange(16))
        vv.plot.scatter(x='t', y='pHi')
        plt.show()
        
for i in calibration_df["p_values"]:
    count = count +1
    if i < 0.5:
        # print('p value of the calibration < 0.5 succeed in getting the pHi recovery rate of'+' '+str(df2.columns[count-1]))
        x = dfa.iloc[:, count-1]
        y = x*calibration_df["beta"][count-1]+calibration_df["const"][count-1]
        t = dfb.iloc[:,count-1]/60

        X = sm.add_constant(t)
        results = sm.OLS(y, X).fit()
        # print(results.params)
        params = results.params
        const_ = np.array(params)[0]
        slope_ = np.array(params)[1]
        index_ = df2.columns[count-1]
        const.append(const_)
        slope.append(slope_)        
        index.append(index_)
        plott(int(sys.argv[1]))
    else:
        print('p value of the calibration >= 0.5 fail to get the pHi recovery rate of' + ' ' + str(
            df2.columns[count - 1]))
dff=pd.DataFrame(
    {
    "pHi recovery rate":slope,
    "const":const
    },index=index
)
print(dff)
os.chdir(os.path.dirname(path))
dff.to_csv('results.csv')