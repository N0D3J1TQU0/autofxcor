from astropy.io import fits
import os
import sys
import fileinput
import numpy as np
from astropy.table import Table, join
import astropy.stats as st
from astropy.stats import jackknife_stats
from astropy.constants import c
from astropy import units as u
from astropy.cosmology import LambdaCDM
from astropy.visualization import hist
from astropy.coordinates import SkyCoord
from statsmodels.stats.diagnostic import normal_ad as adtest
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import random
import multiprocessing as mp
import pickle
from scipy.signal import find_peaks 

#
# Defining constants
#
c = c.to(u.km/u.s).value
#
# Defining cosmology
#
mycosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7,  Tcmb0=0.0, Neff=0.0)
#
# Overall cut in km/s
vel_limit = 5000
# Table colnames: col1: object ID, col13: heliocentric velocity, col14: velocity error

#clusters
#cluster_lst = ["2344-4224","0151-5654","0144-4807","0600-4353","2358-6129","0451-4952","0354-5904","0439-5330","0337-4928","0111-5518","0135-5904","0522-5026","2100-5708","0612-4317","0550-5019"]
#z_phot
#zp_lst = [0.29,0.29,0.31,0.36,0.37,0.39,0.41,0.43,0.53,0.56,0.49,0.52,0.57,0.54,0.65]
#masks
#msk_lst = ["01_02","03_04","05_06","07_08","09_10","11_12","13_14","17_18","33_34","37_38","25_26","31_32","39_40","35_36","41_42"] 
#this file contains the sequence of steps to measure redshifts on an intensive semi-automatic basis with fxcor. 
#this table shows the round's sample regions and prominent lines in each of them.
lns = np.array(["all","4000A,G-band,Mg,Na","4000A,G-band,Mg","G-band,Mg,Na","4000A,G-band","G-band,Mg","Mg,Na","4000A,G-band","Mg","Na","Ha,[SII]"])
rcover = np.array(["all","3500-6000","3500-5300","4200-6000","3500-4600","4200-5300","4820-5920","3800-4400","4800-5400","5700-6300","6500-6780"])
spectab = Table([np.array(range(0,11)),rcover,lns])
#recheck_as2.cat should contain a csv table with the extension of unviable spectra in the col1
#bn = Table.read("recheck_as2.cat",format="csv")


#0) hacer tmp.lst y obj.lst, ojo con los path...
mv redshift_fxcor redshift_fxcor5
mkdir redshift_fxcor
cd redshift_fxcor
mkdir fxcor_autosample2
cd fxcor_autosample2
ls ../../fits/*clean.fits | grep spt > obj.lst
ls ../../../temps/eltemp.fits | grep .fits >> tmp.lst
ls ../../../temps/habtemp0.fits | grep .fits >> tmp.lst
ls ../../../temps/sptemp.fits | grep .fits >> tmp.lst
ls ../../../temps/syn4.fits | grep .fits >> tmp.lst
cp ../../redshift_fxcor1/fxcor_autosample2/recheck_as2.txt .
mv recheck_as2.txt recheck_as2.cat

#1) hacer lst.cl con las lineas de codigo para iraf
t = open("lst.cl","w")
bn = Table.read("recheck_as2.cat",format="csv")     #recheck contains unviable spectra identified when lineclean
print("noao")
t.write("noao\n")
print("rv")
t.write("rv\n")
print("rv.z_threshold = 1")
t.write("rv.z_threshold = 1\n")
fileinput.close()
for line in fileinput.input("obj.lst"):
    obj = line.split("clean")[0]+"clean.fits"
    ext = line.split("_")[1]
    if int(ext) in bn["col1"]:
        continue 
    t.write("fxcor ")
    t.write(str(obj))
    t.write(" @tmp.lst continuum=\"both\" filter=\"none\" rebin=\"object\" background=INDEF output=")
    t.write(str(ext))
    t.write(" imupdate=no interactive=no continpars.c_function=\"chebyshev\" continpars.order=13\n")

t.close()

#1.1) correr lst.cl en iraf y guardar la info de los .txt en fxcor_####.txt
cl < lst.cl
cat *.txt > fxcor.txt

#2) leer fxcor.txt como tabla ascii y separar resultados ideales y no-ideales.  can: tabla ideales con promedio y propagacion de error
z = Table.read("fxcor.txt",format="ascii")
indef = []
lst = []
can = Table([[""],[""],[1.1],[1.1],[""]],names=["col1","col2","col13","col14","round"],dtype=["<U15","<U37","float64","float64","<U4"])
Nr = str(spectab["col0"][0])+"_A" 
for k in range(int(len(z)/4)):
    dts = z[(4*k):((4*k)+4)]
    zind = 0 
    try:
        if "INDEF" in dts["col13"]:
            dts = dts[dts["col13"]!="INDEF"]
        if float(dts["col13"][0]) <= 1 and float(dts["col13"][0]) >= 0: 
            zind=1
        zind = c**zind 
        mvhel = np.median(np.array(dts["col13"],dtype="float")*zind)
        dts = dts[np.array(dts["col13"],dtype="float")*zind <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float")*zind >= (mvhel-100)]
        if len(dts)<=2:
            dts = z[(4*k):((4*k)+4)] 
            for i in range(4):
                indef += [np.array((dts[i]))]
            continue
        dts["col13"] = dts["col13"].astype("float")
        dts.sort("col13")
        if float(dts["col13"][0])*zind <= 25000 or float(dts["col13"][len(dts)-1])*zind >= 300000:
            dts = z[(4*k):((4*k)+4)]
            for i in range(4):
                indef += [np.array((dts[i]))]
            continue
        top_err = len(dts[np.array(dts["col14"],dtype="float")>=350])
        bottom_err = len(dts[np.array(dts["col14"],dtype="float")<=5])
        if top_err != 0 or bottom_err != 0:
            dts = z[(4*k):((4*k)+4)]
            for i in range(4):
                indef += [np.array((dts[i]))]
            continue
        dts["col10"] = dts["col10"].astype("float")
        dts.sort("col10")
        if float(dts["col10"][len(dts)-2])<5:
            dts = z[(4*k):((4*k)+4)]
            for i in range(4):
                indef += [np.array((dts[i]))]
            continue
        prom = np.mean(np.array(dts["col13"],dtype="float")*zind)
        err = np.mean(np.array(dts["col14"],dtype="float"))
        obj = dts["col1"][0]
        ext = dts["col2"][0]
        res = Table([[obj],[ext],[prom],[err],[Nr]],names=("col1","col2","col13","col14","round"))
        can.add_row(res[0])
        lst += [np.array((dts[len(dts)-1]))]
    except:
        for i in range(len(dts)):
           indef += [np.array((dts[i]))]
        if len(dts)!=4:
            print("ERROR: dts with size != 4 into indef list")
lst = Table(np.array(lst))
indef = Table(np.array(indef))
can.remove_row(0)
lst.write("fxcor.cat",format="csv")
can.write("fxcor_can.cat",format="csv")


#3) revisar si quedaron estrellas en la lista de resultados ideales (mas vale asegurarse)
bn = Table.read("recheck_as2.cat",format="csv")
print("ext,index")
for i in range(len(lst)): 
    ext = lst["col2"][i].split("_")[1] 
    if int(ext) in bn["col1"]: 
        print(str(ext)+","+str(i)) 

#can.remove_row(index)           
#lst.remove_row(index)  

#4) generar linea de codigo para iraf lst1.cl, sample regions en base a mediana de can["col13"] (assuming all galaxies are near the cluster redshift, then extend the range). 

z_init = np.median(can["col13"])/c 
t = open("lst1.cl","w") 
t.write("noao\n") 
t.write("rv\n") 
t.write("rv.z_threshold = 1\n") 
for z_init in [z_init,z_init-0.2,z_init+0.2]:
    for i in range(len(spectab)):
        if i == 0:
            continue
        Nr = int(spectab["col0"][i])
        zanch = 0 
        if Nr == 10:
            Nr = 0
            zanch = 0.2 
        rsamp1 = int(spectab["col1"][i].split("-")[0])
        rsamp2 = int(spectab["col1"][i].split("-")[1])
        fileinput.close()
        for line in fileinput.input("obj.lst"):
            obj = line.split("clean")[0]+"clean.fits"
            ext = line.split("_")[1]
            if int(ext) in bn["col1"]:
                continue
            t.write("fxcor ")
            t.write(str(obj))
            t.write(" @tmp.lst continuum=\"both\" filter=\"none\" rebin=\"object\" osample=\"")
            t.write(str(int(rsamp1*(1+z_init-zanch))))
            t.write("-")
            t.write(str(int(rsamp2*(1+z_init+zanch)))+"\"")
            t.write(" rsample=\"")
            t.write(str(rsamp1))
            t.write("-")
            t.write(str(rsamp2)+"\"")
            t.write(" background=INDEF output=\"")
            t.write(str(ext)+"_r"+str(Nr)+"_rch")
            t.write("\" imupdate=no interactive=no continpars.c_function=\"chebyshev\" continpars.order=13\n")

t.close()

#4.1) correr in iraf y guardar los resultados
cl < lst1.cl    #tarda alrededor de 10 min
cat *rch.txt > fxcor_rch.txt


#5) separar resultados en ideales y no-ideales y asignar ronda.

z_rch = Table.read("fxcor_rch.txt",format="ascii")
indef_rch = []
lst_rch = []
can_rch = Table([[""],[""],[1.1],[1.1],[""]],names=["col1","col2","col13","col14","round"],dtype=["<U15","<U37","float64","float64","<U4"])
cluster = z_rch["col2"][0].split("t")[2].split("_")[0]
for k in range(100):
    dtsi = z_rch[z_rch["col2"]=="../../fits/spt"+cluster+"_"+str(k)+"_clean.fits"]
    if len(dtsi)==0:
        continue
    for p in ["B","C","D"]: 
        if p == "B":
            P=0
        if p == "C":
            P=1
        if p == "D":
            P=2 
        for j in spectab["col0"][1:11]:
            dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4]     
            zind = 0
            try:
                if j == 1 and float(dts[dts["col3"]=="C"]["col10"][0])>=15 and float(dts[dts["col3"]=="C"]["col10"][0])<30:
                    dts = Table(dts[dts["col3"]=="C"]) 
                    for i in range(3):
                        dts.add_row(dts[0])
                if "INDEF" in dts["col13"]:
                    dts = dts[dts["col13"]!="INDEF"]
                if float(dts["col13"][0]) <= 1 and float(dts["col13"][0]) >= 0:
                    zind = 1
                zind = c**zind
                mvhel = np.median(np.array(dts["col13"],dtype="float")*zind)
                dts = dts[np.array(dts["col13"],dtype="float")*zind <= (mvhel+100)]
                dts = dts[np.array(dts["col13"],dtype="float")*zind >= (mvhel-100)]
                if len(dts)<=2:
                    dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4]        
                    for i in range(4):
                        indef_rch += [np.array((dts[i]))]
                    continue
                dts["col13"] = dts["col13"].astype("float")  
                dts.sort("col13")
                if float(dts["col13"][0])*zind <= 25000 or float(dts["col13"][len(dts)-1])*zind >= 300000:
                    dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4]    
                    for i in range(4):
                        indef_rch += [np.array((dts[i]))]
                    continue
                top_err = len(dts[np.array(dts["col14"],dtype="float")>=350])
                bottom_err = len(dts[np.array(dts["col14"],dtype="float")<=5])
                if top_err != 0 or bottom_err != 0:
                    dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4] 
                    for i in range(4):
                        indef_rch += [np.array((dts[i]))]
                    continue
                dts["col10"] = dts["col10"].astype("float")  
                dts.sort("col10")
                if float(dts["col10"][len(dts)-2])<5:
                    dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4]
                    for i in range(4):
                        indef_rch += [np.array((dts[i]))]
                    continue
                prom = np.mean(np.array(dts["col13"],dtype="float")*zind)
                err = np.mean(np.array(dts["col14"],dtype="float"))
                obj = dts["col1"][0]
                ext = dts["col2"][0]
                Nr = str(j-1)+"_"+p 
                res = Table([[obj],[ext],[prom],[err],[Nr]],names=("col1","col2","col13","col14","N_R"))
                can_rch.add_row(res[0])
                lst_rch += [np.array((dts[len(dts)-1]))]
            except:
                #dts = dtsi[((12*(j-1))+(P*4)):((12*(j-1))+(P*4))+4]
                for i in range(len(dts)):
                    indef_rch += [np.array((dts[i]))]
                if len(dts)!=4:
                    print("ERROR: dts with size "+str(len(dts))+" into indef list: "+str(dtsi["col2"][0].split("_")[1])+" - "+str(j)+"_"+p)
lst_rch = Table(np.array(lst_rch))
indef_rch = Table(np.array(indef_rch))
can_rch.remove_row(0)

#5.1) añadir resultados a la lista de resultados ideales
for i in range(len(can_rch)): 
    can.add_row(can_rch[i]) 
for i in range(len(lst_rch)): 
    lst.add_row(lst_rch[i]) 

#lst.write("fxcor.cat",format="csv")
#can.write("fxcor_can.cat",format="csv")

#6.A) autodemocracy: comparar resultados de las rondas, ver que esten en +-100 de la mediana y tener cuidado con los resultados de las rondas N_R>=8.

can = Table.read("fxcor_can.cat",format="csv")
lst = Table.read("fxcor.cat",format="csv")
crt = Table([[""],[""],[1.1],[1.1],[""],[1]],names=["col1","col2","col13","col14","round","backers"],dtype=["<U15","<U37","float64","float64","<U4","int64"])
cluster = can["col2"][0].split("t")[2].split("_")[0]
plt.ioff()
for i in range(100):
    dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
    if len(dts) == 0:
        continue
    upper = (np.around((np.max(dts["col13"])/200))+2)*200  
    lower = (np.around(np.min(dts["col13"])/200)-2)*200  
    nbin = int((upper-lower)/200)
    h = plt.hist(dts["col13"],bins=nbin,range=[lower,upper])
    plt.close() 
    ter = [] 
    for j in range(len(h[0])):
        if h[0][j]==max(h[0]) or j in find_peaks(h[0])[0]:
            l = h[1][j]
            u = h[1][j+1]
            vpkts = (u+l)/2 
            ter += [vpkts] 
    tere = [] 
    for k in range(len(ter)):
        mvhel = ter[k] 
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        for r in range(4): 
            mvhel = np.median(dts["col13"]) 
            dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
            dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
            dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        if len(dts)!=1 and np.std(dts["col13"])==0.0:
            continue
        dts.sort("round")
        tere += [(dts["round"][0],len(dts),mvhel,dts["col14"][0])]
    tere = Table(np.array(tere),dtype=["<U3","int64","float64","float64"]) 
    if len(tere)==1:
        mvhel = tere["col2"][0] 
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
        crt.add_row(dts[0])
        continue 
    tere.sort("col1")
    if tere["col1"][len(tere)-1] == tere["col1"][len(tere)-2]:
        tere = tere[tere["col1"]==tere["col1"][len(tere)-1]]
        tere.sort("col0")
        if tere["col0"][len(tere)-1] == tere["col0"][len(tere)-2]: 
            tere = tere[tere["col0"]==tere["col0"][len(tere)-1]]
            tere.sort("col3")
            mvhel = tere["col2"][0]
            dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
            dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
            dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
            dts.sort("round")
            dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
            crt.add_row(dts[0])
            continue
        mvhel = tere["col2"][0]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
        crt.add_row(dts[0])
        continue
    tere.sort("col0")
    if tere["col1"][0] >= 3 and tere["col0"][0] == "0_A": 
        mvhel = tere["col2"][0]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"] 
        crt.add_row(dts[0])
        continue 
    tere.sort("col1")
    mvhel = tere["col2"][len(tere)-1]
    dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
    dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
    dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
    dts.sort("round")
    dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"] 
    crt.add_row(dts[0])
crt.remove_row(0)

#6.B interactive Oligarchy <-- this is an alternative helpful when analyzing problematic clusters

can = Table.read("fxcor_can.cat",format="csv")
lst = Table.read("fxcor.cat",format="csv")
crt = Table([[""],[""],[1.1],[1.1],[""],[1]],names=["col1","col2","col13","col14","round","backers"],dtype=["<U15","<U37","float64","float64","<U4","int64"])
cluster = can["col2"][0].split("t")[2].split("_")[0]
cut_lst = [20.6,21.4,21.16,21.3,21.5]
mgcut = cut_lst[4]
plt.ioff()
for i in range(100):
    crt.write("fxcor_temprom.cat",format="csv",overwrite=True) 
    dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
    if len(dts) == 0:
        continue
    hdul = fits.open("../../fits/spt"+cluster+"_"+str(i)+"_clean.fits") 
    hdr = hdul[0].header 
    try:
        DESM = float(hdr["DES_MAG"])
    except:
        DESM = 30.0
    upper = (np.around((np.max(dts["col13"])/200))+2)*200
    lower = (np.around(np.min(dts["col13"])/200)-2)*200
    nbin = int((upper-lower)/200)
    h = plt.hist(dts["col13"],bins=nbin,range=[lower,upper])
    plt.close()
    ter = []
    for j in range(len(h[0])):
        if h[0][j]==max(h[0]) or j in find_peaks(h[0])[0]:
            l = h[1][j]
            u = h[1][j+1]
            vpkts = (u+l)/2
            ter += [vpkts]
    tere = []
    for k in range(len(ter)):
        mvhel = ter[k]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        for r in range(4):
            mvhel = np.median(dts["col13"])
            dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
            dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
            dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        if len(dts)!=1 and np.std(dts["col13"])==0.0:
            continue
        dts.sort("round")
        tere += [(dts["round"][0],len(dts),mvhel,dts["col14"][0])]
    if len(tere)==0:
        continue
    tere = Table(np.array(tere),dtype=["<U3","int64","float64","float64"])
    if len(tere)==1 and DESM < mgcut:
        mvhel = tere["col2"][0]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
        crt.add_row(dts[0])
        continue
    tere.sort("col1")
    if tere["col1"][len(tere)-1] == tere["col1"][len(tere)-2] and DESM < mgcut:
        tere = tere[tere["col1"]==tere["col1"][len(tere)-1]]
        tere.sort("col0")
        if tere["col0"][len(tere)-1] == tere["col0"][len(tere)-2]:
            tere = tere[tere["col0"]==tere["col0"][len(tere)-1]]
            tere.sort("col3")
            mvhel = tere["col2"][0]
            dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
            dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
            dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
            dts.sort("round")
            dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
            crt.add_row(dts[0])
            continue
        mvhel = tere["col2"][0]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
        crt.add_row(dts[0])
        continue
    if DESM < mgcut:
        mvhel = tere["col2"][len(tere)-1]
        dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
        dts = dts[np.array(dts["col13"],dtype="float") <= (mvhel+100)]
        dts = dts[np.array(dts["col13"],dtype="float") >= (mvhel-100)]
        dts.sort("round")
        dts["backers"] = Table(np.reshape(np.ones(len(dts),dtype="int64")*len(dts),[len(dts),1]))["col0"]
        crt.add_row(dts[0])
        continue
    dts = can[can["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
    dtsl = lst[lst["col2"]=="../../fits/spt"+cluster+"_"+str(i)+"_clean.fits"]
    print("CHECK SPECTRA") 
    print("splot ../../fits/spt"+cluster+"_"+str(i)+"_clean.fits")
    print(dts)
    print(dtsl)
    for k in range(len(dts)):
        shrald = (dts["col13"][k]/c)+1
        print("VEL "+str(k))
        print("======")
        print("[OII] 3727.261 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(3727.261*shrald,3)))
        print("K 3933.363 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(3933.363*shrald,3)))
        print("H 3967.797 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(3967.797*shrald,3)))
        print("G-band 4304.4 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(4304.4*shrald,3)))
        print("Hb 4861 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(4861*shrald,3))) 
        print("Mg 5170.195 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(5170.195*shrald,3)))
        print("Na 5892.372 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(5892.372*shrald,3)))
        print("Halpha 6559.851 ---*"+str(np.around(shrald,4))+"---> "+str(np.around(6559.851*shrald,3)))
    res0 = input("INDEX OF CANNED SPECTRA TO ADD TO CRATE (\"no\" for none): ")
    if res0=="no":
        res1 = input("MANUAL VEL (\"low\" for low-signal): ")
        if res1=="low":
            bn.add_row([i,"low-signal"]) 
            continue 
        prom = float(res1) 
        err = np.median(np.array(dts["col14"],dtype="float"))
        obj = dts["col1"][0]
        ext = dts["col2"][0]
        print(spectab) 
        print("Z_CLUSTER:"+str(z_init-0.2))
        Nr = input("ROUND: ")
        feat = input("NUMBER OF FEATURES IN SPECTRA: ")
        res = Table([[obj],[ext],[prom],[err],[Nr],[feat]],names=("col1","col2","col13","col14","N_R","backers"))
        crt.add_row(res[0])
        continue 
    resi = Table(dts[int(res0)]) 
    feat = input("NUMBER OF FEATURES IN SPECTRA: ")
    resi["backers"] = [feat]
    crt.add_row(resi[0])
    hdul.close()
crt.remove_row(0)



#crt.write("fxcor_prom.cat",format="csv")

for i in list(range(1,41))+list(range(51,84)):   #if there are still non ideal results, send them to recheck as low signals (check spectra interactively first)
    obj = "../../fits/spt"+cluster+"_"+str(i)+"_clean.fits" 
    if obj not in crt["col2"] and int(i)!=50 and int(i) not in bn["col1"]: 
        print(i) 
        #bn.add_row([i,"low-signal"])
#bn.write("recheck_as2.cat",format="csv")


#dts = indef_rch[indef_rch["col2"]=="../../fits/spt"+cluster+"_"+str(19)+"_clean.fits"]    

 
#7) obtener coordenadas y añadirlas a crt, ordenar por ext y cambiar colnames, añadir Z_HEL y Z_ERR, añadir RA DEC y MAG.
crt = Table.read("fxcor_prom.cat",format="csv")
crt["Z_HEL"] = Table(np.reshape(np.around(crt["col13"]/c,6),[len(crt),1]))["col0"]
crt["Z_ERR"] = Table(np.reshape(np.around(crt["col14"]/c,6),[len(crt),1]))["col0"]    
ral = [] 
decl = []
magl = [] 
for i in crt["col2"]: 
    ext = i.split("_")[1] 
    header = fits.getheader(i) 
    ra, dec , mag= header['RAOBJ'], header['DECOBJ'], header['DES_MAG']     #DES_MAG taken from DESY3, added to headers with IRAF hedit
    ral += [np.around(ra*15,5)] 
    decl += [dec]
    magl += [mag] 
ral = Table([ral],names=(["RA"]),dtype=(["float"])) 
decl = Table([decl],names=["DEC"],dtype=(["float"])) 
magl = Table([magl],names=["MAG"],dtype=(["float"]))
crt["RA"] = ral["RA"]
crt["DEC"] = decl["DEC"]
crt["MAG"] = magl["MAG"]
crt.replace_column("col13",Table(np.reshape(np.array(np.around(crt["col13"]),dtype="int"),[len(crt),1]))["col0"])
crt.replace_column("col14",Table(np.reshape(np.array(np.around(crt["col14"]),dtype="int"),[len(crt),1]))["col0"])                    
crt.sort("col2")
crt.rename_column("col1","ID")     
crt.rename_column("col2","EXT")
crt.rename_column("col13","V_HEL")
crt.rename_column("col14","V_ERR")

#crt.write("fxcor_prom.cat",format="csv")
#crt.write("../../../cluster_redshifts/fxcor_"+str(cluster.split("-")[0])+"_prom.cat",format="csv")

#) edit spectra to eliminate 7620A line
mkdir original_fits
mv *clean.fits original_fits
cluster="2344-4224"
for j in range(100):
    try:
        hdul = fits.open("original_fits/spt"+cluster+"_"+str(j)+"_clean.fits")
    except:
        continue 
    data = hdul[0].data
    wave = hdul[0].header["CRVAL1"] + ((hdul[0].header["CDELT1"])*np.array(range(len(hdul[0].data))))
    spectra = Table([wave,hdul[0].data])
    cuter = spectra[spectra["col0"]<7670]
    cuter = cuter[cuter["col0"]>7580]
    for i in range(len(spectra)):
        if spectra["col0"][i] == min(cuter["col0"]):
            min_index = i
        if spectra["col0"][i] == max(cuter["col0"]):   
            max_index = i
    data[min_index:max_index] = data[min_index:max_index][0] + ((data[min_index:max_index][1] - data[min_index:max_index][0])/len(data[min_index:max_index]-1))*np.array(range(len(data[min_index:max_index])))
    hdul.writeto("spt"+cluster+"_"+str(j)+"_clean.fits",output_verify="ignore")
    hdul.close()

             














