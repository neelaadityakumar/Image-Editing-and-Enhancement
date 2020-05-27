from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)
@app.route('/')
def upload_file():
    return render_template('upload.html')
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    def load(fname):
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.image as img
        import math as m
        import copy
        dic={}
        M = img.imread(fname)
        img.imsave('C:/Users/aditya/Desktop/cs300/static/'+fname, M)

        dic['M']=M
        W, H = M.shape[:2]
        dic['W']=W
        dic['H']=H
        NEW =  np.random.rand(W,H)
        dic['NEW']=NEW
        IMAG=  np.random.rand(W,H)
        dic['IMAG']=IMAG
        FB=(1.0/273)*np.array([ [1,4,7,4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1,4,7,4,1]
                        ])
        FS=(1.0)*np.array([
                        [0,-1,0],
                        [-1,5,-1],
                        [0,-1,0],

                        ])
        dic['FB']=FB
        dic['FS']=FS
        return dic
    def convertGray(fname):
        import copy
        import numpy as np
        d=load(fname)
        new=copy.deepcopy(d['NEW'])
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        m=copy.deepcopy(d['M'])
        for i in range(w):
            for j in range(h):
                avg =float((m[i][j][0]*0.299)+(m[i][j][1]*0.587)+(m[i][j][2]*0.114))
                #avg =(m[i][j][0]+m[i][j][1]+m[i][j][2])/3
                new[i][j] = avg
        return copy.deepcopy(new)
    def blur(fname):
        import numpy as np
        import copy
        d=load(fname)
        m=copy.deepcopy(d['M'])
        Xw=Xw=m.shape[1]
        Xh=m.shape[0]

        newss=np.zeros((Xh,Xw,3))
        F=copy.deepcopy(d['FB'])



        g=sum(sum(F))
        Fh=F.shape[0]
        Fw=F.shape[1]
        H=(Fh-1)//2
        W=(Fw-1)//2

        for i in range(H,Xh-H):
            for j in range(W,Xw-W):
                s1=0
                s2=0
                s3=0
                for k in range(-H,H+1):
                    for l in range(-W,W+1):
                        a1=m[i+k,j+l,0]
                        a2=m[i+k,j+l,1]
                        a3=m[i+k,j+l,2]
                        p1=F[H+k,W+l]
                        p2=F[H+k,W+l]
                        p3=F[H+k,W+l]
                        s1+=(p1*a1)
                        s2+=(p2*a2)
                        s3+=(p3*a3)
                newss[i,j,0]=s1
                newss[i,j,1]=s2
                newss[i,j,2]=s3
        return copy.deepcopy(newss/255)
    def sharpen(fname):
        import numpy as np
        import copy
        d=load(fname)
        m=copy.deepcopy(d['M'])
        Xw=Xw=m.shape[1]
        Xh=m.shape[0]

        newsss=np.zeros((Xh,Xw,3))
        F=copy.deepcopy(d['FS'])

        g=sum(sum(F))
        Fh=F.shape[0]
        Fw=F.shape[1]
        H=(Fh-1)//2
        W=(Fw-1)//2

        for i in range(H,Xh-H):
            for j in range(W,Xw-W):
                s1=0
                s2=0
                s3=0
                for k in range(-H,H+1):
                    for l in range(-W,W+1):
                        a1=m[i+k,j+l,0]
                        a2=m[i+k,j+l,1]
                        a3=m[i+k,j+l,2]
                        p1=F[H+k,W+l]
                        p2=F[H+k,W+l]
                        p3=F[H+k,W+l]
                        s1+=(p1*a1)
                        s2+=(p2*a2)
                        s3+=(p3*a3)
                newsss[i,j,0]=s1
                newsss[i,j,1]=s2
                newsss[i,j,2]=s3
        return copy.deepcopy(newsss)
    def histogramequalisation(fname):
        import copy
        import numpy as np

        new=convertGray(fname)
        pmf={}
        cmf={}
        d=load(fname)
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        for i in range(w):
            for j in range(h):
                if new[i][j] in pmf:
                    pmf[new[i][j]]+=1
                else:
                    pmf[new[i][j]]=1

        for i in pmf:
            pmf[i]=pmf[i]/(w*h)
        s=0
        for i in pmf:
            if s==0:
                s+=pmf[i]
                cmf[i]=pmf[i]
            else:
                s+=pmf[i]
                cmf[i]=s
        for i in cmf:
            cmf[i]=int(cmf[i]*255)
        newpmf={}
        for i in pmf:
            newpmf[cmf[i]]=pmf[i]*(w*h)

        pix={}
        for i in pmf:
            pix[i]=cmf[i]
        chan = np.random.rand(w,h)
        for i in range(w):
            for j in range(h):
                chan[i][j]=pix[new[i][j]]
        return copy.deepcopy(chan)
    def negative(fname):
        import numpy as np
        import copy
        from copy import deepcopy
        d=load(fname)
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        neg=np.zeros((w,h))
        new=convertGray(fname)
        for i in range(w):
            for j in range(h):
                neg[i][j]=255-new[i][j]
        return deepcopy(neg)
    def logtransform(fname):
        import numpy as np
        from copy import deepcopy
        import math as m
        import copy

        d=load(fname)
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        log=np.zeros((w,h))

        new=convertGray(fname)
        for i in range(w):
            for j in range(h):
                log[i][j]=0.004*m.log(new[i][j]+1)
        return deepcopy(log)
    def gamma(fname):
        import numpy as np
        import copy
        from copy import deepcopy
        d=load(fname)
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        gama=np.zeros((w,h))
        new=convertGray(fname)

        for i in range(w):
            for j in range(h):
                gama[i][j]=new[i][j]**1.5
        return deepcopy(gama)
    def linear(fname):
        import numpy as np
        import copy

        from copy import deepcopy
        d=load(fname)
        w=copy.deepcopy(d['W'])
        h=copy.deepcopy(d['H'])
        lin=np.zeros((w,h))
        new=convertGray(fname)
        for i in range(w):
            for j in range(h):
                lin[i][j]=new[i][j]*25+1
        return deepcopy(lin)
    def MedianFilter(image, filter_size):
        import numpy
        result = numpy.zeros((len(image),len(image[0])))
        filter_range = filter_size // 2
        garbage = []
        for i in range(len(image)):

            for j in range(len(image[0])):

                for z in range(filter_size):
                    if i + z - filter_range < 0 or i + z +filter_range > len(image) - 1:
                        for c in range(filter_size):
                            garbage.append(0)
                    else:
                        if j + z - filter_range < 0 or j + filter_range > len(image[0]) - 1:
                            garbage.append(0)
                        else:
                            for k in range(filter_size):
                                garbage.append(image[i + z - filter_range][j + k - filter_range])

                garbage.sort()
                result[i][j] = garbage[len(garbage) // 2]
                garbage = []
        return result
    def Mean(image, filter_size):
        import numpy
        result = numpy.zeros((len(image),len(image[0])))
        filter_range = filter_size // 2
        garbage = []
        for i in range(len(image)):

            for j in range(len(image[0])):

                for z in range(filter_size):
                    if i + z - filter_range < 0 or i + z +filter_range > len(image) - 1:
                        for c in range(filter_size):
                            garbage.append(0)
                    else:
                        if j + z - filter_range < 0 or j + filter_range > len(image[0]) - 1:
                            garbage.append(0)
                        else:
                            for k in range(filter_size):
                                garbage.append(image[i + z - filter_range][j + k - filter_range])

                m=sum(garbage)//(len(garbage))
                result[i][j] = m
                garbage = []
        return result

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    def BandB(new):
        s=0
        w, h = new.shape[:2]
        for i in range(w):
            for j in range(h):
                s+=new[i][j]
        threshold=s//(new.shape[0]*new.shape[1])
        for i in range(w):
            for j in range(h):
                if new[i][j]>=threshold:
                    new[i][j]=1
                else:
                    new[i][j]=0
        return new


    if request.method == 'POST':
        import copy
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.image as img
        import math as m
        import copy
        import cv2 as cv
        import io


        f = request.files['file']
        f.save(secure_filename(f.filename))
        if request.form['submit_button'] == 'Gray':
            new=convertGray(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'Gray'+str(f.filename), new,cmap='gray')
            full_filename='Gray'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Blur':
            new=blur(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'Blur'+str(f.filename), new)
            full_filename='Blur'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Sharpen Kernel':
            new=sharpen(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'S'+str(f.filename), new)
            full_filename='S'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Histogram Equalisation':
            new=histogramequalisation(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'Histo'+str(f.filename), new,cmap='gray')
            full_filename='Histo'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Linear Transformation':
            new=linear(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'linear'+str(f.filename), new,cmap='gray')
            full_filename='linear'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Gamma Transformation':
            new=gamma(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'gamma'+str(f.filename), new,cmap='gray')
            full_filename='gamma'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Negative Transformation':
            new=negative(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'negative'+str(f.filename), new,cmap='gray')
            full_filename='negative'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Log Transformation':
            new=logtransform(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'log'+str(f.filename), new,cmap='gray')
            full_filename='log'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Median Filtering':
            import numpy
            imgi = img.imread(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+str(f.filename), imgi)
            gray = rgb2gray(imgi)
            arr = numpy.array(gray)
            imgi = MedianFilter(arr, 3)
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'Med'+str(f.filename), imgi,cmap='gray')
            full_filename='Med'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Mean Filtering':
            import numpy
            imgi = img.imread(str(f.filename))
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+str(f.filename), imgi)
            gray = rgb2gray(imgi)
            arr = numpy.array(gray)
            imgi = Mean(arr, 3)
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'Mean'+str(f.filename), imgi,cmap='gray')
            full_filename='Mean'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))
        elif request.form['submit_button'] == 'Black & White':
            new=convertGray(str(f.filename))
            new=BandB(new)
            img.imsave('C:/Users/aditya/Desktop/cs300/static/'+'B&W'+str(f.filename), new,cmap='gray')
            full_filename='B&W'+str(f.filename)
            return render_template('image.html',name=full_filename,input=str(f.filename))


        return 'file uploaded successfully'
if __name__ == '__main__':
    app.run(debug = True)
