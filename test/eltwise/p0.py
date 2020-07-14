#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001


import caffe
#载入solver文件
solver = caffe.SGDSolver('test_max_solver.prototxt')

solver.net.forward()
# solver.step(1)
#solver.solve()
solver.net.save("test_max.caffemodel")

print (solver.net.blobs['data1'].data.shape)
print (solver.net.blobs['data2'].data.shape)
print (solver.net.blobs['dataout'].data.shape)
