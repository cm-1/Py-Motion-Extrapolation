
I had the following thoughts/test written out when deriving the code that finds
the nearest rotation about a fixed axis. I removed it from the documentation and
testing code because some of it seems "obvious" or at least verbose in
retrospect, and the test (which passed) was replaced with others, but I also
don't want to delete this outright after I spent time writing it, in case it's
helpful if I need to look back on things.
 
```python

'''
Now, x = (x*w)w + x - (x*w)w, where (x*w)w is unaffected by R, and where
x - (x*w)w is orthogonal to w and can be rotated in 2D.

If theta is our angle of rotation, then our rotated component is:
cos(theta)(x - (x*w)w) + sin(theta)(something).

That something must be in the direction of cross(w, x - (x*w)w) and have
length |x - (x*w)w|. Well, cross(w, x - (x*w)w) already has that length,
and cross(w, x - (x*w)w) = cross(w, x) - cross(w, (x*w)w) = cross(w,x),
which is much simpler to write.


The distance of x from w is sqrt(1 - (xw)^2)
The distance between a 2D vector of length L and a rotation of theta degrees is
L*sqrt(2 - 2cos(theta))
So, the distance between x and Rx is sqrt[(1 - (xw)^2)(2 - 2cos(theta))].
Meaning the squared distance between x and Rx is (1 - (xw)^2)(2 - 2cos(theta)).
Meaning the total squared distance for all three vectors is:
(2 - 2cos(theta)) * [1 - (tw)^2 + 1 - (rw)^2 + 1 - (sw)^2]
= 2(2 - 2cos(theta))

Rx*r + Ry*s + Rz*s
 = (x*w)(r*w) + (y*w)(s*w) + (z*w)(t*w)
   + cos0(x*r + y*s + z*t - (x*w)(w*r) - (y*w)(w*s) - (z*w)(w*t)) 
   + sin0((x*w)(z*s - y*t) + (y*w)(x*t - z*r) + (z*w)(y*r - x*s))
'''
dot_sum = pm.einsumDot(newFrame.transpose(), targetFrame.transpose()).sum()

expectedSum = Xw_dot_Fw + np.linalg.norm([cos_component, sin_component])

if (np.abs(dot_sum - expectedSum) > 0.001):
    print(dot_sum, expectedSum)
    raise Exception("Difference!")
```