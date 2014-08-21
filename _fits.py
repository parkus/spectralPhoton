# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:33:34 2014

@author: Parke
"""

from astropy.io import fits

def filterfiles(files, criteria):
    """Rerturns only the files from filelist that match the criteria.
    
    The criteria should be a list of (unit,keyword,function) where the function
    returns true for desired values of the keyword. For example
    criteria = ((0,'OPT_ELEM', lambda x: x == 'G130M'),
                (1,'CENWAVE' , lambda x: x < 1300.0 and x > 1200.0))
    """
    outfiles = list()
    hdus = [c[0] for c in criteria]
    hdus = list(set(hdus)) #keeps the unique elements
    for h in hdus:
        #select the criteria associated with keywords in header h
        hcriteria = [c for c in criteria if c[0] == h]
        for f in files:
            hdr = fits.getheader(f,h)
            for c in hcriteria:
                value = hdr[c[1]]
                #keep file if the keyword value is what we want
                if c[2](value): outfiles.append(f)
                    
    return outfiles
                
def sortfiles(files, sortby='EXPSTART', hdu=1):
    """Order the fits files by a specific keyword value from the header.
    Currently only works for one level of sorting because I haven't yet needed
    to sort with more than one level of heirarchy.
    """
    keyvalues = [fits.getval(f,sortby,hdu) for f in files]
    fkey = zip(files, keyvalues)
    fkey.sort(key=lambda x: x[1])
    return [f[0] for f in fkey]
    